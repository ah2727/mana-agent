"""Repository-isolated, evidence-backed adaptive skills.

This module deliberately keeps generated procedures outside a checkout.  It is
also intentionally independent from the legacy static ``SkillManager``: static
skills remain compatible while adaptive skills have a validated lifecycle.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mana_agent.workspaces.paths import mana_home

SCHEMA_VERSION = 1
SCOPES = ("bundled", "global", "workspace", "repository")
STATUSES = ("candidate", "active", "stale", "blocked", "needs_review", "archived", "rejected")
_SKILL_ID = re.compile(r"^skill_[a-z0-9][a-z0-9_-]{5,80}$")
_REPO_ID = re.compile(r"^repo_[a-z0-9]{8,80}$")
_SECRET = re.compile(r"(?i)(?:api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^\s'\"]{8,}")
_DANGEROUS = ("rm -rf", "git push --force", "git reset --hard", "curl | sh", "chmod 777", "sudo ")


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def skills_root(root: str | Path | None = None) -> Path:
    """Resolve the adaptive root from MANA_SKILLS_ROOT, MANA_HOME, or ~/.mana."""
    configured = str(os.getenv("MANA_SKILLS_ROOT") or "").strip()
    return (Path(configured).expanduser().resolve() if configured else Path(root or mana_home()).expanduser().resolve() / "skills")


class RepositoryIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: int = SCHEMA_VERSION
    repository_id: str
    display_name: str
    canonical_remote: str | None = None
    known_paths: list[str] = Field(default_factory=list)
    first_seen_at: str = Field(default_factory=utcnow)
    last_seen_at: str = Field(default_factory=utcnow)
    current_commit: str | None = None
    workspace_ids: list[str] = Field(default_factory=list)


class SkillPermissions(BaseModel):
    filesystem: Literal["deny", "allow", "ask", "restricted"] = "restricted"
    shell: Literal["deny", "allow", "ask", "restricted"] = "restricted"
    network: Literal["deny", "allow", "ask", "restricted"] = "deny"
    git_commit: Literal["deny", "allow", "ask", "restricted"] = "ask"
    git_push: Literal["deny", "allow", "ask", "restricted"] = "deny"
    secrets: Literal["deny", "allow", "ask", "restricted"] = "deny"


class SkillManifest(BaseModel):
    """Versioned machine-readable contract for a generated procedure."""
    schema_version: int = SCHEMA_VERSION
    id: str
    name: str
    title: str
    description: str
    version: str = "1.0.0"
    scope: Literal["bundled", "global", "workspace", "repository"] = "repository"
    status: Literal["candidate", "active", "stale", "blocked", "needs_review", "archived", "rejected"] = "candidate"
    repository: dict[str, Any] = Field(default_factory=dict)
    origin: dict[str, Any] = Field(default_factory=dict)
    applicability: dict[str, Any] = Field(default_factory=dict)
    permissions: SkillPermissions = Field(default_factory=SkillPermissions)
    verification: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    lifecycle: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _SKILL_ID.fullmatch(value):
            raise ValueError("skill id must be a stable skill_<id> value")
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,80}", value):
            raise ValueError("skill name must be a slug")
        return value


class SkillEvidence(BaseModel):
    schema_version: int = SCHEMA_VERSION
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    repository_id: str
    repository_commit: str | None = None
    supporting_steps: list[dict[str, Any]] = Field(default_factory=list)
    successful_commands: list[str] = Field(default_factory=list)
    failed_commands_and_recoveries: list[dict[str, Any]] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    user_corrections: list[str] = Field(default_factory=list)
    redactions: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class SecurityFinding:
    severity: Literal["info", "low", "medium", "high", "critical"]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    skill_id: str
    selected: bool
    reason: str
    signals: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    policy_result: str = "allowed"
    scope_resolution: str = ""


class RepositoryIdentityService:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = skills_root(root)

    @staticmethod
    def normalize_remote(value: str) -> str:
        value = str(value or "").strip()
        value = re.sub(r"^[a-z]+://[^@/]+@", "", value, flags=re.I)
        value = re.sub(r"^[a-z]+://", "", value, flags=re.I)
        value = re.sub(r"^[^@/]+@", "", value)
        return value.removesuffix("/").removesuffix(".git").lower()

    @staticmethod
    def _git(root: Path, *args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    def identify(self, repository: str | Path) -> RepositoryIdentity:
        path = Path(repository).expanduser().resolve()
        git_root = self._git(path, "rev-parse", "--show-toplevel")
        root = Path(git_root).resolve() if git_root else path
        remote = self.normalize_remote(self._git(root, "remote", "get-url", "origin")) or None
        # Remote identities deliberately do not include a local path. Local repos
        # use a stable git root/initial commit fingerprint rather than machine paths.
        initial = self._git(root, "rev-list", "--max-parents=0", "HEAD")
        material = f"remote:{remote}" if remote else f"local:{root.name}:{initial or self._git(root, 'rev-parse', '--git-dir')}"
        repository_id = "repo_" + hashlib.sha256(material.encode()).hexdigest()[:20]
        directory = self.root / "repositories" / repository_id
        metadata_path = directory / "metadata.json"
        previous: dict[str, Any] = {}
        if metadata_path.exists():
            try: previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): pass
        paths = list(dict.fromkeys([*previous.get("known_paths", []), str(root)]))
        identity = RepositoryIdentity(
            repository_id=repository_id, display_name=root.name, canonical_remote=remote,
            known_paths=paths, first_seen_at=previous.get("first_seen_at", utcnow()),
            last_seen_at=utcnow(), current_commit=self._git(root, "rev-parse", "HEAD") or None,
            workspace_ids=list(previous.get("workspace_ids", [])),
        )
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(metadata_path, identity.model_dump(mode="json"))
        return identity

    def relink(self, repository_id: str, repository: str | Path) -> RepositoryIdentity:
        identity = self.identify(repository)
        if identity.repository_id != repository_id:
            raise ValueError("repository identity conflict: remote/history does not match requested id")
        return identity


class SkillSecurityScanner:
    def scan(self, markdown: str, manifest: SkillManifest, root: Path) -> list[SecurityFinding]:
        findings: list[SecurityFinding] = []
        text = markdown + "\n" + json.dumps(manifest.model_dump(mode="json"))
        if _SECRET.search(text): findings.append(SecurityFinding("critical", "secret", "Candidate appears to contain a secret."))
        if re.search(r"(?m)(?:^|\s)/(?:Users|home)/[^\s]+", text): findings.append(SecurityFinding("high", "absolute-user-path", "Candidate contains an absolute user path."))
        for command in _DANGEROUS:
            if command in text.lower(): findings.append(SecurityFinding("high", "dangerous-command", f"Unsafe instruction detected: {command}"))
        if "ignore previous instructions" in text.lower() or "bypass policy" in text.lower(): findings.append(SecurityFinding("critical", "prompt-injection", "Candidate attempts to bypass instructions or policy."))
        if ".." in " ".join(manifest.applicability.get("related_files", [])):
            findings.append(SecurityFinding("high", "path-traversal", "Candidate references path traversal."))
        if manifest.permissions.network != "deny": findings.append(SecurityFinding("medium", "network", "Candidate requests network access."))
        return findings


class SkillPolicyEngine:
    _rank = {"deny": 0, "restricted": 1, "ask": 2, "allow": 3}
    def effective_permissions(self, requested: SkillPermissions, *policies: SkillPermissions) -> SkillPermissions:
        all_policies = (requested, *policies)
        values = {field: min((getattr(item, field) for item in all_policies), key=self._rank.get) for field in SkillPermissions.model_fields}
        return SkillPermissions(**values)

    def explain(self, manifest: SkillManifest, *policies: SkillPermissions) -> dict[str, dict[str, str]]:
        effective = self.effective_permissions(manifest.permissions, *policies)
        return {key: {"requested": getattr(manifest.permissions, key), "effective": getattr(effective, key)} for key in SkillPermissions.model_fields}


class SkillValidator:
    REQUIRED_SECTIONS = ("Use when", "Do not use when", "Required context", "Procedure", "Verification", "Failure recovery")
    def __init__(self, scanner: SkillSecurityScanner | None = None) -> None: self.scanner = scanner or SkillSecurityScanner()
    def validate(self, markdown: str, manifest: SkillManifest, evidence: SkillEvidence, root: Path) -> list[SecurityFinding]:
        if manifest.schema_version != SCHEMA_VERSION or evidence.schema_version != SCHEMA_VERSION: raise ValueError("unsupported skill schema version")
        if manifest.scope == "repository" and manifest.repository.get("id") != evidence.repository_id: raise ValueError("repository identity mismatch")
        missing = [section for section in self.REQUIRED_SECTIONS if f"## {section}" not in markdown]
        if missing: raise ValueError("SKILL.md missing sections: " + ", ".join(missing))
        if not evidence.verification_results: raise ValueError("candidate requires verification evidence")
        return self.scanner.scan(markdown, manifest, root)


class SkillStorage:
    def __init__(self, root: str | Path | None = None) -> None: self.root = skills_root(root)
    def repository_dir(self, repository_id: str) -> Path: return self.root / "repositories" / repository_id
    def storage_path(self) -> Path: return self.root
    def _candidate_dir(self, repository_id: str, skill_id: str) -> Path: return self.repository_dir(repository_id) / "candidates" / skill_id
    @contextmanager
    def lock(self, repository_id: str, name: str = "repository"):
        lock = self.repository_dir(repository_id) / "locks" / f"{name}.lock"; lock.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 5
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); break
            except FileExistsError:
                if time.monotonic() > deadline: raise TimeoutError(f"skill lock busy: {lock}")
                if lock.exists() and time.time() - lock.stat().st_mtime > 300: lock.unlink(missing_ok=True)
                time.sleep(.05)
        try: yield
        finally: lock.unlink(missing_ok=True)
    def write_candidate(self, markdown: str, manifest: SkillManifest, evidence: SkillEvidence, findings: Iterable[SecurityFinding]) -> Path:
        repository_id = evidence.repository_id
        with self.lock(repository_id, manifest.id):
            target = self._candidate_dir(repository_id, manifest.id)
            if target.exists(): raise FileExistsError(f"candidate already exists: {manifest.id}")
            staging = self.repository_dir(repository_id) / "temporary" / f"{manifest.id}-{uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=False)
            try:
                _atomic_text(staging / "SKILL.md", markdown)
                _atomic_json(staging / "manifest.yaml", manifest.model_dump(mode="json"))
                _atomic_json(staging / "evidence.json", evidence.model_dump(mode="json"))
                _atomic_json(staging / "security.json", [item.__dict__ for item in findings])
                _append_audit(staging / "audit.jsonl", {"event": "skill.candidate.created", "at": utcnow()})
                target.parent.mkdir(parents=True, exist_ok=True); os.replace(staging, target)
            except Exception: shutil.rmtree(staging, ignore_errors=True); raise
        return target
    def load(self, repository_id: str, skill_id: str, *, active: bool | None = None) -> tuple[Path, SkillManifest, SkillEvidence, str]:
        bases = (["active"] if active else ["candidates"] if active is False else ["active", "candidates", "archived", "rejected"])
        for base in bases:
            for path in (self.repository_dir(repository_id) / base).glob("*"):
                manifest_path = path / "manifest.yaml"
                if manifest_path.exists():
                    data = _read_json(manifest_path)
                    if data.get("id") == skill_id:
                        return path, SkillManifest.model_validate(data), SkillEvidence.model_validate(_read_json(path / "evidence.json")), (path / "SKILL.md").read_text(encoding="utf-8")
        raise KeyError(skill_id)
    def activate(self, repository_id: str, skill_id: str) -> Path:
        with self.lock(repository_id, skill_id):
            source, manifest, evidence, markdown = self.load(repository_id, skill_id, active=False)
            target = self.repository_dir(repository_id) / "active" / manifest.name
            if target.exists(): raise FileExistsError(f"active skill already exists: {manifest.name}")
            manifest.status = "active"
            version = target / "versions" / manifest.version
            target.mkdir(parents=True); _atomic_text(target / "SKILL.md", markdown); _atomic_json(target / "manifest.yaml", manifest.model_dump(mode="json")); _atomic_json(target / "evidence.json", evidence.model_dump(mode="json")); _atomic_text(version / "SKILL.md", markdown); _atomic_json(version / "manifest.yaml", manifest.model_dump(mode="json")); _atomic_json(version / "evidence.json", evidence.model_dump(mode="json")); _append_audit(target / "audit.jsonl", {"event": "skill.activated", "at": utcnow()}); shutil.rmtree(source); self.rebuild_index(repository_id)
            return target
    def transition(self, repository_id: str, skill_id: str, destination: Literal["archived", "rejected", "candidates"]) -> Path:
        """Move a non-immutable lifecycle entry while preserving its audit."""
        with self.lock(repository_id, skill_id):
            source, manifest, _evidence, _markdown = self.load(repository_id, skill_id)
            target = self.repository_dir(repository_id) / destination / manifest.id
            if target.exists(): raise FileExistsError(f"destination already contains {skill_id}")
            manifest.status = {"archived": "archived", "rejected": "rejected", "candidates": "candidate"}[destination]
            _atomic_json(source / "manifest.yaml", manifest.model_dump(mode="json"))
            _append_audit(source / "audit.jsonl", {"event": f"skill.{manifest.status}", "at": utcnow()})
            target.parent.mkdir(parents=True, exist_ok=True); os.replace(source, target)
            self.rebuild_index(repository_id)
            return target
    def reject(self, repository_id: str, skill_id: str, reason: str) -> Path:
        path = self.transition(repository_id, skill_id, "rejected")
        _atomic_text(path / "rejection.txt", str(reason).strip() + "\n")
        return path
    def list(self, repository_id: str, *, state: str | None = None) -> list[SkillManifest]:
        result=[]
        for bucket in ([state] if state else ["active", "candidates", "archived", "rejected"]):
            directory=self.repository_dir(repository_id)/bucket
            if not directory.exists(): continue
            for manifest_path in directory.glob("*/manifest.yaml"):
                try: result.append(SkillManifest.model_validate(_read_json(manifest_path)))
                except (OSError, ValueError, json.JSONDecodeError): continue
        return sorted(result, key=lambda item: (item.name, item.version))
    def rebuild_index(self, repository_id: str) -> Path:
        rows=[]
        for path in sorted((self.repository_dir(repository_id) / "active").glob("*/manifest.yaml")):
            manifest=SkillManifest.model_validate(_read_json(path)); rows.append({"id":manifest.id,"name":manifest.name,"description":manifest.description,"scope":manifest.scope,"repository_identity":manifest.repository.get("id"),"version":manifest.version,"status":manifest.status,"semantic_applicability":manifest.applicability.get("semantic_intents", []),"required_tools":manifest.applicability.get("required_tools", []),"permission_summary":manifest.permissions.model_dump(),"quality_score":manifest.quality.get("confidence", 0),"staleness_status":manifest.status})
        target=self.repository_dir(repository_id)/"index.json"; _atomic_json(target,{"schema_version":SCHEMA_VERSION,"skills":rows,"updated_at":utcnow()}); return target


class SkillCandidateGenerator:
    """Writes model-produced, typed candidate content only after evidence checks."""
    def __init__(self, storage: SkillStorage, validator: SkillValidator | None = None) -> None: self.storage=storage; self.validator=validator or SkillValidator()
    def create(self, *, markdown: str, manifest: SkillManifest, evidence: SkillEvidence) -> Path:
        findings=self.validator.validate(markdown, manifest, evidence, self.storage.repository_dir(evidence.repository_id))
        return self.storage.write_candidate(markdown, manifest, evidence, findings)


class SkillSelector:
    """Applies deterministic safety constraints to an explicit model decision.

    Semantic relevance is supplied by the model as ``selected_ids``; this class
    never infers relevance from user text.
    """
    def __init__(self, storage: SkillStorage, maximum: int | None = None) -> None: self.storage=storage; self.maximum=maximum or int(os.getenv("MANA_SKILLS_MAX_LOADED", "4"))
    def select(self, repository_id: str, selected_ids: Iterable[str], *, available_tools: Iterable[str] = ()) -> tuple[list[SelectionDecision], list[SkillManifest]]:
        tools=set(available_tools); decisions=[]; selected=[]
        for skill_id in selected_ids:
            try: _path, manifest, _evidence, _markdown=self.storage.load(repository_id, skill_id, active=True)
            except KeyError: decisions.append(SelectionDecision(str(skill_id),False,"Skill is not active for this repository.",policy_result="denied")); continue
            required=set(manifest.applicability.get("required_tools", []))
            if manifest.status != "active" or not required.issubset(tools): decisions.append(SelectionDecision(manifest.id,False,"Inactive, stale, or required tools unavailable.",policy_result="denied")); continue
            if len(selected) >= self.maximum: decisions.append(SelectionDecision(manifest.id,False,"Maximum loaded skills reached.",policy_result="denied")); continue
            selected.append(manifest); decisions.append(SelectionDecision(manifest.id,True,"Explicit model decision passed scope, status, tool, and token limits.",scope_resolution="repository"))
        return decisions, selected


def _read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle: handle.write(text); temporary=Path(handle.name)
    os.replace(temporary, path)
    try: path.chmod(0o600)
    except OSError: pass
def _atomic_json(path: Path, value: Any) -> None: _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n")
def _append_audit(path: Path, value: dict[str, Any]) -> None: _atomic_text(path, ((path.read_text(encoding="utf-8") if path.exists() else "") + json.dumps(value, sort_keys=True) + "\n"))

# Named public boundaries kept intentionally small; richer lifecycle workers can
# be injected without allowing agents direct filesystem access.
SkillRepository = SkillStorage
SkillIndex = SkillStorage
SkillLoader = SkillSelector
SkillOutcomeRecorder = object
SkillEvaluator = object
SkillCurator = object
SkillVersionManager = object
SkillAuditService = object
ManaPathResolver = SkillStorage
