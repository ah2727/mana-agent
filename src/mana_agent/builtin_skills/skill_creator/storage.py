from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .renderer import render_proposal_yaml, render_readme
from .schema import (
    ProposalEvidence,
    ProposalManifest,
    ValidationReport,
    WorkshopConfig,
    default_workshop_paths,
)
from .validator import ProposalValidator, validate_managed_path


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class WorkshopPaths:
    skills: Path
    proposals: Path
    quarantine: Path

    @classmethod
    def from_config(cls, config: WorkshopConfig | None = None) -> "WorkshopPaths":
        skills, proposals, quarantine = default_workshop_paths(config)
        return cls(skills, proposals, quarantine)

    def ensure(self) -> "WorkshopPaths":
        for path in (self.skills, self.proposals, self.quarantine):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return self


class ProposalStorage:
    """Atomic, locked proposal lifecycle storage outside active skill loading."""

    def __init__(self, paths: WorkshopPaths | None = None, config: WorkshopConfig | None = None) -> None:
        self.config = config or WorkshopConfig.load()
        self.paths = (paths or WorkshopPaths.from_config(self.config)).ensure()
        self._roots = (self.paths.skills, self.paths.proposals, self.paths.quarantine)
        common = Path(os.path.commonpath([str(self.paths.proposals), str(self.paths.quarantine)]))
        self._lock_path = common / ".skill-workshop.lock"

    @contextmanager
    def lock(self, timeout: float = 10.0) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                break
            except FileExistsError:
                if self._lock_path.exists() and time.time() - self._lock_path.stat().st_mtime > 300:
                    self._lock_path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("skill workshop lock is busy")
                time.sleep(0.05)
        try:
            yield
        finally:
            self._lock_path.unlink(missing_ok=True)

    def proposal_path(self, proposal_id: str, *, include_quarantine: bool = True) -> Path:
        for root in (self.paths.proposals, self.paths.quarantine) if include_quarantine else (self.paths.proposals,):
            candidate = validate_managed_path(root / proposal_id, self._roots)
            if candidate.is_dir():
                return candidate
        raise KeyError(proposal_id)

    def write(
        self,
        manifest: ProposalManifest,
        markdown: str,
        evidence: ProposalEvidence,
        report: ValidationReport,
        *,
        quarantine: bool = False,
    ) -> Path:
        root = self.paths.quarantine if quarantine else self.paths.proposals
        target = validate_managed_path(root / manifest.proposal_id, self._roots)
        with self.lock():
            if target.exists():
                raise FileExistsError(f"proposal already exists: {manifest.proposal_id}")
            staging = validate_managed_path(root / f".{manifest.proposal_id}.{uuid4().hex}.tmp", self._roots)
            staging.mkdir(parents=False, exist_ok=False, mode=0o700)
            try:
                self._write_text(staging / "proposal.yaml", render_proposal_yaml(manifest))
                self._write_text(staging / "SKILL.md", markdown)
                self._write_json(staging / "evidence.json", evidence.model_dump(mode="json"))
                self._write_json(staging / "validation.json", report.model_dump(mode="json"))
                self._write_text(staging / "README.md", render_readme(manifest, report))
                os.replace(staging, target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return target

    def load(self, proposal_id: str) -> tuple[Path, ProposalManifest, ProposalEvidence, ValidationReport, str]:
        path = self.proposal_path(proposal_id)
        manifest = ProposalManifest.model_validate_json((path / "proposal.yaml").read_text(encoding="utf-8"))
        evidence = ProposalEvidence.model_validate_json((path / "evidence.json").read_text(encoding="utf-8"))
        report = ValidationReport.model_validate_json((path / "validation.json").read_text(encoding="utf-8"))
        return path, manifest, evidence, report, (path / "SKILL.md").read_text(encoding="utf-8")

    def list(self, *, status: str | None = None, min_confidence: float = 0.0, risk: str | None = None) -> list[ProposalManifest]:
        rows: list[ProposalManifest] = []
        for root in (self.paths.proposals, self.paths.quarantine):
            for manifest_path in root.glob("*/proposal.yaml"):
                try:
                    item = ProposalManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if status and item.status != status:
                    continue
                if item.confidence < min_confidence:
                    continue
                if risk and item.risk.get("level") != risk:
                    continue
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.proposal_id), reverse=True)

    def update(self, path: Path, manifest: ProposalManifest, markdown: str, evidence: ProposalEvidence, report: ValidationReport) -> None:
        validate_managed_path(path, self._roots)
        with self.lock():
            self._write_text(path / "proposal.yaml", render_proposal_yaml(manifest))
            self._write_text(path / "SKILL.md", markdown)
            self._write_json(path / "evidence.json", evidence.model_dump(mode="json"))
            self._write_json(path / "validation.json", report.model_dump(mode="json"))
            self._write_text(path / "README.md", render_readme(manifest, report))

    def install(self, proposal_id: str, *, approved: bool, version: str = "1.0.0") -> Path:
        if not approved:
            raise PermissionError("explicit user approval is required to install a proposal")
        path, manifest, evidence, _old_report, markdown = self.load(proposal_id)
        if manifest.status not in {"pending_review", "needs_attention"}:
            raise ValueError(f"proposal status cannot be installed: {manifest.status}")
        manifest.version = version
        report = ProposalValidator().validate(manifest, markdown, evidence)
        if not report.valid:
            raise ValueError("proposal failed installation revalidation")
        target = validate_managed_path(self.paths.skills / manifest.name, self._roots)
        with self.lock():
            if target.exists():
                raise FileExistsError(f"active skill already exists: {manifest.name}; use an explicit upgrade flow")
            staging = validate_managed_path(self.paths.skills / f".{manifest.name}.{uuid4().hex}.tmp", self._roots)
            staging.mkdir(mode=0o700)
            try:
                self._write_text(staging / "SKILL.md", markdown)
                provenance = {
                    "schema_version": 1,
                    "proposal_id": manifest.proposal_id,
                    "version": version,
                    "installed_at": utcnow(),
                    "source_sessions": manifest.source_sessions,
                    "source_tasks": manifest.source_tasks,
                    "evidence_sha256": self._sha256(path / "evidence.json"),
                }
                self._write_json(staging / "provenance.json", provenance)
                self._write_text(staging / "versions" / version / "SKILL.md", markdown)
                self._write_json(staging / "versions" / version / "provenance.json", provenance)
                os.replace(staging, target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            manifest.status = "installed"
            manifest.installed_at = utcnow()
            manifest.updated_at = manifest.installed_at
            self._write_text(path / "proposal.yaml", render_proposal_yaml(manifest))
            self._rebuild_index()
        self._emit("skill_proposal_installed", manifest, path=target)
        return target

    def reject(self, proposal_id: str, reason: str = "") -> Path:
        path, manifest, evidence, report, markdown = self.load(proposal_id)
        manifest.status = "rejected"
        manifest.rejection_reason = reason.strip() or None
        manifest.updated_at = utcnow()
        self.update(path, manifest, markdown, evidence, report)
        self._emit("skill_proposal_rejected", manifest, path=path)
        return path

    def quarantine(self, proposal_id: str, reason: str) -> Path:
        source, manifest, evidence, report, markdown = self.load(proposal_id)
        if source.parent == self.paths.quarantine:
            return source
        manifest.status = "quarantined"
        manifest.quarantine_reason = reason.strip() or "Quarantined by user."
        manifest.updated_at = utcnow()
        target = validate_managed_path(self.paths.quarantine / proposal_id, self._roots)
        with self.lock():
            if target.exists():
                raise FileExistsError(f"quarantine already contains: {proposal_id}")
            self._write_text(source / "proposal.yaml", render_proposal_yaml(manifest))
            self._write_json(source / "evidence.json", evidence.model_dump(mode="json"))
            self._write_json(source / "validation.json", report.model_dump(mode="json"))
            self._write_text(source / "SKILL.md", markdown)
            os.replace(source, target)
        self._emit("skill_proposal_quarantined", manifest, path=target)
        return target

    def _rebuild_index(self) -> None:
        rows = []
        for path in sorted(self.paths.skills.glob("*/SKILL.md")):
            provenance = path.parent / "provenance.json"
            rows.append({"name": path.parent.name, "path": str(path), "provenance": json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None})
        self._write_json(self.paths.skills / "index.json", {"schema_version": 1, "updated_at": utcnow(), "skills": rows})

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    @classmethod
    def _write_json(cls, path: Path, value: object) -> None:
        cls._write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _emit(event_type: str, manifest: ProposalManifest, *, path: Path) -> None:
        try:
            from mana_agent.services.execution_event_hub import get_execution_event_hub
            get_execution_event_hub().emit(
                event_type,
                title=event_type.replace("_", " ").title(),
                conversation_id=manifest.source_sessions[0] if manifest.source_sessions else "",
                execution_id=manifest.source_tasks[0] if manifest.source_tasks else "",
                status="success",
                metadata={"proposal_id": manifest.proposal_id, "name": manifest.name, "path": str(path)},
            )
        except Exception:
            # Event delivery is observability only and cannot corrupt a completed
            # locked lifecycle transition.
            pass
