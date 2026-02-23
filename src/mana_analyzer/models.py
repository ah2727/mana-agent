from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DependencyPackageRef:
    name: str
    ecosystem: str
    scope: str  # runtime|dev
    manifest_path: str
    package_manager: str
    version_spec_raw: str
    exact_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ecosystem": self.ecosystem,
            "scope": self.scope,
            "manifest_path": self.manifest_path,
            "package_manager": self.package_manager,
            "version_spec_raw": self.version_spec_raw,
            "exact_version": self.exact_version,
        }


@dataclass
class VulnerabilityReference:
    type: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "url": self.url}


@dataclass
class DependencyVulnerability:
    package: DependencyPackageRef
    osv_id: str
    aliases: list[str] = field(default_factory=list)
    cve_aliases: list[str] = field(default_factory=list)
    summary: str = ""
    details: str = ""
    severity: list[Any] = field(default_factory=list)
    references: list[VulnerabilityReference] = field(default_factory=list)
    confidence: str = "potential"  # confirmed|potential

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package.to_dict(),
            "osv_id": self.osv_id,
            "aliases": self.aliases,
            "cve_aliases": self.cve_aliases,
            "summary": self.summary,
            "details": self.details,
            "severity": self.severity,
            "references": [r.to_dict() for r in self.references],
            "confidence": self.confidence,
        }


@dataclass
class SecurityScanReport:
    source: str  # osv
    status: str  # ok|partial|offline|unavailable
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    scanned_packages: list[dict[str, Any]] = field(default_factory=list)
    vulnerabilities: list[DependencyVulnerability] = field(default_factory=list)
    vulnerabilities_by_scope: dict[str, list[DependencyVulnerability]] = field(default_factory=dict)
    severity_buckets: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""
    duration_ms: int = 0

    def compute_counts(self) -> dict[str, int]:
        packages_scanned = len(self.scanned_packages)
        runtime_packages = 0
        dev_packages = 0
        confirmed = 0
        potential = 0
        cves = set()

        for p in self.scanned_packages:
            if p.get("scope") == "runtime":
                runtime_packages += 1
            elif p.get("scope") == "dev":
                dev_packages += 1

        for v in self.vulnerabilities:
            if v.confidence == "confirmed":
                confirmed += 1
            else:
                potential += 1
            for c in v.cve_aliases:
                cves.add(c)

        return {
            "packages_scanned": packages_scanned,
            "runtime_packages_scanned": runtime_packages,
            "dev_packages_scanned": dev_packages,
            "potential_vulns": potential,
            "confirmed_vulns": confirmed,
            "unique_cve_aliases": len(cves),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "warnings": self.warnings,
            "errors": self.errors,
            "scanned_packages": self.scanned_packages,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "vulnerabilities_by_scope": {
                k: [v.to_dict() for v in vs] for k, vs in (self.vulnerabilities_by_scope or {}).items()
            },
            "severity_buckets": self.severity_buckets,
            "generated_at": self.generated_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class FindingSummary:
    counts: dict[str, int]
    by_rule: dict[str, int]
    by_severity: dict[str, int]

    @staticmethod
    def from_findings(static_findings: list[Any], llm_findings: list[Any], merged_findings: list[Any]) -> "FindingSummary":
        by_rule: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for f in merged_findings:
            rule = getattr(f, "rule_id", "unknown")
            sev = getattr(f, "severity", "unknown")
            by_rule[rule] = by_rule.get(rule, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        counts = {
            "static": len(static_findings),
            "llm": len(llm_findings),
            "total": len(merged_findings),
            "warning": by_severity.get("warning", 0),
            "error": by_severity.get("error", 0),
        }
        return FindingSummary(counts=counts, by_rule=by_rule, by_severity=by_severity)


@dataclass
class ProjectReportMeta:
    project_root: str
    generated_at: str
    tool_version: str
    online: bool
    llm_enabled: bool
    output_format: str
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "generated_at": self.generated_at,
            "tool_version": self.tool_version,
            "online": self.online,
            "llm_enabled": self.llm_enabled,
            "output_format": self.output_format,
            "limitations": self.limitations,
        }


@dataclass
class ProjectReportSummary:
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    dependency_counts: dict[str, int] = field(default_factory=dict)
    finding_counts: dict[str, int] = field(default_factory=dict)
    security_counts: dict[str, int] = field(default_factory=dict)
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "languages": self.languages,
            "frameworks": self.frameworks,
            "technologies": self.technologies,
            "dependency_counts": self.dependency_counts,
            "finding_counts": self.finding_counts,
            "security_counts": self.security_counts,
            "status": self.status,
        }


@dataclass
class ProjectAuditReport:
    meta: ProjectReportMeta
    summary: ProjectReportSummary
    project_summary: dict[str, Any]
    dependencies: dict[str, Any]
    findings: dict[str, Any]
    security: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "summary": self.summary.to_dict(),
            "project_summary": self.project_summary,
            "dependencies": self.dependencies,
            "findings": self.findings,
            "security": self.security,
            "warnings": self.warnings,
        }


@dataclass
class DependencyPackageRef:
    name: str
    ecosystem: str               # PyPI, npm, Go, crates.io, RubyGems, Packagist, Pub
    scope: str                   # runtime|dev
    manifest_path: str           # relative to project root
    package_manager: str         # pip|npm|go|cargo|gem|composer|pub
    version_spec_raw: str
    exact_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ecosystem": self.ecosystem,
            "scope": self.scope,
            "manifest_path": self.manifest_path,
            "package_manager": self.package_manager,
            "version_spec_raw": self.version_spec_raw,
            "exact_version": self.exact_version,
        }
        
        
@dataclass
class FileHotspot:
    path: str
    reason: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "reason": self.reason, "score": self.score}


@dataclass
class FileStructureSummary:
    scope: str = "source+config"
    total_files: int = 0
    language_counts: dict[str, int] = field(default_factory=dict)
    tree_markdown: str = ""
    hotspots: list[FileHotspot] = field(default_factory=list)
    exclusions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "total_files": self.total_files,
            "language_counts": self.language_counts,
            "tree_markdown": self.tree_markdown,
            "hotspots": [h.to_dict() for h in self.hotspots],
            "exclusions": self.exclusions,
        }


@dataclass
class FlowAnalysis:
    mode: str = "local-fallback"   # llm|local-fallback
    line_target: int = 350
    security_lens: str = "defensive-red-team"
    content_markdown: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "line_target": self.line_target,
            "security_lens": self.security_lens,
            "content_markdown": self.content_markdown,
            "warnings": self.warnings,
        }