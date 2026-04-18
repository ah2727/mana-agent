from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mana_analyzer.models import (
    ProjectAuditReport,
    ProjectReportSummary,
    FindingSummary,
    ProjectReportMeta,
    FileStructureSummary,
    FileHotspot,
    FlowAnalysis,
)
from mana_analyzer.services.analyze_service import AnalyzeService
from mana_analyzer.dependencies.dependency_service import DependencyService
from mana_analyzer.describe.describe_service import DescribeService
from mana_analyzer.services.llm_analyze_service import LlmAnalyzeService
from mana_analyzer.services.structure_service import StructureService
from mana_analyzer.services.vulnerability_service import VulnerabilityService


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_version_string() -> str:
    try:
        import importlib.metadata as md

        return md.version("mana-analyzer")  # adjust if dist name differs
    except Exception:
        return "dev"


def _find_project_root(start: Path) -> Path:
    """
    Find a reasonable project root by walking upwards until a marker file/dir is found.
    Markers: pyproject.toml, setup.cfg, requirements.txt, .git
    If none found, return the original directory.
    """
    markers_files = {"pyproject.toml", "setup.cfg", "requirements.txt"}
    markers_dirs = {".git"}

    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent

    # include current and parents
    for candidate in [cur, *cur.parents]:
        try:
            # file markers
            for mf in markers_files:
                if (candidate / mf).exists():
                    return candidate
            # dir markers
            for md in markers_dirs:
                if (candidate / md).exists():
                    return candidate
        except Exception:
            # ignore permission issues and keep walking
            pass

    return cur

def _safe_to_dict(obj: Any) -> dict[str, Any]:
    """Convert object to dict, handling cases where it's already a dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return {}
    

@dataclass(frozen=True)
class _DepsBundle:
    deps_graph: Any
    inventory: list[Any]
    inventory_runtime: list[Any]
    inventory_dev: list[Any]
    warnings: list[str]


@dataclass(frozen=True)
class _FindingsBundle:
    static_findings: list[Any]
    llm_findings: list[Any]
    merged_findings: list[Any]
    finding_summary: FindingSummary
    warnings: list[str]


@dataclass(frozen=True)
class _StructureBundle:
    structure_report: Any | None
    warnings: list[str]


@dataclass(frozen=True)
class _SecurityBundle:
    security_report: Any
    warnings: list[str]


@dataclass(frozen=True)
class _DeepBundle:
    file_structure_payload: FileStructureSummary | None
    flow_payload: FlowAnalysis | None
    warnings: list[str]


class ReportService:
    def __init__(
        self,
        *,
        dependency_service: DependencyService,
        analyze_service: AnalyzeService,
        llm_analyze_service: LlmAnalyzeService | None,
        describe_service: DescribeService,
        structure_service: StructureService,
        vulnerability_service: VulnerabilityService,
    ) -> None:
        self.dependency_service = dependency_service
        self.analyze_service = analyze_service
        self.llm_analyze_service = llm_analyze_service
        self.describe_service = describe_service
        self.structure_service = structure_service
        self.vulnerability_service = vulnerability_service
        

    # ---------------------------
    # Public API
    # ---------------------------
    def generate(
        self,
        *,
        target_path: str,
        with_llm: bool,
        model_override: str | None,
        llm_max_files: int,
        summary_max_files: int,
        full_structure: bool,
        online: bool,
        osv_timeout_seconds: int,
        security_scope: str,
        report_profile: str = "standard",
        detail_line_target: int = 350,
        security_lens: str = "defensive-red-team",
    ) -> ProjectAuditReport:
        # Validate new deep-profile options
        if report_profile not in {"standard", "deep"}:
            raise ValueError("report_profile must be standard|deep")
        if security_lens not in {"defensive-red-team", "architecture", "compliance"}:
            raise ValueError("security_lens must be defensive-red-team|architecture|compliance")

        effective_full_structure = full_structure or (report_profile == "deep")
        if report_profile == "deep":
            detail_line_target = self._clamp(detail_line_target, lo=300, hi=400)

        root = Path(target_path).resolve()
        project_root = _find_project_root(root)

        warnings: list[str] = []

        # Dependencies / inventory
        deps_bundle = self._build_dependencies(project_root)
        warnings.extend(deps_bundle.warnings)

        # Static + optional LLM findings
        findings_bundle = self._build_findings(
            project_root,
            with_llm=with_llm,
            model_override=model_override,
            llm_max_files=llm_max_files,
        )
        warnings.extend(findings_bundle.warnings)

        # Describe summary (architecture + file summaries)
        describe_report = self._build_describe(project_root)


        # Structure payload
        structure_bundle = self._build_structure(project_root, enabled=effective_full_structure)
        warnings.extend(structure_bundle.warnings)

        # OSV scan
        security_bundle = self._build_security(
            inventory=deps_bundle.inventory,
            online=online,
            osv_timeout_seconds=osv_timeout_seconds,
            security_scope=security_scope,
        )
        warnings.extend(security_bundle.warnings)

        # Deep payloads (additive)
        deep_bundle = self._build_deep_payloads(
            report_profile=report_profile,
            with_llm=with_llm,
            detail_line_target=detail_line_target,
            security_lens=security_lens,
            deps_graph=deps_bundle.deps_graph,
            finding_summary=findings_bundle.finding_summary,
            describe_report=describe_report,
            structure_report=structure_bundle.structure_report,
            security_report=security_bundle.security_report,
        )
        warnings.extend(deep_bundle.warnings)

        summary = ProjectReportSummary(
            languages=deps_bundle.deps_graph.languages,
            frameworks=deps_bundle.deps_graph.frameworks,
            technologies=deps_bundle.deps_graph.technologies,
            dependency_counts={
                "runtime": len(deps_bundle.deps_graph.runtime_dependencies),
                "dev": len(deps_bundle.deps_graph.dev_dependencies),
                "inventory_total": len(deps_bundle.inventory),
            },
            finding_counts=findings_bundle.finding_summary.counts,
            security_counts=security_bundle.security_report.compute_counts(),
            status=self._derive_status(findings_bundle.finding_summary, security_bundle.security_report, warnings),
        )

        meta = ProjectReportMeta(
            project_root=str(project_root),
            generated_at=_iso_utc_now(),
            tool_version=_safe_version_string(),
            online=online,
            llm_enabled=with_llm,
            output_format="all",
            limitations=[
                "Direct dependencies only (no transitive lockfile scan in v1).",
                "Static analysis is Python-only in v1.",
                "OSV matches without exact version are labeled potential.",
            ],
        )

        # Build the project_summary dict
        project_summary: dict[str, Any] = {
            "describe": _safe_to_dict(describe_report),
            "structure": _safe_to_dict(structure_bundle.structure_report) if structure_bundle.structure_report else None,
            "file_structure": _safe_to_dict(deep_bundle.file_structure_payload) if deep_bundle.file_structure_payload else None,
            "flow_analysis": _safe_to_dict(deep_bundle.flow_payload) if deep_bundle.flow_payload else None,
        }

        return ProjectAuditReport(
            meta=meta,
            summary=summary,
            project_summary=project_summary,
            dependencies={
                "graph": deps_bundle.deps_graph.to_dict(),
                "inventory": [d.to_dict() for d in deps_bundle.inventory],
                "inventory_by_scope": {
                    "runtime": [d.to_dict() for d in deps_bundle.inventory_runtime],
                    "dev": [d.to_dict() for d in deps_bundle.inventory_dev],
                },
            },
            findings={
                "static_findings": [f.to_dict() for f in findings_bundle.static_findings],
                "llm_findings": [f.to_dict() for f in findings_bundle.llm_findings],
                "merged_findings": [f.to_dict() for f in findings_bundle.merged_findings],
                "by_rule": findings_bundle.finding_summary.by_rule,
                "by_severity": findings_bundle.finding_summary.by_severity,
            },
            security=security_bundle.security_report.to_dict(),
            warnings=warnings,
        )

    # ---------------------------
    # Builders (refactor points)
    # ---------------------------
    def _build_dependencies(self, project_root: Path) -> _DepsBundle:
        warnings: list[str] = []

        deps_graph = self.dependency_service.analyze(str(project_root))

        try:
            inventory = self.dependency_service.collect_inventory(str(project_root))
        except Exception as exc:
            inventory = []
            warnings.append(f"collect_inventory failed ({type(exc).__name__}): {exc}")

        inventory_runtime = [d for d in inventory if getattr(d, "scope", None) == "runtime"]
        inventory_dev = [d for d in inventory if getattr(d, "scope", None) == "dev"]

        return _DepsBundle(
            deps_graph=deps_graph,
            inventory=inventory,
            inventory_runtime=inventory_runtime,
            inventory_dev=inventory_dev,
            warnings=warnings,
        )

    def _build_findings(
        self,
        project_root: Path,
        *,
        with_llm: bool,
        model_override: str | None,
        llm_max_files: int,
    ) -> _FindingsBundle:
        warnings: list[str] = []

        static_findings = self.analyze_service.analyze(str(project_root))

        llm_findings: list[Any] = []
        if with_llm and self.llm_analyze_service is not None:
            try:
                # If your LlmAnalyzeService supports a model override, pass it.
                # If it doesn't, this kwarg will raise; we fall back gracefully.
                try:
                    llm_findings = self.llm_analyze_service.analyze(
                        str(project_root),
                        static_findings=static_findings,
                        max_files=llm_max_files,
                        model_override=model_override,
                    )
                except TypeError:
                    llm_findings = self.llm_analyze_service.analyze(
                        str(project_root),
                        static_findings=static_findings,
                        max_files=llm_max_files,
                    )
            except Exception as exc:
                warnings.append(f"LLM analyze failed ({type(exc).__name__}): {exc}")
                llm_findings = []

        merged_findings = list(static_findings) + list(llm_findings)
        finding_summary = FindingSummary.from_findings(static_findings, llm_findings, merged_findings)

        return _FindingsBundle(
            static_findings=list(static_findings),
            llm_findings=list(llm_findings),
            merged_findings=merged_findings,
            finding_summary=finding_summary,
            warnings=warnings,
        )

    def _build_describe(self, project_root: Path) -> Any:
        """
        Call DescribeService.describe with just the project path.
        The DescribeService instance was already configured
        (via build_describe_service) with its file_agent and llm_chain.
        """
        return self.describe_service.describe(project_root)

    def _build_structure(self, project_root: Path, *, enabled: bool) -> _StructureBundle:
        warnings: list[str] = []
        if not enabled:
            return _StructureBundle(structure_report=None, warnings=warnings)

        try:
            structure_report = self.structure_service.analyze_project(str(project_root))
            return _StructureBundle(structure_report=structure_report, warnings=warnings)
        except Exception as exc:
            warnings.append(f"StructureService failed ({type(exc).__name__}): {exc}")
            return _StructureBundle(structure_report=None, warnings=warnings)
        

    def _build_security(
        self,
        *,
        inventory: list[Any],
        online: bool,
        osv_timeout_seconds: int,
        security_scope: str,
    ) -> _SecurityBundle:
        warnings: list[str] = []

        security_report = self.vulnerability_service.scan_dependencies(
            inventory,
            online=online,
            timeout_seconds=osv_timeout_seconds,
            scope=security_scope,
        )
        warnings.extend(getattr(security_report, "warnings", []) or [])
        return _SecurityBundle(security_report=security_report, warnings=warnings)

    def _build_deep_payloads(
        self,
        *,
        report_profile: str,
        with_llm: bool,
        detail_line_target: int,
        security_lens: str,
        deps_graph: Any,
        finding_summary: FindingSummary,
        describe_report: Any,
        structure_report: Any | None,
        security_report: Any,
    ) -> _DeepBundle:
        warnings: list[str] = []

        if report_profile != "deep":
            return _DeepBundle(file_structure_payload=None, flow_payload=None, warnings=warnings)

        # ✅ LLM is mandatory for deep profile
        if not with_llm:
            raise ValueError("LLM-only mode: with_llm must be True for deep profile.")

        # Delegate to DescribeService; it will raise if synthesize_deep_flow_analysis is missing
        synthesize = self.describe_service.synthesize_deep_flow_analysis

        # ----- file structure payload (still best-effort; not LLM)
        file_structure_payload: FileStructureSummary | None = None
        if structure_report is not None:
            files = getattr(structure_report, "files", []) or []
            try:
                tree_md = self.structure_service.render_file_tree_markdown(files)
            except Exception:
                tree_md = ""
            language_counts = getattr(structure_report, "language_counts", {}) or {}
            try:
                hotspots_raw = self.structure_service.compute_hotspots(structure_report, top_n=15)
            except Exception:
                hotspots_raw = []

            file_structure_payload = FileStructureSummary(
                scope="source+config",
                total_files=len(files),
                language_counts=language_counts,
                tree_markdown=tree_md,
                hotspots=[FileHotspot(**h) for h in hotspots_raw],
                exclusions=getattr(structure_report, "discovery_stats", {}) or {},
            )

        # ----- LLM flow synthesis (no fallback)
        # sample up to 8 file summaries from describe_report
        if isinstance(describe_report, dict):
            describe_dict = describe_report
        else:
            describe_dict = describe_report.to_dict()
        sampled = describe_dict.get("files", [])[:8]

        structure_summary = {
            "total_files": (file_structure_payload.total_files if file_structure_payload else 0),
            "language_counts": (file_structure_payload.language_counts if file_structure_payload else {}),
            "hotspots": [h.to_dict() for h in (file_structure_payload.hotspots if file_structure_payload else [])],
            "tree_markdown": (file_structure_payload.tree_markdown if file_structure_payload else ""),
        }
        findings_summary = {
            "counts": finding_summary.counts,
            "top_rules": sorted(finding_summary.by_rule.items(), key=lambda kv: (-kv[1], kv[0]))[:15],
            "by_severity": finding_summary.by_severity,
        }

        # If LLM errors, we fail hard (as requested)
        content = synthesize(
             dependency_report=deps_graph,
             structure_summary=structure_summary,
             findings_summary=findings_summary,
             security_summary=security_report.to_dict(),
             sampled_file_summaries=sampled,
             line_target=detail_line_target,
             security_lens=security_lens,
         )

        if not str(content).strip():
            raise RuntimeError("LLM-only mode: synthesize_deep_flow_analysis returned empty content.")

        flow_payload = FlowAnalysis(
            mode="llm",
            line_target=detail_line_target,
            security_lens=security_lens,
            content_markdown=content,
            warnings=[],
        )

        return _DeepBundle(file_structure_payload=file_structure_payload, flow_payload=flow_payload, warnings=warnings)
    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _clamp(value: int, *, lo: int, hi: int) -> int:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    def _derive_status(self, finding_summary: FindingSummary, security_report, warnings: list[str]) -> str:
        has_errors = finding_summary.counts.get("error", 0) > 0
        has_warn = len(warnings) > 0 or finding_summary.counts.get("warning", 0) > 0
        has_vulns = security_report.compute_counts().get("potential_vulns", 0) > 0
        if has_errors:
            return "errors_found"
        if has_vulns:
            return "security_issues_found"
        if has_warn:
            return "warnings"
        return "ok"

    def _render_local_fallback_flow_analysis(
        self,
        *,
        deps_report: Any,
        structure_report: Any | None,
        describe_report: Any,
        security_lens: str,
        sampled_file_summaries: list[dict[str, Any]] | None = None,
        file_tree_markdown: str = "",
    ) -> str:
        """
        Local deterministic fallback. Includes sampled summaries (when available) and
        respects the security_lens by adjusting headings and checklists.
        """
        sampled_file_summaries = sampled_file_summaries or []

        lens_title = {
            "defensive-red-team": "## System Flow & Attack Surface (Defensive Red-Team)",
            "architecture": "## System Flow & Key Components (Architecture)",
            "compliance": "## System Flow & Controls Overview (Compliance)",
        }.get(security_lens, "## System Flow Overview")

        lines: list[str] = []
        lines.append(lens_title)
        lines.append("")
        lines.append("> Local fallback synthesis (LLM disabled/unavailable). Defensive-only; no exploit instructions.")
        lines.append("")

        # Include a file tree if available (deep mode attempted but structure missing)
        if file_tree_markdown.strip():
            lines.append("### File structure (if available)")
            lines.append(file_tree_markdown.rstrip())
            lines.append("")

        # Observed system shape
        lines.append("### Observed system shape")
        lines.append(f"- Languages: {', '.join(getattr(deps_report, 'languages', []) or []) or 'unknown'}")
        lines.append(f"- Frameworks: {', '.join(getattr(deps_report, 'frameworks', []) or []) or 'none'}")
        lines.append(f"- Technologies: {', '.join(getattr(deps_report, 'technologies', []) or []) or 'none'}")
        lines.append(f"- External import edges: {len(getattr(deps_report, 'dependency_edges', []) or [])}")
        lines.append(f"- Internal module edges: {len(getattr(deps_report, 'module_edges', []) or [])}")
        if structure_report is not None:
            lines.append(f"- Source+config files: {len(getattr(structure_report, 'files', []) or [])}")
            lines.append(f"- Commands discovered: {len(getattr(structure_report, 'commands', []) or [])}")
        lines.append("")

        # Sampled file summaries (fix: previously missing)
        if sampled_file_summaries:
            lines.append("### Sampled file summaries (fallback)")
            for d in sampled_file_summaries[:12]:
                fp = d.get("file_path", "unknown")
                lang = d.get("language", "text")
                summ = (d.get("summary", "") or "").strip()
                if summ:
                    lines.append(f"- `{fp}` ({lang}) — {summ}")
                else:
                    lines.append(f"- `{fp}` ({lang})")
            lines.append("")

        # Lens-specific guidance
        if security_lens == "architecture":
            lines.append("### Architecture checklist")
            lines.append("- Identify entrypoints (CLI, HTTP handlers, workers) and their main responsibilities")
            lines.append("- Trace core data flows (input → validation → business logic → persistence → output)")
            lines.append("- Identify shared services (config, logging, database, messaging) and coupling points")
            lines.append("- Identify failure modes (timeouts, retries, partial failures) and resiliency patterns")
            lines.append("")
        elif security_lens == "compliance":
            lines.append("### Controls checklist (compliance-oriented)")
            lines.append("- Access control: least privilege, separation of duties, audit logs for sensitive actions")
            lines.append("- Data protection: encryption in transit/at rest, secret management, redaction policies")
            lines.append("- SDLC: dependency pinning, vulnerability scanning, SBOM generation, change approvals")
            lines.append("- Monitoring: alerting, incident response hooks, log retention, traceability")
            lines.append("")
        else:
            lines.append("### Trust boundaries checklist")
            lines.append("- Entry points: CLI commands, HTTP routes, job runners, webhook handlers")
            lines.append("- Inputs: env/config, request payloads, filesystem reads, third-party callbacks")
            lines.append("- Sinks: database writes, file writes, outbound network calls, template rendering")
            lines.append("")
            lines.append("### Defensive abuse paths (non-procedural)")
            lines.append("- Input validation drift across multiple entrypoints")
            lines.append("- Authorization gaps on privileged operations")
            lines.append("- Secret exposure via logs/errors/config dumps")
            lines.append("- Dependency risk: weak pinning, stale packages, supply-chain issues")
            lines.append("")
            lines.append("### Hardening priorities")
            lines.append("1. Centralize authN/authZ and enforce deny-by-default for privileged actions.")
            lines.append("2. Enforce schemas at edges; validate types and sizes; reject unexpected fields.")
            lines.append("3. Add structured logging with redaction; add audit trails for sensitive actions.")
            lines.append("4. Tighten dependency pinning and add CI auditing + SBOM generation.")
            lines.append("5. Add monitoring for auth failures, spikes, unusual access patterns, and risky calls.")
            lines.append("")
            lines.append("### Verification checklist")
            lines.append("- [ ] List all entrypoints and their input schemas")
            lines.append("- [ ] Confirm authZ checks at every privileged boundary")
            lines.append("- [ ] Confirm secrets are never logged")
            lines.append("- [ ] Confirm rate limits / timeouts / size limits exist on external inputs")
            lines.append("- [ ] Confirm dependency policy: updates, lockfile integrity, advisories")

        return "\n".join(lines)

    # ---------------------------
    # Markdown Rendering
    # ---------------------------
    def render_markdown(self, report: ProjectAuditReport) -> str:
        lines: list[str] = []
        lines.append("# Project Audit Report")
        lines.append("")
        lines.append("## Overview")
        lines.append(f"- Root: `{report.meta.project_root}`")
        lines.append(f"- Generated: {report.meta.generated_at}")
        lines.append(f"- Online OSV: {report.meta.online}")
        lines.append(f"- LLM enabled: {report.meta.llm_enabled}")
        lines.append(f"- Tool version: {report.meta.tool_version}")
        lines.append("")

        lines.append("## Technologies & Dependencies")
        lines.append(f"- Languages: {', '.join(report.summary.languages) if report.summary.languages else 'unknown'}")
        lines.append(f"- Frameworks: {', '.join(report.summary.frameworks) if report.summary.frameworks else 'none'}")
        lines.append(f"- Technologies: {', '.join(report.summary.technologies) if report.summary.technologies else 'none'}")
        lines.append(f"- Runtime deps (graph): {report.summary.dependency_counts.get('runtime', 0)}")
        lines.append(f"- Dev deps (graph): {report.summary.dependency_counts.get('dev', 0)}")
        lines.append("")

        ps = report.project_summary or {}
        is_deep = bool(ps.get("file_structure")) or bool(ps.get("flow_analysis"))

        lines.append("## Project Summary")
        describe = (ps.get("describe") or {})
        lines.append("### Architecture")
        lines.append(str(describe.get("architecture_summary", "")).strip() or "Architecture summary unavailable.")
        lines.append("")
        lines.append("### Technology")
        lines.append(str(describe.get("tech_summary", "")).strip() or "Technology summary unavailable.")
        lines.append("")

        if not is_deep:
            lines.append("### File Summaries")
            descs = describe.get("descriptions", []) or []
            if descs:
                for d in descs:
                    lines.append(
                        f"- `{d.get('file_path','unknown')}` ({d.get('language','text')}) — {d.get('summary','')}"
                    )
            else:
                lines.append("- none")
            lines.append("")
        else:
            fs = ps.get("file_structure") or {}
            lines.append("### File Structure Diagram")
            lines.append((fs.get("tree_markdown") or "").rstrip() or "Structure diagram unavailable.")
            lines.append("")
            lines.append("### File Inventory")
            lines.append(f"- Scope: {fs.get('scope','source+config')}")
            lines.append(f"- Total files: {fs.get('total_files', 0)}")

            # FIX: sort language counts to show top languages by count
            lang_counts = fs.get("language_counts") or {}
            if lang_counts:
                top = sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
                lines.append("- Languages: " + ", ".join(f"{k}={v}" for k, v in top))

            exclusions = fs.get("exclusions") or {}
            if exclusions:
                lines.append("- Exclusions / filters applied:")
                for k, v in exclusions.items():
                    lines.append(f"  - {k}: {v}")
            lines.append("")
            lines.append("### Hotspots")
            hotspots = fs.get("hotspots") or []
            if hotspots:
                for h in hotspots:
                    lines.append(f"- `{h.get('path')}` (score={h.get('score')}) — {h.get('reason')}")
            else:
                lines.append("- none")
            lines.append("")

        lines.append("## Bugs & Errors")
        fc = report.summary.finding_counts
        lines.append(f"- Static findings: {fc.get('static', 0)}")
        lines.append(f"- LLM findings: {fc.get('llm', 0)}")
        lines.append(f"- Total findings: {fc.get('total', 0)}")
        lines.append(f"- Warnings: {fc.get('warning', 0)} | Errors: {fc.get('error', 0)}")
        lines.append("")
        lines.append("### Top Rules")
        by_rule = (report.findings.get("by_rule") or {})
        for rule, count in sorted(by_rule.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            lines.append(f"- {rule}: {count}")
        lines.append("")

        lines.append("## Cyber Issues (OSV)")
        sec = report.security or {}
        lines.append(f"- Source: {sec.get('source', 'osv')}")
        lines.append(f"- Status: {sec.get('status', 'unknown')}")
        counts = report.summary.security_counts
        lines.append(f"- Packages scanned: {counts.get('packages_scanned', 0)}")
        lines.append(
            f"- Potential vulns: {counts.get('potential_vulns', 0)} | Confirmed: {counts.get('confirmed_vulns', 0)}"
        )
        lines.append("")

        lines.append("### Runtime")
        runtime_v = ((sec.get("vulnerabilities_by_scope") or {}).get("runtime") or [])
        if runtime_v:
            for v in runtime_v[:50]:
                pkg = (v.get("package") or {})
                lines.append(
                    f"- `{pkg.get('name')}` ({pkg.get('ecosystem')}) — {v.get('osv_id')} [{v.get('confidence')}]"
                )
                cves = v.get("cve_aliases") or []
                if cves:
                    lines.append(f"  - CVEs: {', '.join(cves[:8])}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("### Dev")
        dev_v = ((sec.get("vulnerabilities_by_scope") or {}).get("dev") or [])
        if dev_v:
            for v in dev_v[:50]:
                pkg = (v.get("package") or {})
                lines.append(
                    f"- `{pkg.get('name')}` ({pkg.get('ecosystem')}) — {v.get('osv_id')} [{v.get('confidence')}]"
                )
                cves = v.get("cve_aliases") or []
                if cves:
                    lines.append(f"  - CVEs: {', '.join(cves[:8])}")
        else:
            lines.append("- none")
        lines.append("")

        # Deep flow section (only in deep mode)
        if is_deep:
            fa = ps.get("flow_analysis") or {}
            content = (fa.get("content_markdown") or "").strip()
            if content:
                lines.append(content)
                lines.append("")

        lines.append("## Warnings & Limitations")
        if report.warnings:
            for w in report.warnings:
                lines.append(f"- {w}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("### Limitations")
        for lim in report.meta.limitations:
            lines.append(f"- {lim}")

        return "\n".join(lines)