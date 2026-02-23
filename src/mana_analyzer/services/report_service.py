from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from mana_analyzer.services.dependency_service import DependencyService
from mana_analyzer.services.describe_service import DescribeService
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

        # Deep implies structure and clamps the target size
        effective_full_structure = full_structure or (report_profile == "deep")
        if report_profile == "deep":
            if detail_line_target < 300:
                detail_line_target = 300
            elif detail_line_target > 400:
                detail_line_target = 400

        warnings: list[str] = []
        root = Path(target_path).resolve()
        project_root = root if root.is_dir() else root.parent

        # Dependencies / tech
        deps_graph = self.dependency_service.analyze(str(project_root))

        # Version-aware direct deps inventory
        try:
            inventory = self.dependency_service.collect_inventory(str(project_root))
        except Exception as exc:
            inventory = []
            warnings.append(f"collect_inventory failed: {exc}")

        inventory_runtime = [d for d in inventory if d.scope == "runtime"]
        inventory_dev = [d for d in inventory if d.scope == "dev"]

        # Static findings (Python-only in v1)
        static_findings = self.analyze_service.analyze(str(project_root))

        # Optional LLM findings
        llm_findings = []
        if with_llm and self.llm_analyze_service is not None:
            try:
                llm_findings = self.llm_analyze_service.analyze(
                    str(project_root),
                    static_findings=static_findings,
                    max_files=llm_max_files,
                )
            except Exception as exc:
                warnings.append(f"LLM analyze failed: {exc}")

        merged_findings = list(static_findings) + list(llm_findings)

        # Describe summary (architecture + file summaries)
        describe_report = self.describe_service.describe(
            str(project_root),
            max_files=summary_max_files,
            include_functions=False,
            use_llm=with_llm,
        )

        # Structure payload (forced for deep)
        structure_report = None
        if effective_full_structure:
            try:
                structure_report = self.structure_service.analyze_project(str(project_root))
            except Exception as exc:
                warnings.append(f"StructureService failed: {exc}")
                structure_report = None

        # OSV scan (graceful)
        security_report = self.vulnerability_service.scan_dependencies(
            inventory,
            online=online,
            timeout_seconds=osv_timeout_seconds,
            scope=security_scope,
        )
        warnings.extend(security_report.warnings)

        # Summary counts
        finding_summary = FindingSummary.from_findings(static_findings, llm_findings, merged_findings)

        # Deep payloads (additive)
        file_structure_payload: FileStructureSummary | None = None
        flow_payload: FlowAnalysis | None = None

        if report_profile == "deep":
            if structure_report is None:
                warnings.append("Deep profile requested but structure analysis is unavailable; falling back to sampled file summaries.")
            else:
                # Build file_structure from StructureService inventory
                files = getattr(structure_report, "files", []) or []
                try:
                    tree_md = self.structure_service.render_file_tree_markdown(files)
                except Exception as exc:
                    tree_md = ""
                    warnings.append(f"Failed to render file tree: {exc}")

                language_counts = getattr(structure_report, "language_counts", {}) or {}
                try:
                    hotspots_raw = self.structure_service.compute_hotspots(structure_report, top_n=15)
                except Exception as exc:
                    hotspots_raw = []
                    warnings.append(f"Failed to compute hotspots: {exc}")

                file_structure_payload = FileStructureSummary(
                    scope="source+config",
                    total_files=len(files),
                    language_counts=language_counts,
                    tree_markdown=tree_md,
                    hotspots=[FileHotspot(**h) for h in hotspots_raw],
                    exclusions=getattr(structure_report, "discovery_stats", {}) or {},
                )

            # Flow analysis payload (LLM or deterministic fallback)
            flow_warnings: list[str] = []
            mode = "local-fallback"
            content = ""

            llm_chain = getattr(self.describe_service, "llm_chain", None)
            if with_llm and llm_chain is not None and hasattr(llm_chain, "synthesize_deep_flow_analysis"):
                try:
                    mode = "llm"
                    sampled = (describe_report.to_dict().get("descriptions") or [])[:8]

                    structure_summary = {
                        "total_files": (file_structure_payload.total_files if file_structure_payload else 0),
                        "language_counts": (file_structure_payload.language_counts if file_structure_payload else {}),
                        "hotspots": [h.to_dict() for h in (file_structure_payload.hotspots if file_structure_payload else [])],
                    }
                    findings_summary = {
                        "counts": finding_summary.counts,
                        "top_rules": sorted(finding_summary.by_rule.items(), key=lambda kv: (-kv[1], kv[0]))[:15],
                        "by_severity": finding_summary.by_severity,
                    }

                    content = llm_chain.synthesize_deep_flow_analysis(
                        dependency_report=deps_graph,
                        structure_summary=structure_summary,
                        findings_summary=findings_summary,
                        security_summary=security_report.to_dict(),
                        sampled_file_summaries=sampled,
                        line_target=detail_line_target,
                        security_lens=security_lens,
                    )
                except Exception as exc:
                    flow_warnings.append(f"LLM flow synthesis failed; fallback used: {exc}")
                    mode = "local-fallback"
                    content = ""

            if not content.strip():
                flow_warnings.append("Deep flow analysis generated via local fallback; may be shorter than target.")
                content = self._render_local_fallback_flow_analysis(
                    deps_report=deps_graph,
                    structure_report=structure_report,
                    describe_report=describe_report,
                    security_lens=security_lens,
                )

            flow_payload = FlowAnalysis(
                mode=mode,
                line_target=detail_line_target,
                security_lens=security_lens,
                content_markdown=content,
                warnings=flow_warnings,
            )

            # Add flow warnings to top-level warnings list (keeps behavior transparent)
            warnings.extend(flow_warnings)

        summary = ProjectReportSummary(
            languages=deps_graph.languages,
            frameworks=deps_graph.frameworks,
            technologies=deps_graph.technologies,
            dependency_counts={
                "runtime": len(deps_graph.runtime_dependencies),
                "dev": len(deps_graph.dev_dependencies),
                "inventory_total": len(inventory),
            },
            finding_counts=finding_summary.counts,
            security_counts=security_report.compute_counts(),
            status=self._derive_status(finding_summary, security_report, warnings),
        )

        meta = ProjectReportMeta(
            project_root=str(project_root),
            generated_at=_iso_utc_now(),
            tool_version=_safe_version_string(),
            online=online,
            llm_enabled=with_llm,
            output_format="both",
            limitations=[
                "Direct dependencies only (no transitive lockfile scan in v1).",
                "Static analysis is Python-only in v1.",
                "OSV matches without exact version are labeled potential.",
            ],
        )

        return ProjectAuditReport(
            meta=meta,
            summary=summary,
            project_summary={
                "describe": describe_report.to_dict(),
                "structure": structure_report.to_dict() if structure_report else None,
                # DEEP: additive fields
                "file_structure": file_structure_payload.to_dict() if file_structure_payload else None,
                "flow_analysis": flow_payload.to_dict() if flow_payload else None,
            },
            dependencies={
                "graph": deps_graph.to_dict(),
                "inventory": [d.to_dict() for d in inventory],
                "inventory_by_scope": {
                    "runtime": [d.to_dict() for d in inventory_runtime],
                    "dev": [d.to_dict() for d in inventory_dev],
                },
            },
            findings={
                "static_findings": [f.to_dict() for f in static_findings],
                "llm_findings": [f.to_dict() for f in llm_findings],
                "merged_findings": [f.to_dict() for f in merged_findings],
                "by_rule": finding_summary.by_rule,
                "by_severity": finding_summary.by_severity,
            },
            security=security_report.to_dict(),
            warnings=warnings,
        )

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

    def _render_local_fallback_flow_analysis(self, *, deps_report, structure_report, describe_report, security_lens: str) -> str:
        lines: list[str] = []
        lines.append("## System Flow & Attack Surface (Defensive Red-Team)")
        lines.append("")
        lines.append("> Local fallback synthesis (LLM disabled/unavailable). Defensive-only; no exploit instructions.")
        lines.append("")
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
            lang_counts = fs.get("language_counts") or {}
            if lang_counts:
                top = list(lang_counts.items())[:12]
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