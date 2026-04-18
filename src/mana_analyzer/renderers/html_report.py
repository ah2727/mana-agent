from __future__ import annotations

import html
import json
from typing import Any, Iterable


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "section"


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _badge(label: str, tone: str = "neutral") -> str:
    return f'<span class="badge badge-{_escape(tone)}">{_escape(label)}</span>'


def _stat_card(label: str, value: Any, tone: str = "neutral") -> str:
    return (
        '<article class="stat-card">'
        f'<div class="stat-label">{_escape(label)}</div>'
        f'<div class="stat-value tone-{_escape(tone)}">{_escape(value)}</div>'
        "</article>"
    )


def _details_block(title: str, body: str, *, open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f'<details class="detail-block"{open_attr}>'
        f'<summary>{_escape(title)}</summary>'
        f'<div class="detail-content">{body}</div>'
        "</details>"
    )


def _list_items(values: Iterable[Any], *, empty: str = "None") -> str:
    items = [f"<li>{_escape(value)}</li>" for value in values if value not in (None, "")]
    if not items:
        return f'<p class="empty-state">{_escape(empty)}</p>'
    return '<ul class="clean-list">' + "".join(items) + "</ul>"


def _kv_grid(items: Iterable[tuple[str, Any]], *, empty: str = "No data") -> str:
    rows = []
    for key, value in items:
        if value in (None, "", [], {}, ()):  # skip empty noise
            continue
        rows.append(
            '<div class="kv-row">'
            f'<div class="kv-key">{_escape(key)}</div>'
            f'<div class="kv-value">{_escape(value)}</div>'
            "</div>"
        )
    if not rows:
        return f'<p class="empty-state">{_escape(empty)}</p>'
    return '<div class="kv-grid">' + "".join(rows) + "</div>"


def _code_block(text: str, *, block_id: str) -> str:
    return (
        '<div class="code-shell">'
        f'<button class="copy-button" type="button" data-copy-target="{_escape(block_id)}">Copy</button>'
        f'<pre id="{_escape(block_id)}"><code>{_escape(text)}</code></pre>'
        "</div>"
    )


def _section(title: str, body: str, *, kicker: str | None = None) -> tuple[str, str]:
    section_id = _slugify(title)
    kicker_html = f'<div class="section-kicker">{_escape(kicker)}</div>' if kicker else ""
    markup = (
        f'<section id="{_escape(section_id)}" class="content-section">'
        f'{kicker_html}<h2>{_escape(title)}</h2>{body}'
        "</section>"
    )
    return title, markup


def _table(columns: list[str], rows: list[list[Any]], *, empty: str = "No rows") -> str:
    if not rows:
        return f'<p class="empty-state">{_escape(empty)}</p>'
    header = "".join(f"<th>{_escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def _document(*, title: str, subtitle: str, badges: list[str], stats: list[str], sections: list[tuple[str, str]]) -> str:
    nav = "".join(
        f'<a href="#{_escape(_slugify(name))}">{_escape(name)}</a>' for name, _markup in sections
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --bg-accent: radial-gradient(circle at top left, rgba(187, 110, 65, 0.16), transparent 28%), linear-gradient(180deg, #fcfaf5 0%, #f3ede3 100%);
      --paper: rgba(255, 252, 246, 0.9);
      --paper-strong: #fffdf8;
      --ink: #1f2933;
      --muted: #5a6672;
      --line: rgba(31, 41, 51, 0.12);
      --accent: #0f766e;
      --accent-soft: rgba(15, 118, 110, 0.12);
      --error: #b42318;
      --warning: #b54708;
      --ok: #027a48;
      --shadow: 0 20px 60px rgba(75, 57, 42, 0.12);
      --radius: 22px;
      --radius-sm: 14px;
      --mono: "SFMono-Regular", "Consolas", "Liberation Mono", monospace;
      --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; font-family: var(--sans); color: var(--ink); background: var(--bg); background-image: var(--bg-accent); }}
    a {{ color: inherit; text-decoration: none; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 60px; }}
    .hero {{ background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(247,240,229,0.96)); border: 1px solid var(--line); border-radius: calc(var(--radius) + 6px); padding: 30px; box-shadow: var(--shadow); position: relative; overflow: hidden; }}
    .hero::after {{ content: ""; position: absolute; inset: auto -40px -60px auto; width: 220px; height: 220px; background: radial-gradient(circle, rgba(15,118,110,0.16), transparent 68%); pointer-events: none; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.18em; font-size: 11px; color: var(--muted); margin-bottom: 10px; }}
    h1 {{ margin: 0; font-size: clamp(2rem, 4vw, 3.6rem); line-height: 0.96; font-family: var(--serif); font-weight: 700; max-width: 11ch; }}
    .subtitle {{ color: var(--muted); font-size: 1rem; line-height: 1.6; max-width: 72ch; margin-top: 14px; }}
    .badge-row, .stat-grid {{ display: grid; gap: 12px; }}
    .badge-row {{ grid-template-columns: repeat(auto-fit, minmax(140px, max-content)); margin-top: 20px; }}
    .badge {{ display: inline-flex; align-items: center; gap: 8px; padding: 9px 12px; border-radius: 999px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); font-size: 0.88rem; }}
    .badge-neutral {{ color: var(--muted); }}
    .badge-ok {{ color: var(--ok); background: rgba(2,122,72,0.1); }}
    .badge-warning {{ color: var(--warning); background: rgba(181,71,8,0.1); }}
    .badge-error {{ color: var(--error); background: rgba(180,35,24,0.1); }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 24px; margin-top: 26px; align-items: start; }}
    .main {{ display: grid; gap: 20px; }}
    .sidebar {{ position: sticky; top: 16px; }}
    .nav-card, .content-section, .appendix-card {{ background: var(--paper); backdrop-filter: blur(6px); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); }}
    .nav-card {{ padding: 18px; }}
    .nav-card h3 {{ margin: 0 0 12px; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); }}
    .nav-card nav {{ display: grid; gap: 8px; }}
    .nav-card a {{ padding: 10px 12px; border-radius: 12px; color: var(--muted); transition: background 140ms ease, color 140ms ease, transform 140ms ease; }}
    .nav-card a:hover, .nav-card a:focus-visible, .nav-card a.active {{ background: var(--accent-soft); color: var(--accent); transform: translateX(2px); outline: none; }}
    .stat-grid {{ grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-top: 22px; }}
    .stat-card {{ background: var(--paper-strong); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 18px; min-height: 120px; display: flex; flex-direction: column; justify-content: space-between; }}
    .stat-label {{ font-size: 0.82rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat-value {{ font-size: clamp(1.5rem, 3vw, 2.45rem); line-height: 1; font-weight: 700; }}
    .tone-neutral {{ color: var(--ink); }}
    .tone-ok {{ color: var(--ok); }}
    .tone-warning {{ color: var(--warning); }}
    .tone-error {{ color: var(--error); }}
    .content-section {{ padding: 24px; }}
    .content-section h2 {{ margin: 0 0 14px; font-family: var(--serif); font-size: 1.8rem; }}
    .section-kicker {{ text-transform: uppercase; letter-spacing: 0.18em; font-size: 11px; color: var(--muted); margin-bottom: 10px; }}
    .lede {{ color: var(--muted); line-height: 1.7; margin: 0 0 16px; }}
    .split-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .panel {{ border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: rgba(255,255,255,0.62); }}
    .panel h3 {{ margin: 0 0 10px; font-size: 1rem; }}
    .panel p {{ margin: 0; color: var(--muted); line-height: 1.7; }}
    .clean-list {{ margin: 0; padding-left: 18px; display: grid; gap: 8px; }}
    .empty-state {{ margin: 0; color: var(--muted); font-style: italic; }}
    .kv-grid {{ display: grid; gap: 10px; }}
    .kv-row {{ display: grid; grid-template-columns: minmax(120px, 180px) minmax(0, 1fr); gap: 14px; padding: 10px 0; border-bottom: 1px solid rgba(31,41,51,0.08); }}
    .kv-row:last-child {{ border-bottom: 0; }}
    .kv-key {{ color: var(--muted); font-size: 0.92rem; }}
    .kv-value {{ font-weight: 600; word-break: break-word; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.58); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 620px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid rgba(31,41,51,0.08); vertical-align: top; }}
    th {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); background: rgba(255,255,255,0.7); }}
    tr:last-child td {{ border-bottom: 0; }}
    .detail-block {{ border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.56); }}
    .detail-block + .detail-block {{ margin-top: 12px; }}
    .detail-block summary {{ list-style: none; cursor: pointer; padding: 16px 18px; font-weight: 700; }}
    .detail-block summary::-webkit-details-marker {{ display: none; }}
    .detail-content {{ padding: 0 18px 18px; }}
    .code-shell {{ position: relative; }}
    .copy-button {{ position: absolute; right: 12px; top: 12px; border: 0; border-radius: 999px; padding: 8px 12px; background: var(--accent); color: white; cursor: pointer; font: inherit; }}
    pre {{ margin: 0; padding: 22px 16px 16px; overflow-x: auto; background: #17212b; color: #f9fafb; border-radius: 18px; font-family: var(--mono); font-size: 0.9rem; line-height: 1.55; white-space: pre-wrap; word-break: break-word; }}
    code {{ font-family: var(--mono); }}
    .chip-list {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .chip {{ border-radius: 999px; padding: 8px 12px; background: rgba(15,118,110,0.08); color: var(--accent); border: 1px solid rgba(15,118,110,0.16); }}
    .appendix-card {{ padding: 18px; }}
    .button-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }}
    .ghost-button {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; padding: 10px 14px; border: 1px solid var(--line); background: rgba(255,255,255,0.7); cursor: pointer; font: inherit; color: var(--ink); }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; order: -1; }}
      h1 {{ max-width: none; }}
    }}
    @media (max-width: 640px) {{
      .page {{ padding: 18px 14px 40px; }}
      .hero, .content-section, .nav-card, .appendix-card {{ border-radius: 18px; padding: 18px; }}
      .kv-row {{ grid-template-columns: 1fr; gap: 6px; }}
      .copy-button {{ position: static; margin: 12px 0 0; }}
      pre {{ padding-top: 16px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      html {{ scroll-behavior: auto; }}
      .nav-card a {{ transition: none; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="eyebrow">mana-analyzer html export</div>
      <h1>{_escape(title)}</h1>
      <p class="subtitle">{_escape(subtitle)}</p>
      <div class="badge-row">{''.join(badges)}</div>
      <div class="stat-grid">{''.join(stats)}</div>
    </header>
    <div class="layout">
      <main class="main">
        {''.join(markup for _name, markup in sections)}
      </main>
      <aside class="sidebar">
        <div class="nav-card">
          <h3>Jump to</h3>
          <nav>{nav}</nav>
          <div class="button-row">
            <button class="ghost-button" type="button" data-toggle-details="open">Expand all</button>
            <button class="ghost-button" type="button" data-toggle-details="close">Collapse all</button>
          </div>
        </div>
      </aside>
    </div>
  </div>
  <script>
    const navLinks = [...document.querySelectorAll('.nav-card a')];
    const sections = navLinks.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {{
      entries.forEach((entry) => {{
        const id = entry.target.getAttribute('id');
        const link = document.querySelector(`.nav-card a[href="#${{id}}"]`);
        if (link && entry.isIntersecting) {{
          navLinks.forEach(item => item.classList.remove('active'));
          link.classList.add('active');
        }}
      }});
    }}, {{ rootMargin: '-25% 0px -60% 0px', threshold: 0.1 }});
    sections.forEach(section => observer.observe(section));

    document.querySelectorAll('[data-copy-target]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const target = document.getElementById(button.dataset.copyTarget);
        if (!target) return;
        const text = target.innerText;
        try {{
          await navigator.clipboard.writeText(text);
          const original = button.textContent;
          button.textContent = 'Copied';
          setTimeout(() => button.textContent = original, 1200);
        }} catch (_error) {{}}
      }});
    }});

    document.querySelectorAll('[data-toggle-details]').forEach((button) => {{
      button.addEventListener('click', () => {{
        const shouldOpen = button.dataset.toggleDetails === 'open';
        document.querySelectorAll('details').forEach((node) => {{ node.open = shouldOpen; }});
      }});
    }});
  </script>
</body>
</html>
"""


def render_analyze_html(payload: dict[str, Any], markdown: str) -> str:
    findings = payload.get("findings") or []
    summarization = payload.get("summarization") or {}
    structure_keys = [key for key in payload.keys() if key not in {"findings", "summarization", "tech", "project_structure_analysis"}]
    project_analysis = payload.get("project_structure_analysis") or {}
    tech = payload.get("tech") or {}

    sections = [
        _section(
            "Overview",
            '<p class="lede">A browser-friendly pass over analyzer findings, repository structure, and generated project notes.</p>'
            + '<div class="split-grid">'
            + '<article class="panel"><h3>Architecture</h3><p>'
            + _escape(summarization.get("architecture_summary") or "Architecture summary unavailable.")
            + '</p></article>'
            + '<article class="panel"><h3>Technology</h3><p>'
            + _escape(summarization.get("tech_summary") or "Technology summary unavailable.")
            + '</p></article></div>',
            kicker="Executive pass",
        ),
        _section(
            "Findings",
            _table(
                ["Severity", "Rule", "Location", "Message"],
                [
                    [
                        item.get("severity", "unknown"),
                        item.get("rule_id", "unknown"),
                        f"{item.get('file_path', 'unknown')}:{item.get('line', '?')}:{item.get('column', '?')}",
                        item.get("message", ""),
                    ]
                    for item in findings
                ],
                empty="No findings recorded.",
            ),
            kicker="Issues",
        ),
        _section(
            "Repository Summary",
            _kv_grid((key.replace("_", " ").title(), summarization.get(key)) for key in sorted(summarization.keys())),
            kicker="Metadata",
        ),
        _section(
            "Technology Snapshot",
            _kv_grid(
                [
                    ("Languages", ", ".join(tech.get("languages") or []) or "unknown"),
                    ("File count", tech.get("file_count", 0)),
                    ("Chain profile", tech.get("chain_profile", "default")),
                    ("Chain config", tech.get("chain_config", "")),
                ]
            ),
            kicker="Inputs",
        ),
        _section(
            "Structure Payload",
            _details_block(
                "Show raw structure payload",
                _code_block(_pretty_json({key: payload.get(key) for key in structure_keys}), block_id="analyze-structure-json"),
                open_by_default=False,
            ),
            kicker="Structure",
        ),
        _section(
            "Project Structure Analysis",
            _code_block("\n".join(project_analysis.get("analysis_lines") or []), block_id="analyze-psa")
            if project_analysis.get("analysis_lines")
            else '<p class="empty-state">Project structure analysis unavailable.</p>',
            kicker="Generated narrative",
        ),
        _section(
            "Markdown Appendix",
            _details_block("Show generated markdown", _code_block(markdown, block_id="analyze-markdown"), open_by_default=False),
            kicker="Raw export",
        ),
    ]

    return _document(
        title="Analyze Report",
        subtitle="Self-contained HTML output for repository analysis with findings, structure, and generated summaries.",
        badges=[
            _badge(f"{len(findings)} findings", "warning" if findings else "ok"),
            _badge(f"{len(tech.get('languages') or [])} languages", "neutral"),
            _badge(f"{len(structure_keys)} structure blocks", "neutral"),
        ],
        stats=[
            _stat_card("Findings", len(findings), "warning" if findings else "ok"),
            _stat_card("Files", tech.get("file_count", 0), "neutral"),
            _stat_card("Languages", len(tech.get("languages") or []), "neutral"),
            _stat_card("PSA Lines", project_analysis.get("line_count", 0), "neutral"),
        ],
        sections=sections,
    )


def render_describe_html(payload: dict[str, Any], markdown: str) -> str:
    descriptions = payload.get("descriptions") or payload.get("files") or []
    selected_files = payload.get("selected_files") or [item.get("file_path") or item.get("path") for item in descriptions]
    metrics = payload.get("metrics") or {}
    detail_blocks = []
    for idx, item in enumerate(descriptions):
        file_path = item.get("file_path") or item.get("path") or f"file-{idx + 1}"
        body = (
            '<div class="split-grid">'
            '<article class="panel"><h3>Summary</h3><p>'
            + _escape(item.get("summary") or "No summary available.")
            + '</p></article>'
            '<article class="panel"><h3>Details</h3>'
            + _kv_grid(
                [
                    ("Language", item.get("language", "unknown")),
                    ("Symbols", ", ".join(item.get("symbols") or []) or "none"),
                    ("Imports", ", ".join(item.get("imports") or []) or "none"),
                ],
                empty="No details",
            )
            + '</article></div>'
        )
        detail_blocks.append(_details_block(file_path, body, open_by_default=idx < 3))

    sections = [
        _section(
            "Overview",
            '<div class="split-grid">'
            '<article class="panel"><h3>Architecture</h3><p>'
            + _escape(payload.get("architecture_summary") or "Architecture summary unavailable.")
            + '</p></article>'
            '<article class="panel"><h3>Technology</h3><p>'
            + _escape(payload.get("tech_summary") or "Technology summary unavailable.")
            + '</p></article></div>',
            kicker="Repository shape",
        ),
        _section(
            "Coverage",
            _kv_grid(
                [
                    ("Project root", payload.get("project_root") or payload.get("root") or "unknown"),
                    ("Selected files", len(selected_files)),
                    ("Descriptions", len(descriptions)),
                    ("Chain steps", ", ".join(payload.get("chain_steps") or []) or "none"),
                    ("Cache hits", metrics.get("cache_hits", 0)),
                ]
            )
            + '<div class="chip-list">'
            + "".join(f'<span class="chip">{_escape(item)}</span>' for item in selected_files[:24])
            + "</div>",
            kicker="Selection",
        ),
        _section(
            "File Descriptions",
            "".join(detail_blocks) or '<p class="empty-state">No file descriptions available.</p>',
            kicker="Narrative inventory",
        ),
        _section(
            "Architecture Diagram",
            _code_block(str(payload.get("architecture_mermaid") or "No diagram available."), block_id="describe-mermaid"),
            kicker="Diagram source",
        ),
        _section(
            "Markdown Appendix",
            _details_block("Show generated markdown", _code_block(markdown, block_id="describe-markdown"), open_by_default=False),
            kicker="Raw export",
        ),
    ]

    return _document(
        title="Repository Description",
        subtitle="A polished HTML companion for describe output, tuned for browsing file summaries and architecture context.",
        badges=[
            _badge(f"{len(descriptions)} file summaries", "ok" if descriptions else "warning"),
            _badge(f"{len(selected_files)} selected files", "neutral"),
            _badge(f"{len(payload.get('chain_steps') or [])} chain steps", "neutral"),
        ],
        stats=[
            _stat_card("Files", len(selected_files), "neutral"),
            _stat_card("Descriptions", len(descriptions), "ok" if descriptions else "warning"),
            _stat_card("Symbols", sum(len(item.get("symbols") or []) for item in descriptions), "neutral"),
            _stat_card("Cache Hits", metrics.get("cache_hits", 0), "neutral"),
        ],
        sections=sections,
    )


def render_report_html(payload: dict[str, Any], markdown: str) -> str:
    meta = payload.get("meta") or {}
    summary = payload.get("summary") or {}
    project_summary = payload.get("project_summary") or {}
    describe = project_summary.get("describe") or {}
    findings = payload.get("findings") or {}
    security = payload.get("security") or {}
    vulnerabilities_by_scope = security.get("vulnerabilities_by_scope") or {}
    warnings = payload.get("warnings") or []
    merged_findings = findings.get("merged_findings") or []
    runtime_vulns = vulnerabilities_by_scope.get("runtime") or []
    dev_vulns = vulnerabilities_by_scope.get("dev") or []
    file_structure = project_summary.get("file_structure") or {}
    flow_analysis = project_summary.get("flow_analysis") or {}

    sections = [
        _section(
            "Executive Summary",
            '<div class="split-grid">'
            '<article class="panel"><h3>Architecture</h3><p>'
            + _escape(describe.get("architecture_summary") or "Architecture summary unavailable.")
            + '</p></article>'
            '<article class="panel"><h3>Technology</h3><p>'
            + _escape(describe.get("tech_summary") or "Technology summary unavailable.")
            + '</p></article></div>',
            kicker="At a glance",
        ),
        _section(
            "Repository Profile",
            _kv_grid(
                [
                    ("Project root", meta.get("project_root", "unknown")),
                    ("Generated at", meta.get("generated_at", "unknown")),
                    ("Tool version", meta.get("tool_version", "unknown")),
                    ("Status", summary.get("status", "unknown")),
                    ("Languages", ", ".join(summary.get("languages") or []) or "unknown"),
                    ("Frameworks", ", ".join(summary.get("frameworks") or []) or "none"),
                    ("Technologies", ", ".join(summary.get("technologies") or []) or "none"),
                ]
            ),
            kicker="Metadata",
        ),
        _section(
            "Findings",
            _table(
                ["Severity", "Rule", "Path", "Message"],
                [
                    [
                        item.get("severity", "unknown"),
                        item.get("rule_id", "unknown"),
                        item.get("file_path", "unknown"),
                        item.get("message", ""),
                    ]
                    for item in merged_findings[:200]
                ],
                empty="No merged findings.",
            )
            + _details_block(
                "Top rules",
                _list_items(
                    [f"{rule}: {count}" for rule, count in sorted((findings.get('by_rule') or {}).items(), key=lambda kv: (-kv[1], kv[0]))[:15]],
                    empty="No rule summary available.",
                ),
                open_by_default=True,
            ),
            kicker="Bugs & errors",
        ),
        _section(
            "Security",
            '<div class="split-grid">'
            '<article class="panel"><h3>Runtime vulnerabilities</h3>'
            + _list_items(
                [
                    f"{((item.get('package') or {}).get('name') or 'unknown')} - {item.get('osv_id', 'unknown')} [{item.get('confidence', 'unknown')}]"
                    for item in runtime_vulns[:60]
                ],
                empty="No runtime vulnerabilities listed.",
            )
            + '</article>'
            '<article class="panel"><h3>Dev vulnerabilities</h3>'
            + _list_items(
                [
                    f"{((item.get('package') or {}).get('name') or 'unknown')} - {item.get('osv_id', 'unknown')} [{item.get('confidence', 'unknown')}]"
                    for item in dev_vulns[:60]
                ],
                empty="No dev vulnerabilities listed.",
            )
            + '</article></div>',
            kicker="OSV lens",
        ),
        _section(
            "Deep Structure",
            _details_block(
                "File structure",
                _code_block(file_structure.get("tree_markdown") or "Structure diagram unavailable.", block_id="report-tree"),
                open_by_default=bool(file_structure),
            )
            + _details_block(
                "Flow analysis",
                _code_block(flow_analysis.get("content_markdown") or "Flow analysis unavailable.", block_id="report-flow"),
                open_by_default=bool(flow_analysis),
            ),
            kicker="Deep profile",
        ),
        _section(
            "Warnings & Limitations",
            '<div class="split-grid">'
            '<article class="panel"><h3>Warnings</h3>'
            + _list_items(warnings, empty="No warnings.")
            + '</article>'
            '<article class="panel"><h3>Limitations</h3>'
            + _list_items(meta.get("limitations") or [], empty="No limitations listed.")
            + '</article></div>',
            kicker="Guardrails",
        ),
        _section(
            "Markdown Appendix",
            _details_block("Show generated markdown", _code_block(markdown, block_id="report-markdown"), open_by_default=False)
            + _details_block("Show JSON payload", _code_block(_pretty_json(payload), block_id="report-json"), open_by_default=False),
            kicker="Raw export",
        ),
    ]

    severity_counts = summary.get("finding_counts") or {}
    security_counts = summary.get("security_counts") or {}
    return _document(
        title="Project Audit Report",
        subtitle="A standalone HTML audit view for architecture, findings, and vulnerability posture with a browser-friendly navigation experience.",
        badges=[
            _badge(summary.get("status", "unknown"), "ok" if summary.get("status") == "ok" else "warning"),
            _badge(f"{len(summary.get('languages') or [])} languages", "neutral"),
            _badge("LLM enabled" if meta.get("llm_enabled") else "LLM disabled", "neutral"),
            _badge("Online" if meta.get("online") else "Offline", "neutral"),
        ],
        stats=[
            _stat_card("Findings", severity_counts.get("total", 0), "warning" if severity_counts.get("total", 0) else "ok"),
            _stat_card("Errors", severity_counts.get("error", 0), "error" if severity_counts.get("error", 0) else "neutral"),
            _stat_card("Warnings", severity_counts.get("warning", 0), "warning" if severity_counts.get("warning", 0) else "neutral"),
            _stat_card("Potential Vulns", security_counts.get("potential_vulns", 0), "warning" if security_counts.get("potential_vulns", 0) else "ok"),
        ],
        sections=sections,
    )
