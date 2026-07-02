from __future__ import annotations

import hashlib
from pathlib import Path

from mana_agent.skills.manager import DEFAULT_SKILL_NAMES, SkillManager, detect_skill_names


def _first_meaningful_line(content: str) -> str:
    for line in str(content or "").splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned[:160]
    return "No summary available."


def _content_hash(content: str) -> str:
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()[:12]


def render_compact_skills_index(request: str, *, repo_root: str | Path | None = None, limit: int = 6) -> str:
    manager = SkillManager(project_root=repo_root)
    names = detect_skill_names(request)[: max(1, limit)]
    lines = ["Compact Skills Index"]
    if not names:
        lines.append("- none matched")
        return "\n".join(lines)
    for name in names:
        skill = manager.get(name)
        if skill is None:
            lines.append(f"- {name}: unavailable")
            continue
        lines.append(f"- {skill.name} ({skill.source}, hash={_content_hash(skill.content)}): {_first_meaningful_line(skill.content)}")
    return "\n".join(lines)


def render_stable_skills_index(*, repo_root: str | Path | None = None, limit: int = 24) -> str:
    """Render a compact, request-independent skill index for the stable prompt."""
    manager = SkillManager(project_root=repo_root)
    listed = manager.list_by_source()
    names: list[str] = []
    for group in ("Project Root Skills", "Global Skills", "Built-in Skills"):
        for name in listed.get(group, []):
            if name not in names:
                names.append(name)
            if len(names) >= max(1, limit):
                break
        if len(names) >= max(1, limit):
            break
    if not names:
        names = list(DEFAULT_SKILL_NAMES[: max(1, limit)])

    lines = ["Compact Skills Index", "- Stable index; task-specific skill details belong in ephemeral context."]
    for name in names[: max(1, limit)]:
        skill = manager.get(name)
        if skill is None:
            lines.append(f"- {name}: unavailable")
            continue
        lines.append(f"- {skill.name} ({skill.source}, hash={_content_hash(skill.content)}): {_first_meaningful_line(skill.content)}")
    return "\n".join(lines)
