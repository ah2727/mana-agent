from __future__ import annotations

import re

_CODING_RE = re.compile(r"\b(edit|write|create|delete|fix|implement|refactor|patch|change|update)\b", re.I)
_PLAN_RE = re.compile(r"(^|\s)(/plan|plan|design|proposal)\b", re.I)
_ANALYZE_RE = re.compile(r"(^|\s)(/analyze|analyze|audit|inspect|review repo)\b", re.I)
_TOOL_RE = re.compile(r"\b(run|shell|command|pytest|ruff|mypy|git status|git diff)\b", re.I)
_GIT_RISK_RE = re.compile(r"\b(git reset|git clean|sudo|rm -rf|force push)\b", re.I)


def classify_request(text: str) -> str:
    request = str(text or "").strip()
    if _GIT_RISK_RE.search(request):
        return "high_risk_tool"
    if _ANALYZE_RE.search(request):
        return "analyze"
    if _PLAN_RE.search(request):
        return "planning"
    if _CODING_RE.search(request):
        return "coding"
    if _TOOL_RE.search(request):
        return "tool"
    return "simple"
