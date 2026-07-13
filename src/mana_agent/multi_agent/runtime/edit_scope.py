"""Deterministic runtime policy for model-selected explicit edit targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


TaskScope = Literal["direct_edit", "localized_change", "cross_file_change", "architecture_change", "unknown"]
PathResolutionMethod = Literal["exact", "case_insensitive", "missing", "ambiguous", "rejected"]


@dataclass(frozen=True, slots=True)
class PathResolution:
    requested_path: str
    resolved_path: str = ""
    method: PathResolutionMethod = "missing"
    matches: tuple[str, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.resolved_path) and self.method in {"exact", "case_insensitive"}


@dataclass(frozen=True, slots=True)
class ScopeBudget:
    scope: TaskScope
    max_initial_searches: int
    max_initial_read_files: int
    max_direct_dependencies: int
    architecture_evidence_allowed: bool
    mutation_plan_limit: int
    patch_retry_limit: int


SCOPE_BUDGETS: dict[TaskScope, ScopeBudget] = {
    "direct_edit": ScopeBudget("direct_edit", 0, 2, 1, False, 1, 1),
    "localized_change": ScopeBudget("localized_change", 0, 4, 2, False, 1, 1),
    "cross_file_change": ScopeBudget("cross_file_change", 1, 8, 4, False, 2, 1),
    "architecture_change": ScopeBudget("architecture_change", 4, 16, 8, True, 3, 1),
    "unknown": ScopeBudget("unknown", 1, 8, 4, False, 1, 1),
}


def normalize_explicit_path(path: str) -> str:
    text = str(path or "").strip().strip("`'\"").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def resolve_repo_path(repo_root: Path, requested_path: str) -> PathResolution:
    """Resolve a path component-by-component without content search or inventory."""

    root = Path(repo_root).resolve()
    normalized = normalize_explicit_path(requested_path)
    candidate = Path(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or candidate.is_absolute()
        or any(part == ".." for part in candidate.parts)
    ):
        return PathResolution(requested_path=normalized, method="rejected", reason="path_outside_repository")

    current = root
    actual: list[str] = []
    method: PathResolutionMethod = "exact"
    for part in candidate.parts:
        try:
            children = list(current.iterdir())
        except OSError:
            return PathResolution(requested_path=normalized, method="missing", reason="parent_not_readable")
        exact = [child for child in children if child.name == part]
        if len(exact) == 1:
            selected = exact[0]
        else:
            folded = [child for child in children if child.name.casefold() == part.casefold()]
            if len(folded) > 1:
                matches = tuple((Path(*actual) / child.name).as_posix() for child in folded)
                return PathResolution(
                    requested_path=normalized,
                    method="ambiguous",
                    matches=matches,
                    reason="multiple_case_insensitive_matches",
                )
            if not folded:
                return PathResolution(requested_path=normalized, method="missing", reason="path_not_found")
            selected = folded[0]
            method = "case_insensitive"
        current = selected
        actual.append(selected.name)

    try:
        current.resolve().relative_to(root)
    except ValueError:
        return PathResolution(requested_path=normalized, method="rejected", reason="path_outside_repository")
    if not current.is_file():
        return PathResolution(requested_path=normalized, method="missing", reason="not_a_file")
    return PathResolution(requested_path=normalized, resolved_path=Path(*actual).as_posix(), method=method)


def select_scope(*, resolved_targets: Sequence[str], model_scope: str = "", architecture_sync: bool = False) -> TaskScope:
    """Apply bounded runtime policy after the semantic/model scope decision."""

    targets = tuple(dict.fromkeys(str(path) for path in resolved_targets if str(path).strip()))
    if architecture_sync or model_scope == "project_wide":
        return "architecture_change"
    if len(targets) == 1 and model_scope in {"single_file", "single_file_section"}:
        return "direct_edit"
    if 1 < len(targets) <= 3 and model_scope in {"multi_file", "single_file", "single_file_section"}:
        return "localized_change"
    if targets:
        return "cross_file_change"
    return "unknown"


def budget_for_scope(scope: TaskScope) -> ScopeBudget:
    return SCOPE_BUDGETS[scope]


__all__ = [
    "PathResolution",
    "PathResolutionMethod",
    "SCOPE_BUDGETS",
    "ScopeBudget",
    "TaskScope",
    "budget_for_scope",
    "normalize_explicit_path",
    "resolve_repo_path",
    "select_scope",
]
