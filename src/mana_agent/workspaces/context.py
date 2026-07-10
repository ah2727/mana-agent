from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mana_agent.workspaces.models import RepositoryRecord, RepositoryScopeDecision, SessionRecord, WorkspaceRecord


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    workspace: WorkspaceRecord
    session: SessionRecord
    repositories: dict[str, RepositoryRecord]

    @property
    def primary_repository(self) -> RepositoryRecord:
        return self.repositories[self.session.primary_repository_id]

    @property
    def primary_root(self) -> Path:
        return Path(self.primary_repository.canonical_path)

    def repository(self, repository_id: str) -> RepositoryRecord:
        if repository_id not in self.session.attached_repository_ids:
            raise PermissionError(f"repository {repository_id} is not attached to session {self.session.session_id}")
        return self.repositories[repository_id]

    def validate_scope(self, decision: RepositoryScopeDecision) -> RepositoryScopeDecision:
        if decision.workspace_id != self.workspace.workspace_id or decision.session_id != self.session.session_id:
            raise ValueError("scope decision does not match workspace session")
        if decision.primary_repository_id != self.session.primary_repository_id:
            raise ValueError("scope decision primary repository mismatch")
        unknown = set(decision.repository_ids) - set(self.session.attached_repository_ids)
        if unknown:
            raise PermissionError(f"scope decision selected unattached repositories: {', '.join(sorted(unknown))}")
        if not decision.repository_ids:
            raise ValueError("scope decision must select at least one repository")
        if not decision.safe_to_continue:
            raise PermissionError("scope decision is not safe to continue")
        return decision
