from __future__ import annotations

from collections import deque

from mana_agent.workspaces.models import ImpactNode, ImpactReport
from mana_agent.workspaces.relationships import RelationshipService
from mana_agent.workspaces.store import WorkspaceStore


class ImpactService:
    def __init__(self, store: WorkspaceStore | None = None) -> None:
        self.store = store or WorkspaceStore()
        self.relationships = RelationshipService(self.store)

    def analyze(
        self,
        workspace_id: str,
        source_repository_id: str,
        changed_paths: list[str],
        *,
        max_depth: int = 3,
    ) -> ImpactReport:
        workspace = self.store.get_workspace(workspace_id)
        if source_repository_id not in workspace.repository_ids:
            raise ValueError("source repository is not a workspace member")
        relationships = self.relationships.list(workspace_id) or self.relationships.detect(workspace)
        outgoing: dict[str, list] = {}
        # Consumers of a changed dependency are affected, so traverse reverse edges.
        for edge in relationships:
            outgoing.setdefault(edge.target_repository_id, []).append(edge)
        affected: list[ImpactNode] = []
        queue = deque([(source_repository_id, 0)])
        seen = {source_repository_id}
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in outgoing.get(current, []):
                target = edge.source_repository_id
                if target in seen:
                    continue
                seen.add(target)
                affected.append(
                    ImpactNode(
                        repository_id=target,
                        reason=f"depends on {current} via {edge.kind}",
                        depth=depth + 1,
                        confidence=edge.confidence,
                    )
                )
                queue.append((target, depth + 1))
        verification: dict[str, list[str]] = {}
        for repository_id in seen:
            repo = self.store.get_repository(repository_id)
            commands: list[str] = []
            if "python" in repo.languages:
                commands.append("python -m pytest -q")
            if {"javascript", "typescript"} & set(repo.languages):
                commands.append("npm test")
            if "go" in repo.languages:
                commands.append("go test ./...")
            if "rust" in repo.languages:
                commands.append("cargo test")
            if "dart" in repo.languages:
                commands.append("dart test")
            verification[repository_id] = commands or ["model-selected project verification required"]
        return ImpactReport(
            workspace_id=workspace_id,
            source_repository_id=source_repository_id,
            changed_paths=changed_paths,
            affected=affected,
            verification_by_repository=verification,
        )
