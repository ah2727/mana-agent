from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mana_agent.ui.streamlit_helpers import (
    get_index_stats,
    get_metrics_summary,
    list_analysis_artifacts,
    load_recent_traces,
    load_taskboard_state,
    run_dashboard_chat,
    safe_read_json,
)
from mana_agent.workspaces.impact import ImpactService
from mana_agent.workspaces.models import WorkspaceSearchRequest
from mana_agent.workspaces.relationships import RelationshipService
from mana_agent.workspaces.search import WorkspaceSearchService
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.skills.adaptive import RepositoryIdentityService, SkillStorage


st.set_page_config(page_title="Mana Agent", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
service = WorkspaceService()
workspaces = service.store.list_workspaces()

st.title("🧠 Mana Agent")
st.caption("Multi-repository workspace and isolated session dashboard")

if not workspaces:
    st.info("No workspace is registered. Run `mana-agent workspace create <name> --root <path>`.")
    st.stop()

workspace_labels = {f"{item.name} · {item.workspace_id}": item for item in workspaces}
selected_workspace = workspace_labels[
    st.sidebar.selectbox("Workspace", list(workspace_labels), key="workspace_selector")
]
repositories = [service.store.get_repository(item) for item in selected_workspace.repository_ids]
if not repositories:
    st.warning("The selected workspace has no repositories.")
    st.stop()
repo_labels = {f"{item.name} · {item.repository_id}": item for item in repositories}
selected_repo = repo_labels[st.sidebar.selectbox("Repository", list(repo_labels), key="repository_selector")]
root = Path(selected_repo.canonical_path)

sessions = [item for item in service.store.list_sessions() if item.workspace_id == selected_workspace.workspace_id]
session_labels = {f"{item.session_id} · {Path(item.cwd).name}": item for item in sessions}
if session_labels:
    selected_session = session_labels[st.sidebar.selectbox("Session", list(session_labels), key="session_selector")]
else:
    selected_session = None
if st.sidebar.button("New isolated session", use_container_width=True):
    created = service.create_session(root, workspace_id=selected_workspace.workspace_id)
    st.session_state.dashboard_session_id = created.session_id
    st.rerun()

pages = ["Overview", "Chat", "Search", "Relationships & Impact", "Taskboard & Traces", "Skills", "Reports"]
page = st.sidebar.radio("Navigation", pages)
st.caption(f"Workspace `{selected_workspace.name}` · Repository `{selected_repo.name}` · `{root}`")

if page == "Overview":
    index = get_index_stats(root)
    metrics = get_metrics_summary(root)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Index", "Ready" if index.get("ready") else "Missing", index.get("chunks", 0))
    c2.metric("Sessions", len(sessions))
    c3.metric("Tasks", metrics.get("task_count", 0))
    c4.metric("Success", f"{metrics.get('success_rate', 0)}%")
    st.subheader("Repository metadata")
    st.json(selected_repo.model_dump(mode="json"))

elif page == "Chat":
    st.header("Repository chat")
    st.caption("The active repository remains the tool boundary; cross-repository evidence is selected through workspace search.")
    key = f"chat_{selected_session.session_id if selected_session else selected_repo.repository_id}"
    messages = st.session_state.setdefault(key, [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask about this repository"):
        messages.append({"role": "user", "content": prompt})
        result = run_dashboard_chat(prompt, root=root)
        answer = str(result.get("answer") or "No answer returned.")
        messages.append({"role": "assistant", "content": answer})
        st.rerun()

elif page == "Search":
    st.header("Workspace search")
    query = st.text_input("Query")
    mode = st.selectbox("Mode", ["text", "file", "symbol", "semantic"])
    selected_ids = st.multiselect(
        "Repositories",
        [item.repository_id for item in repositories],
        default=[item.repository_id for item in repositories],
        format_func=lambda value: service.store.get_repository(value).name,
    )
    if st.button("Search", type="primary") and query:
        semantic = None
        if mode == "semantic":
            from mana_agent.commands.cli_internal import Settings, build_search_service

            semantic = build_search_service(Settings())
        result = WorkspaceSearchService(semantic=semantic).search(
            WorkspaceSearchRequest(
                workspace_id=selected_workspace.workspace_id,
                query=query,
                mode=mode,
                repository_ids=selected_ids,
                limit=50,
            )
        )
        st.json(result)

elif page == "Relationships & Impact":
    st.header("Repository relationships")
    relation_service = RelationshipService(service.store)
    if st.button("Refresh relationship graph"):
        relation_service.detect(selected_workspace)
    relationships = relation_service.list(selected_workspace.workspace_id)
    st.json([item.model_dump(mode="json") for item in relationships])
    st.subheader("Impact analysis")
    changed = st.text_area("Changed repository-relative paths", placeholder="src/api.py\ncontracts/schema.json")
    if st.button("Analyze impact") and changed.strip():
        report = ImpactService(service.store).analyze(
            selected_workspace.workspace_id,
            selected_repo.repository_id,
            [item.strip() for item in changed.splitlines() if item.strip()],
        )
        st.json(report.model_dump(mode="json"))

elif page == "Taskboard & Traces":
    st.header("Taskboard")
    st.json(load_taskboard_state(root))
    st.header("Recent traces")
    st.json(load_recent_traces(root, limit=10))

elif page == "Skills":
    st.header("Adaptive repository skills")
    identity = RepositoryIdentityService().identify(root)
    storage = SkillStorage()
    st.caption(f"Repository identity: `{identity.repository_id}` · storage: `{storage.storage_path()}`")
    skills = storage.list(identity.repository_id)
    if not skills:
        st.info("No adaptive skills have been generated for this repository.")
    else:
        st.dataframe([item.model_dump(mode="json") for item in skills], use_container_width=True)
        selected = st.selectbox("Skill", skills, format_func=lambda item: f"{item.name} · {item.status} · {item.version}")
        try:
            _path, manifest, evidence, markdown = storage.load(identity.repository_id, selected.id)
            st.markdown(markdown)
            with st.expander("Manifest and evidence"):
                st.json({"manifest": manifest.model_dump(mode="json"), "evidence": evidence.model_dump(mode="json")})
            if selected.status == "candidate" and st.button("Approve and activate"):
                storage.activate(identity.repository_id, selected.id)
                st.rerun()
            if selected.status == "active" and st.button("Archive"):
                storage.transition(identity.repository_id, selected.id, "archived")
                st.rerun()
        except (KeyError, OSError, ValueError) as exc:
            st.error(f"Could not read skill: {exc}")

elif page == "Reports":
    st.header("Repository reports")
    artifacts = list_analysis_artifacts(root)
    if not artifacts:
        st.info("No analysis artifacts are available.")
    for artifact in artifacts:
        with st.expander(artifact["name"]):
            path = Path(artifact["path"])
            if artifact["type"] == "json":
                st.json(safe_read_json(path) or {})
            else:
                st.code(path.read_text(encoding="utf-8", errors="replace")[:10_000])
