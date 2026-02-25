from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from mana_analyzer.config.settings import Settings, default_index_dir
from mana_analyzer.services.ask_service import AskService

logger = logging.getLogger(__name__)


class ChatService:
    """
    ChatService wraps a fully-configured AskService and provides a simple
    ask() API for interactive CLI chat.

    IMPORTANT:
    - This class does NOT build AskService itself.
    - Pass in the AskService created by build_ask_service(...) from CLI.
    """

    def __init__(
        self,
        *,
        ask_service: AskService,
        settings: Settings,
        model_override: Optional[str] = None,
        index_dir: Optional[Union[str, Path]] = None,
        dir_mode: bool = False,
        root_dir: Optional[Union[str, Path]] = None,
        k: Optional[int] = None,
        agent_tools: bool = False,
        agent_max_steps: int = 6,
        agent_timeout_seconds: int = 30,
        # dir-mode options
        max_indexes: int = 0,
        auto_index_missing: bool = True,
    ) -> None:
        self._ask_service = ask_service
        self._settings = settings
        self._model_override = model_override

        self._k = int(k or settings.default_top_k)
        self._agent_tools = bool(agent_tools)
        self._agent_max_steps = int(agent_max_steps)
        self._agent_timeout_seconds = int(agent_timeout_seconds)

        self._dir_mode = bool(dir_mode)
        self._root_dir: Path = (
            Path(root_dir).expanduser().resolve() if root_dir is not None else Path.cwd().resolve()
        )

        self._history: List[tuple[str, str]] = []

        if self._dir_mode:
            # Let AskService handle discovery/auto-indexing in dir-mode,
            # because your CLI already has this logic and AskService expects search_service configured.
            # We will select indexes in the CLI and pass them to ask_dir_mode.
            self._index_dirs: List[Path] = []
            self._max_indexes = int(max_indexes)
            self._auto_index_missing = bool(auto_index_missing)
        else:
            resolved = (
                Path(index_dir).expanduser().resolve()
                if index_dir is not None
                else default_index_dir(Path.cwd())
            )
            self._index_dirs = [resolved]
            self._max_indexes = 0
            self._auto_index_missing = False

    def set_index_dirs(self, index_dirs: List[Path]) -> None:
        """Used by the CLI to supply the computed dir-mode index list."""
        self._index_dirs = [Path(p).resolve() for p in index_dirs]
    def ask(
        self,
        question: str,
        *,
        callbacks: Sequence[Any] | None = None,
        **kwargs: Any,
    ):
        """
        Ask a question using either:
          - dir-mode: multiple indexes via search_service / agent tools, or
          - single-index mode: one index via store / agent tools.

        `callbacks` is optional and is forwarded when supported by downstream services.
        If downstream does not support callbacks, we gracefully retry without them.
        """
        question = (question or "").strip()
        if not question:
            return None

        # Helper: call a function with callbacks if supported; otherwise retry without.
        def _call_with_optional_callbacks(fn, /, **call_kwargs):
            if callbacks is None:
                return fn(**call_kwargs)
            try:
                return fn(**call_kwargs, callbacks=callbacks)
            except TypeError:
                # Downstream method does not accept callbacks
                return fn(**call_kwargs)

        if self._dir_mode:
            if not self._index_dirs:
                raise RuntimeError(
                    "No indexes configured for dir-mode chat. "
                    "Compute selected indexes in CLI and call chat_service.set_index_dirs(...)."
                )

            if self._agent_tools:
                # Tool/agent dir-mode path
                response = _call_with_optional_callbacks(
                    self._ask_service.ask_with_tools_dir_mode,
                    index_dirs=self._index_dirs,
                    question=question,
                    k=self._k,
                    max_steps=self._agent_max_steps,
                    timeout_seconds=self._agent_timeout_seconds,
                    root_dir=self._root_dir,
                )
            else:
                # Classic dir-mode path (usually no callbacks, but we allow if you add it later)
                response = _call_with_optional_callbacks(
                    self._ask_service.ask_dir_mode,
                    index_dirs=self._index_dirs,
                    question=question,
                    k=self._k,
                    root_dir=self._root_dir,
                )

        else:
            if not self._index_dirs:
                raise RuntimeError(
                    "No index configured for chat. "
                    "Compute selected index in CLI and call chat_service.set_index_dirs(...)."
                )

            index_dir = self._index_dirs[0]

            if self._agent_tools:
                # Tool/agent single-index path
                response = _call_with_optional_callbacks(
                    self._ask_service.ask_with_tools,
                    index_dir=index_dir,
                    question=question,
                    k=self._k,
                    max_steps=self._agent_max_steps,
                    timeout_seconds=self._agent_timeout_seconds,
                )
            else:
                # Classic single-index path
                response = _call_with_optional_callbacks(
                    self._ask_service.ask,
                    index_dir=index_dir,
                    question=question,
                    k=self._k,
                )

        # Record history safely
        try:
            answer_text = getattr(response, "answer", None)
            if answer_text is not None:
                self._history.append((question, answer_text))
        except Exception:
            pass

        return response