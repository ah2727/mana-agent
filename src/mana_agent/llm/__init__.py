"""Shared LLM construction and compatibility policy."""

from .compatibility import CompatibleChatOpenAI, ModelCapabilities, create_chat_model, resolve_model_capabilities

__all__ = ["CompatibleChatOpenAI", "ModelCapabilities", "create_chat_model", "resolve_model_capabilities"]
