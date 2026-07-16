"""Codex integration errors."""


class CodexError(RuntimeError):
    pass


class CodexUnavailableError(CodexError):
    pass


class CodexProtocolError(CodexError):
    pass


class CodexExecutionError(CodexError):
    pass


__all__ = ["CodexError", "CodexExecutionError", "CodexProtocolError", "CodexUnavailableError"]
