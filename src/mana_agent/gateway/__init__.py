"""Gateway layer for unified access to Mana-Agent multi-agent chat and runtime.

All frontends (TUI chat, Telegram, Dashboard/API, CLI) should connect through
the gateway to reach agents. The gateway centralizes construction of
AskService / ChatService / CodingAgent stacks, auto-chat, and turn orchestration.

See chat_gateway.py for the main implementation.
"""

from .chat_gateway import AgentChatGateway, RichChatContext
from .config import ChatGatewayConfig
from .stack import ChatStack, build_chat_stack
from .turn_engine import (
    ChatTurnResult,
    is_auto_chat_connector_turn,
    process_chat_turn,
    should_use_coding_agent_turn,
)

__all__ = [
    "AgentChatGateway",
    "RichChatContext",
    "ChatGatewayConfig",
    "ChatStack",
    "build_chat_stack",
    "ChatTurnResult",
    "process_chat_turn",
    "should_use_coding_agent_turn",
    "is_auto_chat_connector_turn",
]
