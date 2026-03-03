"""User context tool for getting current user identifiers."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.tools.base import Tool


class UserTool(Tool):
    """Tool to get information about the current end user in this session.Such as openid, channel, chatid"""

    def __init__(self) -> None:
        self._channel: str = ""
        self._chat_id: str = ""
        self._sender_id: str = ""

    def set_context(self, channel: str, chat_id: str, sender_id: str | None = None) -> None:
        """Set the current session context."""
        self._channel = channel
        self._chat_id = chat_id
        self._sender_id = sender_id or ""

    @property
    def name(self) -> str:
        return "user"

    @property
    def description(self) -> str:
        return (
            "Get information about the current user in this conversation. "
            "On Feishu, the user_id/open_id field is the Feishu user's open_id. "
            "Use this when you need the current user's identifier to call external services."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        # No parameters needed – always returns info for the current session user.
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        if not self._sender_id:
            return "Error: user context is not available for this session."

        data: dict[str, Any] = {
            "channel": self._channel,
            "chat_id": self._chat_id,
            "user_id": self._sender_id,
        }

        # For Feishu, sender_id is the user's open_id.
        if self._channel == "feishu":
            data["open_id"] = self._sender_id

        return json.dumps(data, ensure_ascii=False)

