"""Persistent conversation memory for the agent."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class ConversationMemory:
    """JSON-backed conversation store keyed by conversation id."""

    def __init__(self, storage_path: str) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conversations: dict[str, dict] = {}
        self._load()

    def create_conversation(self) -> str:
        conversation_id = uuid.uuid4().hex
        now = self._now()
        with self._lock:
            self._conversations[conversation_id] = {
                "conversation_id": conversation_id,
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            self._persist_unlocked()
        return conversation_id

    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                raise KeyError(f"Conversation '{conversation_id}' was not found.")
            conversation["messages"].append(
                {"role": role, "content": content, "created_at": self._now()}
            )
            conversation["updated_at"] = self._now()
            self._persist_unlocked()

    def get_messages(self, conversation_id: str) -> list[dict]:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Conversation '{conversation_id}' was not found.")
        return [dict(message) for message in conversation["messages"]]

    def get(self, conversation_id: str) -> dict:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Conversation '{conversation_id}' was not found.")
        return {
            "conversation_id": conversation["conversation_id"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": [dict(message) for message in conversation["messages"]],
        }

    def list_conversations(self) -> list[dict]:
        return sorted(
            (
                {
                    "conversation_id": item["conversation_id"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "message_count": len(item["messages"]),
                }
                for item in self._conversations.values()
            ),
            key=lambda item: item["updated_at"],
            reverse=True,
        )

    def __len__(self) -> int:
        return len(self._conversations)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for conversation in payload.get("conversations", []):
            conversation_id = conversation.get("conversation_id")
            if conversation_id:
                self._conversations[conversation_id] = conversation

    def _persist_unlocked(self) -> None:
        payload = {"conversations": list(self._conversations.values())}
        self.storage_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
