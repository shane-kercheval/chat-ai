"""SQLite-backed conversation manager with in-memory caching."""

from copy import deepcopy
from cachetools import LRUCache
import aiosqlite
import json
import asyncio
from uuid import uuid4
import logging
from google.protobuf.json_format import MessageToDict, ParseDict
from proto.generated import chat_pb2
from google.protobuf import timestamp_pb2


class ConversationNotFoundError(Exception):
    """Raised when a conversation ID doesn't exist."""


class _Conversation:
    """Internal model for conversation state."""

    def __init__(
            self,
            id: str,  # noqa: A002
            messages: list[chat_pb2.ConversationEntry] | None = None,
            ):
        self.id = id
        self.messages = messages or []


class ConversationManager:
    """Manages conversations with SQLite backing."""

    def __init__(
            self,
            db_path: str, initial_conversations: list[chat_pb2.Conversation] | None = None,
        ):
        self.db_path = db_path
        self._logger = logging.getLogger(__name__)
        self._initial_conversations = initial_conversations or []
        # used to synchronize access to the database
        self._lock = asyncio.Lock()
        self._conversations: LRUCache = LRUCache(maxsize=50)
        self._conversation_locks: LRUCache = LRUCache(maxsize=50)

    async def initialize(self) -> None:
        """Initialize the database."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL
                )
            """)
            # Load initial conversations if provided
            if self._initial_conversations:
                for conv in self._initial_conversations:
                    messages_json = json.dumps([
                        MessageToDict(entry, preserving_proto_field_name=True)
                        for entry in conv.entries
                    ])
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO conversations (id, messages)
                        VALUES (?, ?)
                        """,
                        (
                            conv.conversation_id,
                            messages_json,
                        ),
                    )
            await db.commit()

    async def get_conversation_lock(self, conv_id: str) -> asyncio.Lock:
        """Get a lock for a conversation ID."""
        if not conv_id:
            raise ValueError("Conversation ID must be provided")
        async with self._lock:
            if conv_id not in self._conversation_locks:
                self._conversation_locks[conv_id] = asyncio.Lock()
        return self._conversation_locks[conv_id]

    async def create_conversation(self) -> str:
        """Create a new conversation ID."""
        conv_id = str(uuid4())
        conv = _Conversation(id=conv_id)
        lock = await self.get_conversation_lock(conv_id)
        async with lock:
            await self._save_conversation(conv)
            return conv_id

    async def get_messages(self, conv_id: str) -> list[chat_pb2.ConversationEntry]:
        """Get messages from an existing conversation."""
        lock = await self.get_conversation_lock(conv_id)
        async with lock:
            conv = await self._get_conversation(conv_id)
        return conv.messages

    async def add_message(self, conv_id: str, message: chat_pb2.ConversationEntry) -> None:
        """Add a message to an existing conversation."""
        lock = await self.get_conversation_lock(conv_id)
        async with lock:
            conv = await self._get_conversation(conv_id)
            conv.messages.append(deepcopy(message))
            await self._save_conversation(conv)

    async def _save_conversation(self, conversation: _Conversation) -> None:
        """
        Save conversation to SQLite and update in-memory cache.

        This function should be called with the lock held.
        """
        messages_json = json.dumps([
            MessageToDict(msg, preserving_proto_field_name=True)
            for msg in conversation.messages
        ])
        async with aiosqlite.connect(self.db_path) as db:
            self._conversations[conversation.id] = conversation
            await db.execute(
                """
                INSERT OR REPLACE INTO conversations (id, messages)
                VALUES (?, ?)
                """,
                (
                    conversation.id,
                    messages_json,
                ),
            )
            await db.commit()

    async def _get_conversation(self, conv_id: str) -> _Conversation:
        """
        Get an existing conversation or raise if not found.

        This function should be called with the lock held.
        """
        # check in-memory cache
        if conv_id in self._conversations:
            return deepcopy(self._conversations[conv_id])
        # load from SQLite
        self._logger.info(f"Loading conversation {conv_id}")
        async with aiosqlite.connect(self.db_path) as db:  # noqa: SIM117
            async with db.execute(
                "SELECT messages FROM conversations WHERE id = ?",
                (conv_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            raise ConversationNotFoundError(f"Conversation {conv_id} not found")
        messages_json = row[0]
        messages_data = json.loads(messages_json)
        messages = [
            ParseDict(msg_dict, chat_pb2.ConversationEntry())
            for msg_dict in messages_data
        ]
        conv = _Conversation(id=conv_id, messages=messages)
        self._conversations[conv_id] = conv
        return deepcopy(conv)

    async def get_all_conversations(self) -> list[chat_pb2.Conversation]:
        """Get all conversations from the database."""
        async with self._lock, aiosqlite.connect(self.db_path) as db, db.execute(
            "SELECT id, messages FROM conversations",
        ) as cursor:
            rows = await cursor.fetchall()

        conversations = []
        for row in rows:
            conv_id, messages_json = row
            messages_data = json.loads(messages_json)
            messages = [
                ParseDict(msg_dict, chat_pb2.ConversationEntry())
                for msg_dict in messages_data
            ]
            conversations.append(chat_pb2.Conversation(
                conversation_id=conv_id,
                entries=messages,
            ))
        return conversations

    async def delete_conversation(self, conv_id: str) -> None:
        """Delete a conversation from both memory and database."""
        lock = await self.get_conversation_lock(conv_id)
        async with lock, aiosqlite.connect(self.db_path) as db:
            # remove from memory cache if it's there
            self._conversations.pop(conv_id, None)
            # check if conversation exists in database
            cursor = await db.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conv_id,),
            )
            exists = await cursor.fetchone()
            if not exists:
                raise ConversationNotFoundError(f"Conversation {conv_id} not found")
            # Delete from database
            await db.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conv_id,),
            )
            await db.commit()
        async with self._lock:
            # remove conversation lock
            self._conversation_locks.pop(conv_id, None)

    async def truncate_conversation(self, conv_id: str, entry_id: str) -> None:
        """
        Delete a specific entry and all remaining entries in the conversation.

        Args:
            conv_id: The ID of the conversation to truncate
            entry_id: The ID of the entry at which to truncate (inclusive)

        Raises:
            ConversationNotFoundError: If the conversation doesn't exist
            ValueError: If the entry_id doesn't exist in the conversation
        """
        lock = await self.get_conversation_lock(conv_id)
        async with lock:
            conv = await self._get_conversation(conv_id)
            # Find the index of the entry to truncate at
            try:
                truncate_idx = next(
                    i
                    for i, msg in enumerate(conv.messages)
                    if msg.entry_id == entry_id
                )
            except StopIteration:
                raise ValueError(f"Entry `{entry_id}` not found in conversation {conv_id}")
            # Truncate the conversation at this index
            conv.messages = conv.messages[:truncate_idx]
            # Update the conversation in the database
            await self._save_conversation(conv)

    async def branch_conversation(
            self,
            conv_id: str,
            entry_id: str,
            model_index: int | None = None,
        ) -> str:
        """
        Create a new conversation branched from an existing one at a specific entry.

        Args:
            conv_id: ID of the conversation to branch from
            entry_id: ID of the entry to branch at (inclusive)
            model_index: Optional model index for MultiChatModelResponse entries

        Returns:
            ID of the new conversation

        Raises:
            ConversationNotFoundError: If the conversation doesn't exist
            ValueError: If the entry_id doesn't exist or model_index is invalid
        """
        # Get conversation lock for source conversation
        lock = await self.get_conversation_lock(conv_id)
        source_conv = None
        branch_idx = None

        async with lock:
            source_conv = await self._get_conversation(conv_id)
        # Find the index of the entry to branch at
        try:
            branch_idx = next(
                i
                for i, msg in enumerate(source_conv.messages)
                if msg.entry_id == entry_id
            )
        except StopIteration:
            raise ValueError(f"Entry `{entry_id}` not found in conversation {conv_id}")

        # Create new messages array for branched conversation
        new_messages = []
        for i in range(branch_idx + 1):
            msg = source_conv.messages[i]
            if i == branch_idx:
                # Check if this is a multi-model response
                # set the timestamp to the current time which is technically more accurate
                # since this is the latest activity, but will also allow the conversation to be
                # sorted correctly
                timestamp = timestamp_pb2.Timestamp()
                timestamp.GetCurrentTime()
                if msg.HasField("multi_model_response"):
                    if model_index is None:
                        raise ValueError(f"Entry {entry_id} is a multi-model response but no model_index provided")  # noqa: E501
                    # Find the specified model response
                    try:
                        selected_response = next(
                            r for r in msg.multi_model_response.responses
                            if r.model_index == model_index
                        )
                    except StopIteration:
                        raise ValueError(f"Model index {model_index} not found in multi-model response")  # noqa: E501
                    # Create new message with single model response
                    new_msg = chat_pb2.ConversationEntry(
                        entry_id=str(uuid4()),
                        single_model_response=chat_pb2.ChatModelResponse(
                            message=selected_response.message,
                            config_snapshot=selected_response.config_snapshot,
                            model_index=0,  # Reset to 0 since it's now a single response
                        ),
                        timestamp=timestamp,
                    )
                    new_messages.append(new_msg)
                else:
                    msg.timestamp.CopyFrom(timestamp)
                    new_messages.append(msg)
            else:
                new_messages.append(msg)
        new_conv_id = str(uuid4())
        new_conv = _Conversation(id=new_conv_id, messages=new_messages)
        await self._save_conversation(new_conv)
        return new_conv_id

    async def set_multi_response_selected_model(
            self,
            conv_id: str,
            entry_id: str,
            selected_model_index: int,
        ) -> None:
        """
        Set the selected model index for a multi-model response entry.

        Args:
            conv_id: The conversation ID
            entry_id: The entry ID containing the multi-model response
            selected_model_index: The model index to set as selected

        Raises:
            ConversationNotFoundError: If the conversation doesn't exist
            ValueError: If the entry doesn't exist or isn't a multi-model response,
                    or if the model index is invalid
        """
        lock = await self.get_conversation_lock(conv_id)
        async with lock:
            conv = await self._get_conversation(conv_id)
            # Find the entry
            try:
                entry = next(
                    entry for entry in conv.messages
                    if entry.entry_id == entry_id
                )
            except StopIteration:
                raise ValueError(f"Entry `{entry_id}` not found in conversation {conv_id}")

            # Verify it's a multi-model response
            if not entry.HasField("multi_model_response"):
                raise ValueError(f"Entry `{entry_id}` is not a multi-model response")

            # Verify model index is valid
            if not any(
                response.model_index == selected_model_index
                for response in entry.multi_model_response.responses
            ):
                raise ValueError(
                    f"Model index {selected_model_index} not found in multi-model response",
                )
            # Update the selected index
            entry.multi_model_response.selected_model_index.value = selected_model_index
            await self._save_conversation(conv)


def convert_proto_messages_to_model_messages(  # noqa: PLR0912
        messages: list[chat_pb2.ConversationEntry],
        model_config: chat_pb2.ModelConfig | None = None,
    ) -> list[dict]:
    """
    Convert list of protobuf ConversationEntry to format expected by model APIs.

    For multi-model responses, the `selected_model_index` will be used if it exists in the
    MultiChatModelResponse. Otherwise, a model_config can be provided to match the response's
    config_snapshot. If neither is provided, the first response will be used.

    Args:
        messages: List of ConversationEntry protos
        model_config: Optional ModelConfig to match responses against for multi-model responses

    Returns:
        List of dicts in format expected by model APIs:
        [{'role': 'user'|'assistant'|'system', 'content': str}, ...]
    """
    model_messages = []
    for conv_msg in messages:
        msg_type = conv_msg.WhichOneof("message")
        if msg_type == "chat_message":
            model_messages.append({
                'role': {
                    chat_pb2.Role.SYSTEM: 'system',
                    chat_pb2.Role.USER: 'user',
                    chat_pb2.Role.ASSISTANT: 'assistant',
                }[conv_msg.chat_message.role],
                'content': conv_msg.chat_message.content,
            })
        elif msg_type == "single_model_response":
            model_messages.append({
                'role': 'assistant',
                'content': conv_msg.single_model_response.message.content,
            })
        elif msg_type == "multi_model_response":
            # First check for selected_model_index
            selected_response = None
            if conv_msg.multi_model_response.HasField("selected_model_index"):
                for response in conv_msg.multi_model_response.responses:
                    if response.model_index == conv_msg.multi_model_response.selected_model_index.value:  # noqa: E501
                        selected_response = response
                        break
            # If no selected index or response not found, fall back to matching config
            if not selected_response and model_config:
                for response in conv_msg.multi_model_response.responses:
                    if response.config_snapshot == model_config:
                        selected_response = response
                        break
            # If still no match, use model_index = 0
            if not selected_response:
                for response in conv_msg.multi_model_response.responses:
                    if response.model_index == 0:
                        selected_response = response
                        break
            if not selected_response:
                raise ValueError("No matching response found in multi-model response")
            model_messages.append({
                'role': 'assistant',
                'content': selected_response.message.content,
            })
    return model_messages

def convert_to_conversations(conversations: list[dict]) -> list[chat_pb2.Conversation]:
    """Convert list of conversation dicts to protobuf Conversation messages."""
    return [ParseDict(conv_data, chat_pb2.Conversation()) for conv_data in conversations]
