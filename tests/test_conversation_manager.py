"""Tests for ConversationManager."""
from datetime import datetime, timezone
import os
import tempfile
from pathlib import Path
import aiofiles
import pytest
import aiosqlite
import json
import asyncio
from uuid import UUID, uuid4
import yaml

from google.protobuf.json_format import ParseDict
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from proto.generated import chat_pb2
from server.conversation_manager import (
    ConversationManager,
    ConversationNotFoundError,
    _Conversation,
    convert_proto_messages_to_model_messages,
    convert_to_conversations,
)
from typing import Optional

def create_temp_db_path():
    """Create a temporary database file."""
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)  # noqa: SIM115
    f.close()
    return Path(f.name)


def create_fake_config_snapshot(client_type: str, model_name: str) -> chat_pb2.ModelConfig:
    return chat_pb2.ModelConfig(
        client_type=client_type,
        model_name=model_name,
        model_parameters=chat_pb2.ModelParameters(
                temperature=0.5,
                server_url='http://localhost',
            ),
    )

@pytest.mark.asyncio
class TestConversationManager:
    """Tests for the ConversationManager."""

    async def test_conversation_initialization(self):
        """Test that a conversation is properly initialized."""
        conv_id = "test_conversation_id"
        conv = _Conversation(conv_id)
        assert conv.id == conv_id
        assert len(conv.messages) == 0

    async def test__initialization(self):
        """Test database initialization."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            assert os.path.exists(str(db_path))
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='conversations'
                """)
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == "conversations"
        finally:
            db_path.unlink(missing_ok=True)

    async def test__create_conversation(self):
        """Test creating a new conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()
            # Verify UUID format
            assert UUID(conv_id)

            # Verify in database
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT messages FROM conversations WHERE id = ?",
                    (conv_id,),
                )
                result = await cursor.fetchone()
                assert result is not None
                messages = json.loads(result[0])
                assert messages == []
        finally:
            db_path.unlink(missing_ok=True)

    async def test__nonexistent_conversation(self):
        """Test handling of nonexistent conversations."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()

            with pytest.raises(ConversationNotFoundError):
                await manager.get_messages("nonexistent")

            fake_message = chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Hello'),
                timestamp=datetime.now(timezone.utc),
            )
            with pytest.raises(ConversationNotFoundError):
                await manager.add_message("nonexistent", fake_message)
        finally:
            db_path.unlink(missing_ok=True)

    async def test__conversation_persistence(self):
        """Test conversation persistence across manager instances."""
        db_path = create_temp_db_path()
        try:
            manager1 = ConversationManager(str(db_path))
            await manager1.initialize()

            conv_id = await manager1.create_conversation()
            assert await manager1.get_messages(conv_id) == []
            fake_message = chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Test1'),
                timestamp=datetime.now(timezone.utc),
            )
            await manager1.add_message(conv_id, fake_message)

            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            messages = await manager2.get_messages(conv_id)
            assert len(messages) == 1
            assert messages[0].chat_message.role == chat_pb2.Role.USER
            assert messages[0].chat_message.content == 'Test1'
            assert messages[0].timestamp is not None
        finally:
            db_path.unlink(missing_ok=True)

    async def test__message_ordering(self):
        """Test message order preservation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()

            conv_id = await manager.create_conversation()
            messages = [
                chat_pb2.ConversationEntry(
                    chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Message 1'),  # noqa: E501
                    timestamp=datetime.now(timezone.utc),
                ),
                chat_pb2.ConversationEntry(
                    single_model_response=chat_pb2.ChatModelResponse(
                        message=chat_pb2.ChatMessage(role=chat_pb2.Role.ASSISTANT, content='Response 1'),  # noqa: E501
                        config_snapshot=create_fake_config_snapshot("api-test", "test-model"),
                        model_index=1,
                    ),
                    timestamp=datetime.now(timezone.utc),
                ),
                chat_pb2.ConversationEntry(
                    chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Message 2'),  # noqa: E501
                    timestamp=datetime.now(timezone.utc),
                ),
                chat_pb2.ConversationEntry(
                    multi_model_response=chat_pb2.MultiChatModelResponse(responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(role=chat_pb2.Role.ASSISTANT, content='Response 2-A'),  # noqa: E501
                            config_snapshot=create_fake_config_snapshot("api-test", "test-model"),
                            model_index=0,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(role=chat_pb2.Role.ASSISTANT, content='Response 2-B'),  # noqa: E501
                            config_snapshot=create_fake_config_snapshot("api-test2", "test-model2"),  # noqa: E501
                            model_index=1,
                        ),
                    ]),
                    timestamp=datetime.now(timezone.utc),
                ),
            ]
            for message in messages:
                await manager.add_message(conv_id=conv_id, message=message)
            stored_messages = await manager.get_messages(conv_id)
            assert len(stored_messages) == len(messages)

            assert stored_messages[0].chat_message.role == chat_pb2.Role.USER
            assert stored_messages[0].chat_message.content == 'Message 1'
            assert stored_messages[0].timestamp is not None
            assert stored_messages[1].single_model_response.message.role == chat_pb2.Role.ASSISTANT
            assert stored_messages[1].single_model_response.message.content == 'Response 1'
            assert stored_messages[1].single_model_response.config_snapshot.client_type == 'api-test'  # noqa: E501
            assert stored_messages[1].single_model_response.config_snapshot.model_name == 'test-model'  # noqa: E501
            assert stored_messages[1].single_model_response.config_snapshot.model_parameters.temperature == pytest.approx(0.5)  # noqa: E501
            assert stored_messages[1].single_model_response.config_snapshot.model_parameters.server_url == 'http://localhost'  # noqa: E501
            assert stored_messages[1].single_model_response.model_index == 1
            assert stored_messages[1].timestamp is not None
            assert stored_messages[2].chat_message.role == chat_pb2.Role.USER
            assert stored_messages[2].chat_message.content == 'Message 2'
            assert stored_messages[2].timestamp is not None
            assert stored_messages[3].multi_model_response.responses[0].model_index == 0
            assert stored_messages[3].multi_model_response.responses[0].message.role == chat_pb2.Role.ASSISTANT  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[0].message.content == 'Response 2-A'  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[0].config_snapshot.client_type == 'api-test'  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[0].config_snapshot.model_name == 'test-model'  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[0].config_snapshot.model_parameters.temperature == pytest.approx(0.5)  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[0].config_snapshot.model_parameters.server_url == 'http://localhost'  # noqa: E501

            assert stored_messages[3].multi_model_response.responses[1].model_index == 1
            assert stored_messages[3].multi_model_response.responses[1].message.role == chat_pb2.Role.ASSISTANT  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[1].message.content == 'Response 2-B'  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[1].config_snapshot.client_type == 'api-test2'  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[1].config_snapshot.model_name == 'test-model2'  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[1].config_snapshot.model_parameters.temperature == pytest.approx(0.5)  # noqa: E501
            assert stored_messages[3].multi_model_response.responses[1].config_snapshot.model_parameters.server_url == 'http://localhost'  # noqa: E501
            assert stored_messages[3].timestamp is not None

            manager_2 = ConversationManager(str(db_path))
            await manager_2.initialize()
            stored_messages_2 = await manager_2.get_messages(conv_id)
            assert stored_messages == stored_messages_2
        finally:
            db_path.unlink(missing_ok=True)

    async def test__concurrent_messages(self):
        """Test concurrent message additions."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()

            def create_message(_id: int) -> chat_pb2.ConversationEntry:
                return chat_pb2.ConversationEntry(
                    chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content=f"Message {_id}"),  # noqa: E501
                    timestamp=datetime.now(timezone.utc),
                )

            async def add_messages(ids: list) -> None:
                for _id in ids:
                    await manager.add_message(conv_id, create_message(_id))

            # Run concurrent additions
            await asyncio.gather(
                add_messages(range(50)),
                add_messages(range(50, 100)),
            )

            messages = await manager.get_messages(conv_id)
            assert len(messages) == 100

            # Verify all messages are present and unique
            contents = [msg.chat_message.content for msg in messages]
            assert all('Message' in content for content in contents)
            assert len(set(contents)) == 100
        finally:
            db_path.unlink(missing_ok=True)

    async def test__get_conversation__immutability(self):
        """Test that returned conversations cannot modify internal state."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()

            conv_id = await manager.create_conversation()
            fake_message = chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Hello'),
                timestamp=datetime.now(timezone.utc),
            )
            await manager.add_message(conv_id, fake_message)

            # Get conversation and try to modify it
            # add_message will put it in the cache
            conversation = await manager._get_conversation(conv_id)
            conversation.messages[0].chat_message.content = 'Modified'
            conversation.messages[0].chat_message.role = chat_pb2.Role.ASSISTANT

            # ensure changes are not reflected in the manager
            stored_conversation = await manager._get_conversation(conv_id)
            assert len(stored_conversation.messages) == 1
            assert stored_conversation.messages[0].chat_message.content == 'Hello'
            assert stored_conversation.messages[0].chat_message.role == chat_pb2.Role.USER

            # test on a different manager instance so that it loads from DB, not cache
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            stored_conversation2 = await manager2._get_conversation(conv_id)
            assert stored_conversation2.messages == stored_conversation.messages

            stored_conversation2.messages[0].chat_message.content = 'Modified'
            stored_conversation2.messages[0].chat_message.role = chat_pb2.Role.ASSISTANT

            stored_conversation3 = await manager2._get_conversation(conv_id)
            assert stored_conversation3.messages == stored_conversation.messages
        finally:
            db_path.unlink(missing_ok=True)

    async def test__message_immutability(self):
        """Test that returned messages cannot modify internal state."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()

            conv_id = await manager.create_conversation()
            fake_message = chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Original'),
                timestamp=datetime.now(timezone.utc),
            )
            await manager.add_message(conv_id, fake_message)

            # Get messages and try to modify them
            messages = await manager.get_messages(conv_id)
            messages[0].chat_message.content = 'Modified'
            messages[0].chat_message.role = chat_pb2.Role.ASSISTANT

            # ensure changes are not reflected in the manager
            stored_messages = await manager.get_messages(conv_id)
            assert len(stored_messages) == 1
            assert stored_messages[0].chat_message.content == 'Original'
            assert stored_messages[0].chat_message.role == chat_pb2.Role.USER
        finally:
            db_path.unlink(missing_ok=True)

    async def test__add_message(self):
        """Test adding a message to a conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()
            fake_message = chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Hello'),
                timestamp=datetime.now(timezone.utc),
            )
            await manager.add_message(conv_id, fake_message)
            # test in-memory storage
            stored_messages = await manager.get_messages(conv_id)
            assert len(stored_messages) == 1
            assert stored_messages[0].chat_message.role == chat_pb2.Role.USER
            assert stored_messages[0].chat_message.content == 'Hello'
            assert stored_messages[0].timestamp is not None

            # test new manager instance (which should load from DB)
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            stored_messages2 = await manager2.get_messages(conv_id)
            assert stored_messages == stored_messages2
        finally:
            db_path.unlink(missing_ok=True)

    async def test__add_message__conv_id_does_not_exist(self):
        """Test adding a message to a nonexistent conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            fake_message = chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(role=chat_pb2.Role.USER, content='Hello'),
                timestamp=datetime.now(timezone.utc),
            )
            with pytest.raises(ConversationNotFoundError):
                await manager.add_message("nonexistent", fake_message)
        finally:
            db_path.unlink(missing_ok=True)

    async def test__get_all_conversations__empty(self):
        """Test getting conversations when none exist."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(db_path)
            await manager.initialize()
            conversations = await manager.get_all_conversations()
            assert len(conversations) == 0
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__get_all_conversations__with_initial_data(self):
        """Test getting conversations with initial example data."""
        db_path = create_temp_db_path()
        try:
            # Load example conversations from YAML
            async with aiofiles.open('artifacts/example_history.yaml') as f:
                content = await f.read()
                yaml_data = yaml.safe_load(content)
            # Convert YAML data to proto conversations
            initial_conversations = []
            for conv_data in yaml_data['conversations']:
                entries = []
                for entry_data in conv_data['entries']:
                    # Parse timestamp
                    timestamp = timestamp_pb2.Timestamp()
                    timestamp.FromJsonString(entry_data['timestamp'])
                    # Create entry proto
                    entry = ParseDict(entry_data, chat_pb2.ConversationEntry())
                    entry.timestamp.CopyFrom(timestamp)
                    entries.append(entry)
                conv = chat_pb2.Conversation(
                    conversation_id=conv_data['conversation_id'],
                    entries=entries,
                )
                initial_conversations.append(conv)
            # Initialize manager with example data
            manager = ConversationManager(db_path, initial_conversations=initial_conversations)
            await manager.initialize()
            # Get all conversations
            conversations = await manager.get_all_conversations()
            # Verify results
            assert len(conversations) == 2
            # Check first conversation (single model responses)
            conv1 = next(c for c in conversations if c.conversation_id == 'conv-1')
            assert len(conv1.entries) == 4
            for i, entry in enumerate(conv1.entries):
                assert entry.entry_id == f'entry1-{i+1}'

            assert conv1.entries[0].HasField('chat_message')
            assert not conv1.entries[0].HasField('single_model_response')
            assert not conv1.entries[0].HasField('multi_model_response')
            assert conv1.entries[0].chat_message.role == chat_pb2.Role.USER
            assert conv1.entries[0].chat_message.content == 'What is the capital of France?'
            assert conv1.entries[0].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 0, 0)

            assert conv1.entries[1].HasField('single_model_response')
            assert not conv1.entries[1].HasField('multi_model_response')
            assert not conv1.entries[1].HasField('chat_message')
            assert conv1.entries[1].single_model_response.message.role == chat_pb2.Role.ASSISTANT
            assert conv1.entries[1].single_model_response.message.content == 'The capital of France is Paris.'  # noqa: E501
            assert conv1.entries[1].single_model_response.config_snapshot.client_type == 'OpenAI'
            assert conv1.entries[1].single_model_response.config_snapshot.model_name == 'gpt-4o-mini'  # noqa: E501
            assert conv1.entries[1].single_model_response.config_snapshot.model_parameters.HasField('temperature')  # noqa: E501
            assert conv1.entries[1].single_model_response.config_snapshot.model_parameters.temperature == pytest.approx(0.2)  # noqa: E501
            assert not conv1.entries[1].single_model_response.config_snapshot.model_parameters.HasField('max_tokens')  # noqa: E501
            assert not conv1.entries[1].single_model_response.config_snapshot.model_parameters.HasField('top_p')  # noqa: E501
            assert not conv1.entries[1].single_model_response.config_snapshot.model_parameters.HasField('server_url')  # noqa: E501
            assert conv1.entries[1].single_model_response.model_index == 0
            assert conv1.entries[1].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 0, 1)

            assert conv1.entries[2].HasField('chat_message')
            assert not conv1.entries[2].HasField('single_model_response')
            assert not conv1.entries[2].HasField('multi_model_response')
            assert conv1.entries[2].chat_message.role == chat_pb2.Role.USER
            assert conv1.entries[2].chat_message.content == 'What is its population?'
            assert conv1.entries[2].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 0, 2)

            assert conv1.entries[3].HasField('single_model_response')
            assert not conv1.entries[3].HasField('multi_model_response')
            assert not conv1.entries[3].HasField('chat_message')
            assert conv1.entries[3].single_model_response.message.role == chat_pb2.Role.ASSISTANT
            assert conv1.entries[3].single_model_response.message.content == 'Paris has a population of about 2.2 million people in the city proper.'  # noqa: E501
            assert conv1.entries[3].single_model_response.config_snapshot.client_type == 'OpenAI'
            assert conv1.entries[3].single_model_response.config_snapshot.model_name == 'gpt-4o-mini'  # noqa: E501
            assert conv1.entries[3].single_model_response.config_snapshot.model_parameters.HasField('temperature')  # noqa: E501
            assert conv1.entries[3].single_model_response.config_snapshot.model_parameters.temperature == pytest.approx(0.2)  # noqa: E501
            assert conv1.entries[3].single_model_response.model_index == 0
            assert conv1.entries[3].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 0, 3)
            assert not conv1.entries[3].single_model_response.config_snapshot.model_parameters.HasField('max_tokens')  # noqa: E501
            assert not conv1.entries[3].single_model_response.config_snapshot.model_parameters.HasField('top_p')  # noqa: E501
            assert not conv1.entries[3].single_model_response.config_snapshot.model_parameters.HasField('server_url')  # noqa: E501

            # Check second conversation (mixed responses)
            conv2 = next(c for c in conversations if c.conversation_id == 'conv-2')
            assert len(conv2.entries) == 4

            for i, entry in enumerate(conv2.entries):
                assert entry.entry_id == f'entry2-{i+1}'

            assert conv1.entries[0].HasField('chat_message')
            assert not conv1.entries[0].HasField('single_model_response')
            assert not conv1.entries[0].HasField('multi_model_response')
            assert conv2.entries[0].chat_message.role == chat_pb2.Role.USER
            assert conv2.entries[0].chat_message.content == 'Compare Python and JavaScript'
            assert conv2.entries[0].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 30, 0)

            assert conv1.entries[1].HasField('single_model_response')
            assert not conv1.entries[1].HasField('multi_model_response')
            assert not conv1.entries[1].HasField('chat_message')
            assert conv2.entries[1].single_model_response.message.role == chat_pb2.Role.ASSISTANT
            assert conv2.entries[1].single_model_response.message.content == 'Python and JavaScript are both popular programming languages, but they serve different primary purposes. Python is known for its simplicity and readability, often used in backend development, data science, and AI. JavaScript was originally designed for web browsers but has evolved to be used in many environments.'  # noqa: E501
            assert conv2.entries[1].single_model_response.config_snapshot.client_type == 'OpenAI'
            assert conv2.entries[1].single_model_response.config_snapshot.model_name == 'gpt-4o-mini'  # noqa: E501
            assert conv2.entries[1].single_model_response.config_snapshot.model_parameters.HasField('temperature')  # noqa: E501
            assert conv2.entries[1].single_model_response.config_snapshot.model_parameters.temperature == pytest.approx(0.5)  # noqa: E501
            assert conv2.entries[1].single_model_response.config_snapshot.model_parameters.HasField('max_tokens')  # noqa: E501
            assert conv2.entries[1].single_model_response.config_snapshot.model_parameters.max_tokens == 256  # noqa: E501
            assert not conv2.entries[1].single_model_response.config_snapshot.model_parameters.HasField('top_p')  # noqa: E501
            assert not conv2.entries[1].single_model_response.config_snapshot.model_parameters.HasField('server_url')  # noqa: E501
            assert conv2.entries[1].single_model_response.model_index == 0
            assert conv2.entries[1].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 30, 1)

            assert conv2.entries[2].HasField('chat_message')
            assert not conv2.entries[2].HasField('single_model_response')
            assert not conv2.entries[2].HasField('multi_model_response')
            assert conv2.entries[2].chat_message.role == chat_pb2.Role.USER
            assert conv2.entries[2].chat_message.content == 'Which is better for a beginner?'
            assert conv2.entries[2].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 30, 2)

            assert conv2.entries[3].HasField('multi_model_response')
            assert not conv2.entries[3].HasField('single_model_response')
            assert not conv2.entries[3].HasField('chat_message')
            assert conv2.entries[3].multi_model_response.responses[0].message.role == chat_pb2.Role.ASSISTANT  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[0].message.content == 'For beginners, Python is often considered the better choice due to its clean syntax and gentle learning curve.'  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[0].config_snapshot.client_type == 'OpenAI'  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[0].config_snapshot.model_name == 'gpt-4o-mini'  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[0].config_snapshot.model_parameters.HasField('temperature')  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[0].config_snapshot.model_parameters.temperature == pytest.approx(0.8)  # noqa: E501
            assert not conv2.entries[3].multi_model_response.responses[0].config_snapshot.model_parameters.HasField('max_tokens')  # noqa: E501
            assert not conv2.entries[3].multi_model_response.responses[0].config_snapshot.model_parameters.HasField('top_p')  # noqa: E501
            assert not conv2.entries[3].multi_model_response.responses[0].config_snapshot.model_parameters.HasField('server_url')  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[0].model_index == 0
            assert conv2.entries[3].multi_model_response.responses[1].message.role == chat_pb2.Role.ASSISTANT  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[1].message.content == "I recommend starting with Python because of its readability and extensive learning resources, though JavaScript is also a viable option if you're interested in web development."  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[1].config_snapshot.client_type == 'Anthropic'  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[1].config_snapshot.model_name == 'claude-3-haiku-latest'  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[1].config_snapshot.model_parameters.HasField('temperature')  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[1].config_snapshot.model_parameters.temperature == pytest.approx(0.2)  # noqa: E501
            assert not conv2.entries[3].multi_model_response.responses[1].config_snapshot.model_parameters.HasField('max_tokens')  # noqa: E501
            assert not conv2.entries[3].multi_model_response.responses[1].config_snapshot.model_parameters.HasField('top_p')  # noqa: E501
            assert not conv2.entries[3].multi_model_response.responses[1].config_snapshot.model_parameters.HasField('server_url')  # noqa: E501
            assert conv2.entries[3].multi_model_response.responses[1].model_index == 1
            assert conv2.entries[3].timestamp.ToDatetime() == datetime(2024, 12, 23, 14, 30, 3)
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__get_all_conversations__persistence(self):
        """Test that conversations persist across manager instances."""
        db_path = create_temp_db_path()
        try:
            # Create first manager and add some conversations
            manager1 = ConversationManager(db_path)
            await manager1.initialize()

            conversations = await manager1.get_all_conversations()
            assert len(conversations) == 0
            conv1_id = await manager1.create_conversation()
            conv2_id = await manager1.create_conversation()
            # Add messages to first conversation
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            await manager1.add_message(
                conv1_id,
                chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content="Test message 1",
                    ),
                    timestamp=timestamp,
                ),
            )
            # Add messages to second conversation
            timestamp.GetCurrentTime()
            await manager1.add_message(
                conv2_id,
                chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content="Test message 2",
                    ),
                    timestamp=timestamp,
                ),
            )
            # Create new manager instance and verify conversations
            manager2 = ConversationManager(db_path)
            await manager2.initialize()

            conversations = await manager2.get_all_conversations()
            assert len(conversations) == 2

            conv_ids = {c.conversation_id for c in conversations}
            assert conv_ids == {conv1_id, conv2_id}

            for conv in conversations:
                assert len(conv.entries) == 1
                if conv.conversation_id == conv1_id:
                    assert conv.entries[0].chat_message.content == "Test message 1"
                else:
                    assert conv.entries[0].chat_message.content == "Test message 2"
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__get_all_conversations__concurrent_access(self):
        """Test concurrent access to conversations."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(db_path)
            await manager.initialize()

            # Create multiple conversations concurrently
            async def create_and_add_message(index: int):  # noqa: ANN202
                conv_id = await manager.create_conversation()
                timestamp=timestamp_pb2.Timestamp()
                timestamp.GetCurrentTime()
                await manager.add_message(
                    conv_id,
                    chat_pb2.ConversationEntry(
                        entry_id=str(uuid4()),
                        chat_message=chat_pb2.ChatMessage(
                            role=chat_pb2.Role.USER,
                            content=f"Message {index}",
                        ),
                        timestamp=timestamp,
                    ),
                )
                return conv_id

            # Create 10 conversations concurrently
            conv_ids = await asyncio.gather(*[
                create_and_add_message(i) for i in range(10)
            ])

            # Verify all conversations are retrievable
            conversations = await manager.get_all_conversations()
            assert len(conversations) == 10

            # Verify all conversation IDs match
            retrieved_ids = {c.conversation_id for c in conversations}
            assert retrieved_ids == set(conv_ids)

            # Verify message contents
            message_contents = {
                c.conversation_id: c.entries[0].chat_message.content
                for c in conversations
            }
            assert all(f"Message {i}" in message_contents.values() for i in range(10))
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__get_all_conversations__large_messages(self):
        """Test handling of conversations with large messages."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(db_path)
            await manager.initialize()
            # Create a conversation with a large message
            conv_id = await manager.create_conversation()
            large_content = "A" * 1000000  # 1MB of data
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            await manager.add_message(
                conv_id,
                chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content=large_content,
                    ),
                    timestamp=timestamp,
                ),
            )
            # Verify large message is retrieved correctly
            conversations = await manager.get_all_conversations()
            assert len(conversations) == 1
            assert len(conversations[0].entries) == 1
            assert len(conversations[0].entries[0].chat_message.content) == 1000000
            assert conversations[0].entries[0].chat_message.content == large_content
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__add_message__get_messages__deserialize_optional_variable_from_db(self):
        """Test adding a message and then getting it from a new manager instance."""
        db_path = create_temp_db_path()
        try:
            manager1 = ConversationManager(db_path)
            await manager1.initialize()
            conv_id = await manager1.create_conversation()
            message = chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(role=chat_pb2.Role.ASSISTANT, content='Test1'),  # noqa: E501
                            config_snapshot=create_fake_config_snapshot("api-test", "test-model"),
                            model_index=0,
                        ),
                    ],
                    # selected selected_model_index not set and should be deserialized with such
                    # that HasField returns False
                ),
            )
            assert message.multi_model_response.HasField('selected_model_index') is False
            await manager1.add_message(conv_id, message)

            manager2 = ConversationManager(db_path)
            await manager2.initialize()
            stored_messages = await manager2.get_messages(conv_id)
            assert len(stored_messages) == 1
            assert stored_messages[0].multi_model_response.HasField('selected_model_index') is False  # noqa: E501

            message.multi_model_response.selected_model_index.value = 1
            await manager1.add_message(conv_id, message)

            manager3 = ConversationManager(db_path)
            await manager3.initialize()
            stored_messages = await manager3.get_messages(conv_id)
            assert len(stored_messages) == 2
            assert stored_messages[0].multi_model_response.HasField('selected_model_index') is False  # noqa: E501
            assert stored_messages[1].multi_model_response.HasField('selected_model_index') is True
            assert stored_messages[1].multi_model_response.selected_model_index.value == 1
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__delete_conversation__in_memory(self):
        """Test deleting a conversation that is loaded in memory."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create conversation and add a message to ensure it's in memory
            conv_id = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            message = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, message)
            # Verify it's in memory
            assert conv_id in manager._conversations
            # Delete conversation
            await manager.delete_conversation(conv_id)
            # Verify removed from memory
            assert conv_id not in manager._conversations
            # Verify removed from database
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conv_id,),
                )
                result = await cursor.fetchone()
                assert result is None
            # Verify get_messages raises error
            with pytest.raises(ConversationNotFoundError):
                await manager.get_messages(conv_id)
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__delete_conversation__not_in_memory(self):
        """Test deleting a conversation that exists in DB but not in memory."""
        db_path = create_temp_db_path()
        try:
            # First manager to create conversation
            manager1 = ConversationManager(str(db_path))
            await manager1.initialize()
            conv_id = await manager1.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            message = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
                timestamp=timestamp,
            )
            await manager1.add_message(conv_id, message)
            # New manager instance (so conversation isn't cached in memory)
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            # Verify not in memory but exists in DB
            assert conv_id not in manager2._conversations
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conv_id,),
                )
                result = await cursor.fetchone()
                assert result is not None
            # Delete conversation
            await manager2.delete_conversation(conv_id)
            # Verify removed from database
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conv_id,),
                )
                result = await cursor.fetchone()
                assert result is None
            # Verify get_messages raises error
            with pytest.raises(ConversationNotFoundError):
                await manager2.get_messages(conv_id)
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__delete_conversation__nonexistent(self):
        """Test deleting a conversation that doesn't exist."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            with pytest.raises(ConversationNotFoundError):
                await manager.delete_conversation("nonexistent-id")
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__delete_conversation__concurrent_access(self):
        """Test concurrent deletion attempts."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create conversation
            conv_id = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            message = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, message)
            # Try to delete the same conversation concurrently
            async def delete_attempt() -> Optional[bool]:
                try:
                    await manager.delete_conversation(conv_id)
                    return True
                except ConversationNotFoundError:
                    return False
            results = await asyncio.gather(
                delete_attempt(),
                delete_attempt(),
                delete_attempt(),
            )
            # Exactly one deletion should succeed
            assert sum(results) == 1
            # Verify conversation is gone
            with pytest.raises(ConversationNotFoundError):
                await manager.get_messages(conv_id)
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__delete_conversation__persistence(self):
        """Test that deletion persists across manager instances."""
        db_path = create_temp_db_path()
        try:
            # First manager instance
            manager1 = ConversationManager(str(db_path))
            await manager1.initialize()
            # Create and delete a conversation
            conv_id = await manager1.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            message = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
                timestamp=timestamp,
            )
            await manager1.add_message(conv_id, message)
            await manager1.delete_conversation(conv_id)
            # New manager instance
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            # Verify conversation remains deleted
            with pytest.raises(ConversationNotFoundError):
                await manager2.get_messages(conv_id)
            # Verify not in database
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conv_id,),
                )
                result = await cursor.fetchone()
                assert result is None
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__delete_conversation__with_multiple_conversations(self):
        """Test deleting one conversation doesn't affect others."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create two conversations
            conv_id1 = await manager.create_conversation()
            conv_id2 = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            message1 = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message 1",
                ),
                timestamp=timestamp,
            )
            timestamp.GetCurrentTime()
            message2 = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message 2",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id1, message1)
            await manager.add_message(conv_id2, message2)
            # Delete first conversation
            await manager.delete_conversation(conv_id1)
            # Verify first conversation is gone
            with pytest.raises(ConversationNotFoundError):
                await manager.get_messages(conv_id1)
            # Verify second conversation is intact
            messages = await manager.get_messages(conv_id2)
            assert len(messages) == 1
            assert messages[0].chat_message.content == "Test message 2"
            # Verify database state
            async with aiosqlite.connect(str(db_path)) as db:
                # First conversation should be gone
                cursor = await db.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conv_id1,),
                )
                result = await cursor.fetchone()
                assert result is None
                # Second conversation should exist
                cursor = await db.execute(
                    "SELECT messages FROM conversations WHERE id = ?",
                    (conv_id2,),
                )
                result = await cursor.fetchone()
                assert result is not None
                messages_data = json.loads(result[0])
                assert len(messages_data) == 1
                assert messages_data[0]['chat_message']['content'] == "Test message 2"
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__truncate__single_message(self):
        """Test truncating a conversation with a single message."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create conversation with one message
            conv_id = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            message = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, message)
            # Truncate at the message
            await manager.truncate_conversation(conv_id, message.entry_id)
            # Verify conversation is empty
            messages = await manager.get_messages(conv_id)
            assert len(messages) == 0
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__truncate__middle_of_conversation(self):
        """Test truncating in the middle of a conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create conversation with multiple messages
            conv_id = await manager.create_conversation()
            messages = []
            for i in range(5):
                timestamp=timestamp_pb2.Timestamp()
                timestamp.GetCurrentTime()
                message = chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content=f"Message {i}",
                    ),
                    timestamp=timestamp,
                )
                messages.append(message)
                await manager.add_message(conv_id, message)
            # Truncate at message 2
            await manager.truncate_conversation(conv_id, messages[2].entry_id)
            # Verify only first two messages remain
            remaining = await manager.get_messages(conv_id)
            assert len(remaining) == 2
            assert remaining[0].chat_message.content == "Message 0"
            assert remaining[1].chat_message.content == "Message 1"
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__truncate__last_message(self):
        """Test truncating at the last message in a conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create conversation with multiple messages
            conv_id = await manager.create_conversation()
            messages = []
            for i in range(3):
                timestamp=timestamp_pb2.Timestamp()
                timestamp.GetCurrentTime()
                message = chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content=f"Message {i}",
                    ),
                    timestamp=timestamp,
                )
                messages.append(message)
                await manager.add_message(conv_id, message)

            # Truncate at last message
            await manager.truncate_conversation(conv_id, messages[-1].entry_id)
            # Verify only first two messages remain
            remaining = await manager.get_messages(conv_id)
            assert len(remaining) == 2
            assert remaining[0].chat_message.content == "Message 0"
            assert remaining[1].chat_message.content == "Message 1"
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__truncate__nonexistent_entry(self):
        """Test truncating at a nonexistent entry ID."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create conversation
            conv_id = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            message = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, message)
            # Try to truncate at nonexistent entry
            with pytest.raises(ValueError, match="Entry .* not found"):
                await manager.truncate_conversation(conv_id, "nonexistent-id")
            # Verify conversation unchanged
            messages = await manager.get_messages(conv_id)
            assert len(messages) == 1
            assert messages[0].chat_message.content == "Test message"
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__truncate__nonexistent_conversation(self):
        """Test truncating a nonexistent conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            with pytest.raises(ConversationNotFoundError):
                await manager.truncate_conversation("nonexistent-conv", "some-entry-id")
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__truncate__persistence(self):
        """Test that truncation persists across manager instances."""
        db_path = create_temp_db_path()
        try:
            manager1 = ConversationManager(str(db_path))
            await manager1.initialize()
            # Create conversation with messages
            conv_id = await manager1.create_conversation()
            messages = []
            for i in range(3):
                timestamp=timestamp_pb2.Timestamp()
                timestamp.GetCurrentTime()
                message = chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content=f"Message {i}",
                    ),
                    timestamp=timestamp,
                )
                messages.append(message)
                await manager1.add_message(conv_id, message)
            # Truncate at second message
            await manager1.truncate_conversation(conv_id, messages[1].entry_id)
            # Create new manager instance
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            # Verify truncation persisted
            remaining = await manager2.get_messages(conv_id)
            assert len(remaining) == 1
            assert remaining[0].chat_message.content == "Message 0"
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__truncate__mixed_message_types(self):
        """Test truncating with a mix of user messages and model responses."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()
            # Add user message
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            user_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="User message",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, user_msg)
            # Add model response
            timestamp.GetCurrentTime()
            model_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                single_model_response=chat_pb2.ChatModelResponse(
                    message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.ASSISTANT,
                        content="Model response",
                    ),
                    config_snapshot=chat_pb2.ModelConfig(
                        client_type="test",
                        model_name="test-model",
                    ),
                    model_index=0,
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, model_msg)
            # Add another user message
            timestamp.GetCurrentTime()
            user_msg2 = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Second user message",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, user_msg2)
            # Truncate at model response
            await manager.truncate_conversation(conv_id, model_msg.entry_id)
            # Verify only first message remains
            remaining = await manager.get_messages(conv_id)
            assert len(remaining) == 1
            assert remaining[0].chat_message.content == "User message"
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__concurrent_conversation_operations(self):
        """Test that operations on different conversations can happen concurrently."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create two conversations
            conv_id1 = await manager.create_conversation()
            conv_id2 = await manager.create_conversation()
            # Test concurrent operations
            async def add_messages(conv_id: str, prefix: str) -> None:
                for i in range(5):
                    timestamp=timestamp_pb2.Timestamp()
                    timestamp.GetCurrentTime()
                    message = chat_pb2.ConversationEntry(
                        entry_id=str(uuid4()),
                        chat_message=chat_pb2.ChatMessage(
                            role=chat_pb2.Role.USER,
                            content=f"{prefix} Message {i}",
                        ),
                        timestamp=timestamp,
                    )
                    await manager.add_message(conv_id, message)
            # Run operations concurrently
            await asyncio.gather(
                add_messages(conv_id1, "Conv1"),
                add_messages(conv_id2, "Conv2"),
            )
            # Verify both conversations were updated correctly
            messages1 = await manager.get_messages(conv_id1)
            messages2 = await manager.get_messages(conv_id2)
            assert len(messages1) == 5
            assert len(messages2) == 5
            assert all("Conv1" in msg.chat_message.content for msg in messages1)
            assert all("Conv2" in msg.chat_message.content for msg in messages2)
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__branch_conversation__basic(self):
        """Test basic branching of a conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create initial conversation
            conv_id = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            messages = [
                chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content="Message 1",
                    ),
                    timestamp=timestamp,
                ),
                chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content="Message 2",
                    ),
                    timestamp=timestamp,
                ),
                chat_pb2.ConversationEntry(
                    entry_id=str(uuid4()),
                    chat_message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content="Message 3",
                    ),
                    timestamp=timestamp,
                ),
            ]
            for msg in messages:
                await manager.add_message(conv_id, msg)
            # Branch at second message
            branch_id = await manager.branch_conversation(conv_id, messages[1].entry_id)
            history = await manager.get_all_conversations()
            assert len(history) == 2
            assert conv_id in {c.conversation_id for c in history}
            assert branch_id in {c.conversation_id for c in history}
            # verify new manager instance has the same conversations
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            history2 = await manager2.get_all_conversations()
            assert len(history2) == 2
            assert conv_id in {c.conversation_id for c in history2}
            assert branch_id in {c.conversation_id for c in history2}

            # Verify original conversation unchanged
            original_msgs = await manager.get_messages(conv_id)
            assert len(original_msgs) == 3
            assert [m.chat_message.content for m in original_msgs] == ["Message 1", "Message 2", "Message 3"]  # noqa: E501
            # Verify branched conversation
            branched_msgs = await manager.get_messages(branch_id)
            assert len(branched_msgs) == 2
            assert [m.chat_message.content for m in branched_msgs] == ["Message 1", "Message 2"]
            # NOTE that the timestamps of the last entry will not match since we create a new
            # timestamp for the branched messages
            assert branched_msgs[0].timestamp.nanos == messages[0].timestamp.nanos
            assert branched_msgs[1].timestamp.nanos != messages[1].timestamp.nanos
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.parametrize("model_index", [0, 1])
    async def test__branch_conversation__with_multi_model(self, model_index: int):
        """Test branching with multi-model response."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create initial conversation
            conv_id = await manager.create_conversation()
            # Add user message
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            user_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="User question",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, user_msg)
            # Add multi-model response
            timestamp.GetCurrentTime()
            multi_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 0",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test0", "test-model0"),  # noqa: E501
                            model_index=0,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 1",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test1", "test-model1"),  # noqa: E501
                            model_index=1,
                        ),
                    ],
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, multi_msg)
            # Add another message
            timestamp.GetCurrentTime()
            follow_up = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Follow up",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, follow_up)
            # Branch at multi-model response selecting model 1
            branch_id = await manager.branch_conversation(
                conv_id,
                multi_msg.entry_id,
                model_index=model_index,
            )
            history = await manager.get_all_conversations()
            assert len(history) == 2
            assert conv_id in {c.conversation_id for c in history}
            assert branch_id in {c.conversation_id for c in history}
            # verify new manager instance has the same conversations
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            history2 = await manager2.get_all_conversations()
            assert len(history2) == 2
            assert conv_id in {c.conversation_id for c in history2}
            assert branch_id in {c.conversation_id for c in history2}

            # Verify original conversation unchanged
            original_msgs = await manager.get_messages(conv_id)
            assert len(original_msgs) == 3
            assert original_msgs[1].multi_model_response == multi_msg.multi_model_response
            # Verify branched conversation
            branched_msgs = await manager.get_messages(branch_id)
            assert len(branched_msgs) == 2
            assert branched_msgs[0] == user_msg
            # Verify multi-model converted to single model
            assert branched_msgs[1].HasField("single_model_response")
            assert branched_msgs[1].single_model_response.message.content == f"Response from model {model_index}"  # noqa: E501
            assert branched_msgs[1].single_model_response.model_index == 0
            assert branched_msgs[1].single_model_response.config_snapshot.client_type == f"api-test{model_index}"  # noqa: E501
            assert branched_msgs[1].single_model_response.config_snapshot.model_parameters == create_fake_config_snapshot('', '').model_parameters  # noqa: E501
            # NOTE that the timestamps will not match since we create a new timestamp for the
            # branched messages
            assert branched_msgs[0].timestamp.nanos == user_msg.timestamp.nanos
            assert branched_msgs[1].timestamp.nanos != multi_msg.timestamp.nanos
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__branch_conversation__nonexistent_conversation(self):
        """Test branching from nonexistent conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            with pytest.raises(ConversationNotFoundError):
                await manager.branch_conversation("nonexistent-id", "entry-id")
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__branch_conversation__nonexistent_entry(self):
        """Test branching from nonexistent entry."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, msg)
            with pytest.raises(ValueError, match="Entry .* not found"):
                await manager.branch_conversation(conv_id, "nonexistent-entry")
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__branch_conversation__invalid_model_index(self):
        """Test branching with invalid model index scenarios."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create conversation with regular message
            conv_id = await manager.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            regular_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, regular_msg)

            # Add multi-model response
            multi_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 0",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test", "test-model"),
                            model_index=0,
                        ),
                    ],
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, multi_msg)

            # Test not providing model_index for multi-model response
            with pytest.raises(ValueError, match="no model_index provided"):
                await manager.branch_conversation(conv_id, multi_msg.entry_id)

            # Test invalid model index
            with pytest.raises(ValueError, match="Model index .* not found"):
                await manager.branch_conversation(conv_id, multi_msg.entry_id, model_index=99)
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__branch_conversation__persistence(self):
        """Test that branched conversations persist across manager instances."""
        db_path = create_temp_db_path()
        try:
            # First manager instance
            manager1 = ConversationManager(str(db_path))
            await manager1.initialize()
            # Create and branch a conversation
            conv_id = await manager1.create_conversation()
            timestamp=timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            msg1 = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Message 1",
                ),
                timestamp=timestamp,
            )
            timestamp.GetCurrentTime()
            msg2 = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Message 2",
                ),
                timestamp=timestamp,
            )
            await manager1.add_message(conv_id, msg1)
            await manager1.add_message(conv_id, msg2)
            branch_id = await manager1.branch_conversation(conv_id, msg1.entry_id)
            # New manager instance
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            # Verify both conversations exist and have correct content
            original_msgs = await manager2.get_messages(conv_id)
            assert len(original_msgs) == 2
            assert [m.chat_message.content for m in original_msgs] == ["Message 1", "Message 2"]
            branched_msgs = await manager2.get_messages(branch_id)
            assert len(branched_msgs) == 1
            assert branched_msgs[0].chat_message.content == "Message 1"
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.parametrize("selected_model_index", [0, 1])
    async def test__set_multi_response_selected_model__success(self, selected_model_index: int):
        """Test successful setting of selected model index."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            # Create a conversation with a multi-model response
            conv_id = await manager.create_conversation()
            timestamp = timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()

            # Add user message
            user_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, user_msg)

            # Add multi-model response
            multi_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 0",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test0", "test-model0"),  # noqa: E501
                            model_index=0,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 1",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test1", "test-model1"),  # noqa: E501
                            model_index=1,
                        ),
                    ],
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, multi_msg)
            original_messages = await manager.get_messages(conv_id)
            assert len(original_messages) == 2
            assert not original_messages[1].multi_model_response.HasField("selected_model_index")

            # Set selected model index
            await manager.set_multi_response_selected_model(
                conv_id=conv_id,
                entry_id=multi_msg.entry_id,
                selected_model_index=selected_model_index,
            )
            # Verify selection was set
            messages = await manager.get_messages(conv_id)
            assert len(messages) == 2
            assert messages[1].multi_model_response.HasField("selected_model_index")
            assert messages[1].multi_model_response.selected_model_index.value == selected_model_index  # noqa: E501

            # Test with new manager instance to verify persistence
            manager2 = ConversationManager(str(db_path))
            await manager2.initialize()
            messages2 = await manager2.get_messages(conv_id)
            assert len(messages2) == 2
            assert messages2[1].multi_model_response.HasField("selected_model_index")
            assert messages2[1].multi_model_response.selected_model_index.value == selected_model_index  # noqa: E501
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__set_multi_response_selected_model__nonexistent_conversation(self):
        """Test setting selected model index for nonexistent conversation."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            with pytest.raises(ConversationNotFoundError):
                await manager.set_multi_response_selected_model(
                    conv_id="nonexistent",
                    entry_id="some-entry",
                    selected_model_index=0,
                )
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__set_multi_response_selected_model__nonexistent_entry(self):
        """Test setting selected model index for nonexistent entry."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()
            with pytest.raises(ValueError, match="Entry .* not found"):
                await manager.set_multi_response_selected_model(
                    conv_id=conv_id,
                    entry_id="nonexistent",
                    selected_model_index=0,
                )
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__set_multi_response_selected_model__non_multi_response_entry(self):
        """Test setting selected model index for non-multi-response entry."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()
            # Add regular message
            timestamp = timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test",
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, msg)

            with pytest.raises(ValueError, match="not a multi-model response"):
                await manager.set_multi_response_selected_model(
                    conv_id=conv_id,
                    entry_id=msg.entry_id,
                    selected_model_index=0,
                )
        finally:
            Path(db_path).unlink(missing_ok=True)

    async def test__set_multi_response_selected_model__invalid_model_index(self):
        """Test setting invalid model index."""
        db_path = create_temp_db_path()
        try:
            manager = ConversationManager(str(db_path))
            await manager.initialize()
            conv_id = await manager.create_conversation()
            timestamp = timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()

            # Add multi-model response
            multi_msg = chat_pb2.ConversationEntry(
                entry_id=str(uuid4()),
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 0",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test", "test-model"),
                            model_index=0,
                        ),
                    ],
                ),
                timestamp=timestamp,
            )
            await manager.add_message(conv_id, multi_msg)

            with pytest.raises(ValueError, match="Model index .* not found"):
                await manager.set_multi_response_selected_model(
                    conv_id=conv_id,
                    entry_id=multi_msg.entry_id,
                    selected_model_index=99,
                )
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestConvertProtoMessages:
    """Tests for convert_proto_messages_to_model_messages."""

    def test__convert_proto__messages_basic(self):
        """Test basic message conversion."""
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ),
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.ASSISTANT,
                    content="Hi there",
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages)
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert result[1] == {'role': 'assistant', 'content': 'Hi there'}

    def test__convert_proto__messages_single_model(self):
        """Test conversion with single model response."""
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ),
            chat_pb2.ConversationEntry(
                single_model_response=chat_pb2.ChatModelResponse(
                    message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.ASSISTANT,
                        content="Model response",
                    ),
                    config_snapshot=create_fake_config_snapshot("api-test", "test_model"),
                    model_index=0,
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages)
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert result[1] == {'role': 'assistant', 'content': 'Model response'}

    def test__convert_proto__messages_multi_model(self):
        """Test conversion with multi-model response."""
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ),
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="First model",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test", "test_model"),
                            model_index=0,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Second model",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test", "test_model2"),
                            model_index=1,
                        ),
                    ],
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages)
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert result[1] == {'role': 'assistant', 'content': 'First model'}

    def test__convert_proto__messages_mixed(self):
        """Test conversion with mix of message types."""
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="User query",
                ),
            ),
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Model A",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test", "test_model"),
                            model_index=0,
                        ),
                    ],
                ),
            ),
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.SYSTEM,
                    content="System message",
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages)
        assert len(result) == 3
        assert result[0] == {'role': 'user', 'content': 'User query'}
        assert result[1] == {'role': 'assistant', 'content': 'Model A'}
        assert result[2] == {'role': 'system', 'content': 'System message'}

    def test__convert_proto__messages_multi_model_with_matching_config(self):
        """
        Test conversion with multi-model response and matching model config including
        parameters.
        """
        model_config = chat_pb2.ModelConfig(
            client_type="OpenAI",
            model_name="gpt-4",
            model_parameters=chat_pb2.ModelParameters(
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
            ),
        )
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ),
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Different params response",
                            ),
                            config_snapshot=chat_pb2.ModelConfig(
                                client_type="OpenAI",
                                model_name="gpt-4",
                                model_parameters=chat_pb2.ModelParameters(
                                    temperature=0.5,  # Different temperature
                                    max_tokens=100,
                                    top_p=0.9,
                                ),
                            ),
                            model_index=2,  # test model_index unordered
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Different params response",
                            ),
                            config_snapshot=chat_pb2.ModelConfig(
                                client_type="OpenAI",
                                model_name="gpt-4",
                                model_parameters=chat_pb2.ModelParameters(
                                    temperature=0.5,  # Different temperature
                                    max_tokens=100,
                                    top_p=0.9,
                                ),
                            ),
                            model_index=0,  # test model_index unordered
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Matching params response",
                            ),
                            config_snapshot=chat_pb2.ModelConfig(
                                client_type="OpenAI",
                                model_name="gpt-4",
                                model_parameters=chat_pb2.ModelParameters(
                                    temperature=0.7,
                                    max_tokens=100,
                                    top_p=0.9,
                                ),
                            ),
                            model_index=1,  # test model_index unordered
                        ),
                    ],
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages, model_config)
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert result[1] == {'role': 'assistant', 'content': 'Matching params response'}

    def test__convert_proto__messages_multi_model_no_match_due_to_params(self):
        """Test conversion fails when model matches but parameters don't."""
        model_config = chat_pb2.ModelConfig(
            client_type="OpenAI",
            model_name="gpt-4",
            model_parameters=chat_pb2.ModelParameters(
                temperature=0.7,
                max_tokens=100,
            ),
        )
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ),
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Should not match",
                            ),
                            config_snapshot=chat_pb2.ModelConfig(
                                client_type="OpenAI",
                                model_name="gpt-4",
                                model_parameters=chat_pb2.ModelParameters(
                                    temperature=0.5,
                                    max_tokens=100,
                                ),
                            ),
                            model_index=1,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                # Does not match from params but should be selected
                                # because model_index=0
                                content="Should match.",
                            ),
                            config_snapshot=chat_pb2.ModelConfig(
                                client_type="OpenAI",
                                model_name="gpt-4",
                                model_parameters=chat_pb2.ModelParameters(
                                    temperature=0.7,
                                    max_tokens=200,
                                ),
                            ),
                            model_index=0,
                        ),
                    ],
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages, model_config)
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert result[1] == {'role': 'assistant', 'content': 'Should match.'}

    def test__convert_proto__messages_multi_model_no_config(self):
        """Test conversion with multi-model response and no model config provided."""
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ),
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Model Index 1",
                            ),
                            config_snapshot=chat_pb2.ModelConfig(
                                client_type="OpenAI",
                                model_name="gpt-4",
                                model_parameters=chat_pb2.ModelParameters(
                                    temperature=0.7,
                                    max_tokens=100,
                                ),
                            ),
                            model_index=1,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Model Index 0",
                            ),
                            config_snapshot=chat_pb2.ModelConfig(
                                client_type="Anthropic",
                                model_name="claude-3",
                                model_parameters=chat_pb2.ModelParameters(
                                    temperature=0.5,
                                    max_tokens=200,
                                ),
                            ),
                            model_index=0,
                        ),
                    ],
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages)
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert result[1] == {'role': 'assistant', 'content': 'Model Index 0'}

    def test__convert_to_conversations(self):
        with open('artifacts/example_history.yaml') as f:
            content = f.read()
            yaml_data = yaml.safe_load(content)
            yaml_conversations = yaml_data['conversations']

        conversations = convert_to_conversations(yaml_conversations)
        assert len(conversations) == len(yaml_conversations)

    def test__convert_proto_messages__multi_model_with_selected_index(self):
        """Test conversion with selected_model_index set."""
        messages = [
            chat_pb2.ConversationEntry(
                chat_message=chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ),
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 0",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test0", "test-model0"),  # noqa: E501
                            model_index=0,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 1",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test1", "test-model1"),  # noqa: E501
                            model_index=1,
                        ),
                    ],
                    selected_model_index=wrappers_pb2.Int32Value(value=1),
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages)
        assert len(result) == 2
        assert result[0] == {'role': 'user', 'content': 'Hello'}
        assert result[1] == {'role': 'assistant', 'content': 'Response from model 1'}

    @pytest.mark.parametrize("selected_model_index", [0, 1])
    def test__convert_proto_messages__multi_model_selected_index_overrides_config(self, selected_model_index: int):  # noqa: E501
        """
        Test that selected_model_index takes precedence over matching config.

        Model 0 has a matching config but response associated with `selected_model_index` is
        selected. If selected_model_index is 1, then it will override the matching config 0.
        """
        model_config = chat_pb2.ModelConfig(
            client_type="api-test0",
            model_name="test-model0",
            model_parameters=chat_pb2.ModelParameters(
                temperature=0.7,
                max_tokens=100,
            ),
        )
        messages = [
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 0",
                            ),
                            config_snapshot=model_config,
                            model_index=0,
                        ),
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 1",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test1", "test-model1"),  # noqa: E501
                            model_index=selected_model_index,
                        ),
                    ],
                    selected_model_index=wrappers_pb2.Int32Value(value=selected_model_index),
                ),
            ),
        ]
        result = convert_proto_messages_to_model_messages(messages, model_config)
        assert len(result) == 1
        # Should use model 1's response despite model 0 matching the config
        assert result[0] == {'role': 'assistant', 'content': f'Response from model {selected_model_index}'}  # noqa: E501

    def test__convert_proto_messages__multi_model_nonexistent_selected_index(self):
        """Test handling of nonexistent selected_model_index."""
        messages = [
            chat_pb2.ConversationEntry(
                multi_model_response=chat_pb2.MultiChatModelResponse(
                    responses=[
                        chat_pb2.ChatModelResponse(
                            message=chat_pb2.ChatMessage(
                                role=chat_pb2.Role.ASSISTANT,
                                content="Response from model 0",
                            ),
                            config_snapshot=create_fake_config_snapshot("api-test0", "test-model0"),  # noqa: E501
                            model_index=0,
                        ),
                    ],
                    selected_model_index=wrappers_pb2.Int32Value(value=99),  # Invalid index
                ),
            ),
        ]
        # Should fall back to model_index=0 since selected index doesn't exist
        result = convert_proto_messages_to_model_messages(messages)
        assert len(result) == 1
        assert result[0] == {'role': 'assistant', 'content': 'Response from model 0'}
