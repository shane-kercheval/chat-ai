"""Tests for ModelManager."""
import tempfile
from pathlib import Path
import pytest
import aiosqlite

from proto.generated import chat_pb2
from server.model_config_manager import ModelConfigManager


def create_temp_db():
    """Create a temporary database file."""
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)  # noqa: SIM115
    f.close()
    return Path(f.name)


def fake_model_configs():
    return [
        {
            'config_id': '7398FE35-5635-4297-A6F6-4AEF5AA88506',
            'config_name': 'GPT-4o-Mini - Default',
            'config': {
                'model_type': 'OpenAI',
                'model_name': 'gpt-4o-mini',
                'model_parameters': {
                    'temperature': 0.2,
                },
            },
        },
        {
            'config_id': '0FD8F56C-B9C5-4477-9980-AA5885BBFA08',
            'config_name': 'http://localhost:1234/v1',
            'config': {
                'model_type': 'OpenAI',
                'model_name': 'openai-compatible-server',
                'model_parameters': {
                    'temperature': 0.3,
                    'server_url': 'http://localhost:1234/v1',
                },
            },
        },
    ]

@pytest.mark.asyncio
class TestModelManager:
    """Tests for the ModelManager."""

    async def test__initialization(self):
        """Test database initialization and YAML loading."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()

            # Verify tables exist and data was loaded correctly
            async with aiosqlite.connect(str(db_path)) as db:
                # Check model_configs table
                cursor = await db.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='model_configs'
                """)
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == "model_configs"

                 # Verify model configs was loaded
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                count = await cursor.fetchone()
                assert count[0] == 2

                # Verify local default config
                cursor = await db.execute("""
                    SELECT *
                    FROM model_configs
                    WHERE config_name = 'http://localhost:1234/v1'
                """)
                result = await cursor.fetchone()
                assert result is not None
                assert result[0] == '0FD8F56C-B9C5-4477-9980-AA5885BBFA08'  # id
                assert result[1] == 'http://localhost:1234/v1'  # config_name
                assert result[2] == 'OpenAI'
                assert result[3] == 'openai-compatible-server'
                assert result[4] == 0.3  # temperature
                assert result[5] is None  # max_tokens
                assert result[6] is None  # top_p
                assert result[7] == 'http://localhost:1234/v1'  # server_url
        finally:
            db_path.unlink(missing_ok=True)

    async def test__default_config_idempotency(self):
        """Test that default configs aren't duplicated on multiple initializations."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()
            # Initialize again
            await manager.initialize()
            # Verify no duplicate configs
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                count = await cursor.fetchone()
                assert count[0] == 2  # Should still be just the original 2 configs

        finally:
            db_path.unlink(missing_ok=True)

    async def test__get_model_configs(self):
        """Test getting configs for a user who already has configs."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()
            # First get creates configs
            configs = await manager.get_model_configs()
            assert len(configs) == 2

            configs[0].config_id = '7398FE35-5635-4297-A6F6-4AEF5AA88506'
            configs[0].config_name = 'GPT-4o-Mini - Default'
            configs[0].config.model_type = 'OpenAI'
            configs[0].config.model_name = 'gpt-4o-mini'
            configs[0].config.model_parameters.HasField('temperature')
            configs[0].config.model_parameters.temperature = 0.2
            assert not configs[0].config.model_parameters.HasField('max_tokens')
            assert not configs[0].config.model_parameters.HasField('top_p')
            assert not configs[0].config.model_parameters.HasField('server_url')
        finally:
            db_path.unlink(missing_ok=True)

    async def test__save_model_config_new(self):
        """Test saving a new model configuration."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()
            new_config = chat_pb2.UserModelConfig(
                config_name='Test Config',
                config=chat_pb2.ModelConfig(
                    model_type='Test Model Type',
                    model_name='Test Model Name',
                    model_parameters=chat_pb2.ModelParameters(
                        temperature=0.7,
                        max_tokens=1000,
                    ),
                ),
            )
            assert not new_config.HasField('config_id')  # Should not have ID yet
            async with aiosqlite.connect(str(db_path)) as db, db.execute(
                "SELECT COUNT(*) FROM model_configs",
            ) as cursor:
                count = await cursor.fetchone()
                assert count[0] == 2

            saved_config = await manager.save_model_config(new_config)
            assert saved_config.HasField('config_id')  # Should have generated ID
            assert saved_config.config_id is not None
            assert saved_config.config_name == 'Test Config'
            assert saved_config.config.model_type == 'Test Model Type'
            assert saved_config.config.model_name == 'Test Model Name'
            assert saved_config.config.model_parameters.HasField('temperature')
            assert saved_config.config.model_parameters.temperature == pytest.approx(0.7, rel=1e-6)
            assert saved_config.config.model_parameters.HasField('max_tokens')
            assert saved_config.config.model_parameters.max_tokens == 1000
            assert not saved_config.config.model_parameters.HasField('top_p')
            assert not saved_config.config.model_parameters.HasField('server_url')

            # check row count of model_configs table
            async with aiosqlite.connect(str(db_path)) as db, db.execute(
                "SELECT COUNT(*) FROM model_configs",
            ) as cursor:
                count = await cursor.fetchone()
                assert count[0] == 3
        finally:
            db_path.unlink(missing_ok=True)

    async def test__save_model_config_update(self):
        """Test updating an existing model configuration."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()

            # First create a config
            new_config = chat_pb2.UserModelConfig(
                config_name='Original Name',
                config=chat_pb2.ModelConfig(
                    model_type='Test Model Type',
                    model_name='Test Model Name',
                    model_parameters=chat_pb2.ModelParameters(
                        temperature=0.7,
                    ),
                ),
            )
            assert not new_config.HasField('config_id')  # Should not have ID yet
            saved_config = await manager.save_model_config(new_config)
            assert saved_config.HasField('config_id')  # Should have generated ID
            assert saved_config.config_id is not None
            assert saved_config.config_name == 'Original Name'
            assert saved_config.config.model_type == 'Test Model Type'
            assert saved_config.config.model_name == 'Test Model Name'
            assert saved_config.config.model_parameters.HasField('temperature')
            assert saved_config.config.model_parameters.temperature == pytest.approx(0.7, rel=1e-6)
            assert not saved_config.config.model_parameters.HasField('max_tokens')
            assert not saved_config.config.model_parameters.HasField('top_p')
            assert not saved_config.config.model_parameters.HasField('server_url')

            # check row count of model_configs table
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                count = await cursor.fetchone()
                assert count[0] == 3

            # Update the config
            update_config = chat_pb2.UserModelConfig(
                config_id=saved_config.config_id,  # get the ID from the saved config
                config_name='Updated Name',
                config=chat_pb2.ModelConfig(
                    model_type='Updated Model Type',
                    model_name='Updated Model Name',
                    model_parameters=chat_pb2.ModelParameters(
                        temperature=0.8,
                        max_tokens=2000,
                        top_p=0.5,
                        server_url='http://localhost:1234/v1',
                    ),
                ),
            )
            updated_config = await manager.save_model_config(update_config)
            assert updated_config.config_id == saved_config.config_id
            assert updated_config.config_name == 'Updated Name'
            assert updated_config.config.model_type == 'Updated Model Type'
            assert updated_config.config.model_name == 'Updated Model Name'
            assert updated_config.config.model_parameters.HasField('temperature')
            assert updated_config.config.model_parameters.temperature == pytest.approx(0.8, rel=1e-6)  # noqa: E501
            assert updated_config.config.model_parameters.HasField('max_tokens')
            assert updated_config.config.model_parameters.max_tokens == 2000
            assert updated_config.config.model_parameters.HasField('top_p')
            assert updated_config.config.model_parameters.top_p == pytest.approx(0.5, rel=1e-6)
            assert updated_config.config.model_parameters.HasField('server_url')
            assert updated_config.config.model_parameters.server_url == 'http://localhost:1234/v1'

            # check row count of model_configs table
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                count = await cursor.fetchone()
                assert count[0] == 3
        finally:
            db_path.unlink(missing_ok=True)

    async def test__delete_model_config(self):
        """Test deleting a model configuration."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()

            # get current number of configs
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                config_count = await cursor.fetchone()
                assert config_count[0] == 2

            config_to_delete = await manager.get_model_configs()
            config_to_delete = config_to_delete[0]

            await manager.delete_model_config(config_to_delete.config_id)
            # check row count of model_configs table
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                config_count = await cursor.fetchone()
                assert config_count[0] == 1

                # ensure the id of the deleted config is not in the table
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs WHERE config_id = ?", (config_to_delete.config_id,))  # noqa: E501
                count = await cursor.fetchone()
                assert count[0] == 0

            # Create a config
            config = chat_pb2.UserModelConfig(
                config_name='Test Config',
                config=chat_pb2.ModelConfig(
                    model_type='Test Model Type',
                    model_name='Test Model Name',
                    model_parameters=chat_pb2.ModelParameters(
                        temperature=0.7,
                    ),
                ),
            )
            saved_config = await manager.save_model_config(config)
            configs = await manager.get_model_configs()
            assert len(configs) == 2
            assert saved_config.config_id is not None
            assert saved_config.config_id in [c.config_id for c in configs]

            # cross reference
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                config_count = await cursor.fetchone()
                assert config_count[0] == 2

                cursor = await db.execute("SELECT COUNT(*) FROM model_configs WHERE config_id = ?", (saved_config.config_id,))  # noqa: E501
                count = await cursor.fetchone()
                assert count[0] == 1

            # Delete the config
            await manager.delete_model_config(saved_config.config_id)

            # cross reference
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                config_count = await cursor.fetchone()
                assert config_count[0] == 1

            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs WHERE config_id = ?", (saved_config.config_id,))  # noqa: E501
                count = await cursor.fetchone()
                assert count[0] == 0

        finally:
            db_path.unlink(missing_ok=True)

    async def test__delete_model_config_nonexistent(self):
        """Test error handling when deleting non-existent config."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()

            with pytest.raises(ValueError, match="Config .* does not exist"):
                await manager.delete_model_config("nonexistent-id")

            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM model_configs")
                config_count = await cursor.fetchone()
                assert config_count[0] == 2

        finally:
            db_path.unlink(missing_ok=True)

    async def test__get_model_config(self):
        """Test getting a specific model configuration."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()

            # First get copies default configs for new user
            configs = await manager.get_model_configs()
            config_id = configs[0].config_id

            # Get specific config
            config = await manager.get_model_config(config_id)
            assert config is not None
            assert config.config_id == config_id
            assert config.config_name == configs[0].config_name
            assert config.config.model_type == configs[0].config.model_type
            assert config.config.model_name == configs[0].config.model_name

        finally:
            db_path.unlink(missing_ok=True)

    async def test__get_model_config__nonexistent(self):
        """Test getting a non-existent config."""
        db_path = create_temp_db()
        try:
            manager = ModelConfigManager(db_path=str(db_path), default_model_configs=fake_model_configs())  # noqa: E501
            await manager.initialize()

            config = await manager.get_model_config('nonexistent-id')
            assert config is None

        finally:
            db_path.unlink(missing_ok=True)
