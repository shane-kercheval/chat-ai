"""SQLite-backed model configuration manager."""
from uuid import uuid4
import aiosqlite
import asyncio
import logging
from proto.generated import chat_pb2


class ModelConfigManager:
    """Manages model info and configurations with SQLite backing."""

    def __init__(self, db_path: str, default_model_configs: list[dict]):
        self.db_path = db_path
        self._default_model_configs = default_model_configs
        self._logger = logging.getLogger(__name__)
        self._db_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database and load models from YAML."""
        async with self._db_lock, aiosqlite.connect(self.db_path) as db:
            # Create tables if they don't exist
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_configs (
                    config_id TEXT PRIMARY KEY,
                    config_name TEXT NOT NULL,
                    client_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    temperature REAL,
                    max_tokens INTEGER,
                    top_p REAL,
                    server_url TEXT
                )
            """)
            await db.commit()

        # Insert default configurations if table count is 0
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_configs") as cursor:
                row = await cursor.fetchone()
            if row[0] == 0:
                async with self._db_lock:
                    # Insert default configurations if they don't exist
                    for config in self._default_model_configs:
                        await db.execute("""
                            INSERT OR IGNORE INTO model_configs
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                config['config_id'],
                                config['config_name'],
                                config['config']['client_type'],
                                config['config']['model_name'],
                                config['config']['model_parameters'].get('temperature'),
                                config['config']['model_parameters'].get('max_tokens'),
                                config['config']['model_parameters'].get('top_p'),
                                config['config']['model_parameters'].get('server_url'),
                            ))
                    await db.commit()

    async def get_model_configs(self) -> list[chat_pb2.UserModelConfig]:
        """Get all model configurations for a user."""
        async with aiosqlite.connect(self.db_path) as db, db.execute(
            "SELECT * FROM model_configs",
        ) as cursor:
            rows = await cursor.fetchall()
            configs = []
            for row in rows:
                config = chat_pb2.UserModelConfig(
                    config_id=row[0],
                    config_name=row[1],
                    config=chat_pb2.ModelConfig(
                        client_type=row[2],
                        model_name=row[3],
                        model_parameters=chat_pb2.ModelParameters(
                            temperature=row[4],
                            max_tokens=row[5],
                            top_p=row[6],
                            server_url=row[7],
                        ),
                    ),
                )
                configs.append(config)
            return configs

    async def get_model_config(self, config_id: str) -> chat_pb2.UserModelConfig | None:
        """
        Get a specific model configuration.

        Args:
            config_id: The ID of the configuration to retrieve

        Returns:
            The requested UserModelConfig or None if not found
        """
        async with aiosqlite.connect(self.db_path) as db, db.execute(
            "SELECT * FROM model_configs WHERE config_id = ?",
            (config_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return chat_pb2.UserModelConfig(
                config_id=row[0],
                config_name=row[1],
                config=chat_pb2.ModelConfig(
                    client_type=row[2],
                    model_name=row[3],
                    model_parameters=chat_pb2.ModelParameters(
                        temperature=row[4],
                        max_tokens=row[5],
                        top_p=row[6],
                        server_url=row[7],
                    ),
                ),
            )

    async def save_model_config(
        self,
        config: chat_pb2.UserModelConfig,
    ) -> chat_pb2.UserModelConfig:
        """
        Save (create or update) a model configuration.

        Args:
            config: The configuration to save. If id is provided, updates existing config.

        Returns:
            The saved configuration with generated ID if new.

        Raises:
            ValueError: If updating a non-existent config.
        """
        async with self._db_lock, aiosqlite.connect(self.db_path) as db:
            config_id = config.config_id if config.HasField('config_id') else str(uuid4())
            await db.execute("""
                INSERT OR REPLACE INTO model_configs
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config_id,
                config.config_name,
                config.config.client_type,
                config.config.model_name,
                config.config.model_parameters.temperature if config.config.model_parameters.HasField('temperature') else None,  # noqa: E501
                config.config.model_parameters.max_tokens if config.config.model_parameters.HasField('max_tokens') else None,  # noqa: E501
                config.config.model_parameters.top_p if config.config.model_parameters.HasField('top_p') else None,  # noqa: E501
                config.config.model_parameters.server_url if config.config.model_parameters.HasField('server_url') else None,  # noqa: E501
            ))
            await db.commit()

            # Return the saved config
            async with db.execute(
                "SELECT * FROM model_configs WHERE config_id = ?",
                (config_id,),
            ) as cursor:
                row = await cursor.fetchone()
                return chat_pb2.UserModelConfig(
                    config_id=row[0],
                    config_name=row[1],
                    config=chat_pb2.ModelConfig(
                        client_type=row[2],
                        model_name=row[3],
                        model_parameters=chat_pb2.ModelParameters(
                            temperature=row[4],
                            max_tokens=row[5],
                            top_p=row[6],
                            server_url=row[7],
                        ),
                    ),
                )

    async def delete_model_config(self, config_id: str) -> None:
        """
        Delete a model configuration.

        Args:
            config_id: ID of config to delete

        Raises:
            ValueError: If config doesn't exists
        """
        async with self._db_lock, aiosqlite.connect(self.db_path) as db:
            # Verify config exists and belongs to user
            async with db.execute(
                "SELECT * FROM model_configs WHERE config_id = ?",
                (config_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ValueError(f"Config {config_id} does not exist")

            await db.execute(
                "DELETE FROM model_configs WHERE config_id = ?",
                (config_id,),
            )
            await db.commit()
