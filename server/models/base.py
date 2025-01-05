"""Base functionality and classes for various models."""
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from server.models import ChatChunkResponse, ChatStreamResponseSummary


class BaseModelWrapper(ABC):
    """Base class for model wrappers."""

    @abstractmethod
    async def __call__(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **model_kwargs: dict[str, Any],
    ) -> AsyncGenerator[ChatChunkResponse | ChatStreamResponseSummary, None]:
        """
        Send messages to model (e.g. chat).

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
            model:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            **model_kwargs:
                Additional parameters to pass to the API call (e.g. temperature, max_tokens).
        """
        pass

    @classmethod
    @abstractmethod
    def provider_name(cls) -> str:
        """Get the provider name of the model."""

    @classmethod
    @abstractmethod
    def primary_chat_model_names(cls) -> list[str]:
        """Get the primary model names (e.g. the models we would want to display to the user)."""

    @classmethod
    @abstractmethod
    def supported_chat_model_names(cls) -> list[str]:
        """Get all model names supported by the wrapper."""

    @classmethod
    @abstractmethod
    def cost_per_token(cls, model_name: str, token_type: str) -> float:
        """Get the cost per token for the model."""
