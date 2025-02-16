"""Base functionality and classes for various models."""
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from copy import deepcopy
from enum import Enum
from typing import Any, TypeVar

from server.models import ChatChunkResponse, ChatStreamResponseSummary
from server.utilities import Registry

M = TypeVar('M', bound='Model')


class Model(ABC):
    """Base class for model wrappers."""

    registry = Registry()

    @abstractmethod
    async def __call__(
        self,
        messages: list[dict[str, Any]],
        model_name: str | None = None,
        **model_kwargs: dict[str, Any],
    ) -> AsyncGenerator[ChatChunkResponse | ChatStreamResponseSummary, None]:
        """
        Send messages to model (e.g. chat).

        Args:
            messages:
                List of messages to send to the model (i.e. model input).
            model_name:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            **model_kwargs:
                Additional parameters to pass to the API call (e.g. temperature, max_tokens).
        """
        pass

    @classmethod
    def register(cls, model_type: str | Enum):
        """Register a subclass of Model."""

        def decorator(subclass: type[Model]) -> type[Model]:
            assert issubclass(
                subclass,
                Model,
            ), f"Model '{model_type}' ({subclass.__name__}) must extend Model"
            cls.registry.register(type_name=model_type, item=subclass)
            return subclass

        return decorator

    @classmethod
    def is_registered(cls, model_type: str | Enum) -> bool:
        """Check if a model type is registered."""
        return model_type in cls.registry

    @classmethod
    def instantiate(
        cls: type[M],
        data: dict,
    ) -> M | list[M]:
        """
        Creates a Model object.

        This method requires that the Model subclass has been registered with the `register`
        decorator before calling this method. It also requires that the dictionary has a
        `Model_type` field that matches the type name of the registered Model subclass.

        Args:
            data:
                Dictionary containing the model data in the format:
                {
                    # required
                    'model_type': 'OpenAI',
                    'model_name': 'gpt-4o-mini',
                    # optional model parameters
                    'temperature': 0.5,
                    'max_tokens': 100,
                    'top_p': 0.9,
                }
        """
        data = deepcopy(data)
        if 'model_type' not in data:
            raise ValueError("Model type not found in data")
        model_type = data.pop('model_type')
        if cls.is_registered(model_type):
            return cls.registry.create_instance(type_name=model_type, **data)
        raise ValueError(f"Unknown Model type `{model_type}`")
