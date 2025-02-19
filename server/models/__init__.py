"""Public facing functions and classes for models."""
from pydantic import BaseModel
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from copy import deepcopy
from enum import Enum
from typing import Any, TypeVar
from server.utilities import Registry


def user_message(content: str) -> dict:
    """Returns a user message."""
    return {'role': 'user', 'content': content}


def assistant_message(content: str) -> dict:
    """Returns an assistant message."""
    return {'role': 'assistant', 'content': content}


def system_message(content: str) -> dict:
    """Returns a system message."""
    return {'role': 'system', 'content': content}


class ChatChunkResponse(BaseModel):
    """A chunk returned when streaming."""

    content: str
    logprob: float | None = None


class ChatStreamResponseSummary(BaseModel):
    """Summary of a chat response."""

    total_input_tokens: int
    total_output_tokens: int
    total_input_cost: float
    total_output_cost: float
    duration_seconds: float


class Parameter(BaseModel):
    """
    Represents a parameter property in a function's schema.

    Supported types
        The following types are supported for Structured Outputs:

        String
        Number
        Boolean
        Integer
        Object
        Array
        Enum
        anyOf

    """

    name: str
    type: str
    required: bool
    description: str | None = None
    enum: list[str] | None = None


class Function(BaseModel):
    """Represents a function that can be called by the model."""

    name: str
    parameters: list[Parameter]
    description: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the function to the format expected by OpenAI API."""
        properties = {}
        required = []

        for param in self.parameters:
            param_dict = {"type": param.type}
            if param.description:
                param_dict["description"] = param.description
            if param.enum:
                param_dict["enum"] = param.enum
            properties[param.name] = param_dict
            if param.required:
                required.append(param.name)

        parameters_dict = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters_dict["required"] = required
        parameters_dict["additionalProperties"] = False

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "strict": True,
                **({"description": self.description} if self.description else {}),
                "parameters": parameters_dict,
            },
        }

class FunctionCallResult(BaseModel):
    """The function call details extracted from the model's response."""

    name: str
    arguments: dict[str, object]
    call_id: str

class FunctionCallResponse(BaseModel):
    """Response containing just the essential function call information and usage stats."""

    function_call: FunctionCallResult
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float



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
