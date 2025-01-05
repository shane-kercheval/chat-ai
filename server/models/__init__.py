"""Public facing functions and classes for models."""
from pydantic import BaseModel

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
