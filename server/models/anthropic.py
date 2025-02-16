"""Helper functions for interacting with the Anthropic API."""
import os
import time
from collections.abc import AsyncGenerator
from anthropic import AsyncAnthropic, Anthropic
from server.models.base import Model, ChatChunkResponse, ChatStreamResponseSummary


ANTHROPIC = 'Anthropic'

CHAT_MODEL_COST_PER_TOKEN = {
    'claude-3-5-haiku-20241022': {'input': 0.80 / 1_000_000, 'output': 4.0 / 1_000_000},
    'claude-3-5-sonnet-20241022': {'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000},
    'claude-3-opus-20240229': {'input': 15.00 / 1_000_000, 'output': 75.00 / 1_000_000},
}
CHAT_MODEL_COST_PER_TOKEN_LATEST = {
    'claude-3-5-haiku-latest': CHAT_MODEL_COST_PER_TOKEN['claude-3-5-haiku-20241022'],
    'claude-3-5-sonnet-latest': CHAT_MODEL_COST_PER_TOKEN['claude-3-5-sonnet-20241022'],
    'claude-3-opus-latest': CHAT_MODEL_COST_PER_TOKEN['claude-3-opus-20240229'],
}
CHAT_MODEL_COST_PER_TOKEN.update(CHAT_MODEL_COST_PER_TOKEN_LATEST)


def num_tokens(model_name: str, content: str) -> int:
    """Returns the number of tokens for a given string."""
    client = Anthropic()
    response = client.messages.count_tokens(
        model=model_name,
        messages=[{"role": "user", "content": content}],
    )
    return response.tokens

def num_tokens_from_messages(model_name: str, messages: list[dict]) -> int:
    """Returns the number of tokens for a list of messages."""
    client = Anthropic()
    response = client.messages.count_tokens(
        model=model_name,
        messages=messages,
    )
    return response.tokens


def _parse_completion_chunk(chunk) -> ChatChunkResponse | None:  # noqa: ANN001
    """Parse a chunk from the Anthropic API."""
    # Different chunk types in Anthropic's streaming response
    if hasattr(chunk, 'type'):
        if chunk.type == 'content_block_start':
            return None
        if chunk.type == 'content_block_delta':
            return ChatChunkResponse(content=chunk.delta.text)
        if chunk.type == 'message_start':
            return None
        if chunk.type == 'message_delta':
            return ChatChunkResponse(
                content=chunk.delta.text if hasattr(chunk.delta, 'text') else "",
            )
    return None


@Model.register(ANTHROPIC)
class AsyncAnthropicCompletionWrapper(Model):
    """
    Wrapper for Anthropic API which provides a simple interface for calling the
    messages.create method and parsing the response.
    """

    def __init__(
            self,
            model: str,
            max_tokens: int = 1_000,
            **model_kwargs: dict,
    ) -> None:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        if not max_tokens:
            max_tokens = 1_000
        self.model_parameters = {'max_tokens': max_tokens, **model_kwargs}
        # remove any None values
        self.model_parameters = {k: v for k, v in self.model_parameters.items() if v is not None}

    @classmethod
    def provider_name(cls) -> str:
        """Get the provider name of the model."""
        return ANTHROPIC

    @classmethod
    def primary_chat_model_names(cls) -> list[str]:
        """Get the primary model names (e.g. the models we would want to display to the user)."""
        return list(CHAT_MODEL_COST_PER_TOKEN_LATEST.keys())

    @classmethod
    def supported_chat_model_names(cls) -> list[str]:
        """Get all model names supported by the wrapper."""
        return list(CHAT_MODEL_COST_PER_TOKEN.keys())

    @classmethod
    def cost_per_token(cls, model_name: str, token_type: str) -> float:
        """Get the cost per token for the model."""
        return CHAT_MODEL_COST_PER_TOKEN[model_name][token_type]

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            else:
                anthropic_messages.append({'role': msg['role'], 'content': msg['content']})
        return system_content, anthropic_messages

    async def __call__(
    self,
    messages: list[dict],
    model: str | None = None,
    **model_kwargs: dict,
    ) -> AsyncGenerator[ChatChunkResponse | ChatStreamResponseSummary, None]:
        """
        Streams chat chunks and returns a final summary. Parameters passed here
        override those passed to the constructor.
        """
        model = model or self.model
        model_parameters = {**self.model_parameters, **model_kwargs}
        system_content, anthropic_messages = self._convert_messages(messages)
        api_params = {
            'model': model,
            'messages': anthropic_messages,
            'stream': True,
            **model_parameters,
        }
        if system_content:
            api_params['system'] = system_content
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0

        response = await self.client.messages.create(**api_params)
        async for chunk in response:
            if chunk.type == 'message_start':
                input_tokens = chunk.message.usage.input_tokens
            elif chunk.type == 'message_delta' and hasattr(chunk, 'usage'):
                output_tokens = chunk.usage.output_tokens
            parsed_chunk = _parse_completion_chunk(chunk)
            if parsed_chunk and parsed_chunk.content:
                yield parsed_chunk
        end_time = time.time()

        yield ChatStreamResponseSummary(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_input_cost=input_tokens * CHAT_MODEL_COST_PER_TOKEN[model]['input'],
            total_output_cost=output_tokens * CHAT_MODEL_COST_PER_TOKEN[model]['output'],
            duration_seconds=end_time - start_time,
        )
