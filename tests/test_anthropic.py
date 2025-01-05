"""Test the Anthropic Wrapper."""
import asyncio
import os
import pytest
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from server.models.anthropic import (
    AsyncAnthropicCompletionWrapper,
    ChatChunkResponse,
    ChatStreamResponseSummary,
)
from server.models import system_message, user_message

load_dotenv()

ANTHROPIC_TEST_MODEL = 'claude-3-5-haiku-latest'

class TestClassHelperFucntions:  # noqa: D101

    def test_provider_name(self):
        assert AsyncAnthropicCompletionWrapper.provider_name() == 'Anthropic'

    def test_primary_chat_model_names(self):
        primary_names = AsyncAnthropicCompletionWrapper.primary_chat_model_names()
        all_names = AsyncAnthropicCompletionWrapper.supported_chat_model_names()
        assert ANTHROPIC_TEST_MODEL in AsyncAnthropicCompletionWrapper.primary_chat_model_names()
        assert all(name in all_names for name in primary_names)

    def test_supported_chat_model_names(self):
        supported_names = AsyncAnthropicCompletionWrapper.supported_chat_model_names()
        assert ANTHROPIC_TEST_MODEL in supported_names

    def test_cost_per_token(self):
        cost = AsyncAnthropicCompletionWrapper.cost_per_token(ANTHROPIC_TEST_MODEL, 'input')
        assert isinstance(cost, float)
        assert cost > 0

        cost = AsyncAnthropicCompletionWrapper.cost_per_token(ANTHROPIC_TEST_MODEL, 'output')
        assert isinstance(cost, float)
        assert cost > 0


@pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")
@pytest.mark.asyncio
async def test_async_anthropic_completion_wrapper_call():
    """Test the Anthropic wrapper with multiple concurrent calls."""
    # Create an instance of the wrapper
    model = AsyncAnthropicCompletionWrapper(
        client=AsyncAnthropic(),
        model=ANTHROPIC_TEST_MODEL,
        max_tokens=100,
    )

    assert AsyncAnthropicCompletionWrapper.provider_name() == 'Anthropic'
    assert ANTHROPIC_TEST_MODEL in AsyncAnthropicCompletionWrapper.primary_chat_model_names()
    assert ANTHROPIC_TEST_MODEL in AsyncAnthropicCompletionWrapper.supported_chat_model_names()

    messages = [
        system_message("You are a helpful assistant."),
        user_message("What is the capital of France?"),
    ]

    async def run_model():  # noqa: ANN202
        chunks = []
        summary = None
        try:
            async for response in model(messages=messages):
                if isinstance(response, ChatChunkResponse):
                    chunks.append(response)
                elif isinstance(response, ChatStreamResponseSummary):
                    summary = response
            return chunks, summary
        except Exception:
            return [], None

    results = await asyncio.gather(*(run_model() for _ in range(10)))
    passed_tests = []

    for chunks, summary in results:
        response = ''.join([chunk.content for chunk in chunks])
        passed_tests.append(
            'Paris' in response
            and isinstance(summary, ChatStreamResponseSummary)
            and summary.total_input_tokens > 0
            and summary.total_output_tokens > 0
            and summary.total_input_cost > 0
            and summary.total_output_cost > 0
            and summary.duration_seconds > 0,
        )

    assert sum(passed_tests) / len(passed_tests) >= 0.9, (
        f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."
    )
