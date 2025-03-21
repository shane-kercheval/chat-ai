"""Fixtures and helper functions for testing."""
from collections.abc import AsyncGenerator
import os
import tempfile
import time
from pydantic import BaseModel
import pytest
from sik_llms import (
    Tool,
    ToolPredictionResponse,
    ToolPrediction,
    Client,
    TextChunkEvent,
    TextResponse,
    StructuredOutputResponse,
)


SKIP_CI = pytest.mark.skipif(
    os.getenv('SKIP_CI_TESTS') == 'true',
    reason='Skipping test in CI environment',
)

def create_temp_file(content: str, prefix: str | None = None, suffix: str | None = None) -> str:
    """Create a temporary file with content."""
    f = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode='w',
        delete=False,
        prefix=prefix,
        suffix=suffix,
    )
    f.write(content)
    f.close()
    return f.name



@Client.register('MockAsyncOpenAICompletionWrapper')
class MockAsyncOpenAICompletionWrapper(Client):
   """Mock wrapper that simulates OpenAI API streaming responses."""

   def __init__(
       self,
       model_name: str,
       server_url: str | None = None,
       **model_kwargs: object,
   ) -> None:
       self.model = model_name
       self.server_url = server_url
       self.mock_responses = model_kwargs.pop('mock_responses', 'This is a mock response.')
       self.model_parameters = model_kwargs

   async def stream(
        self,
        messages: list[dict[str, str]],
        model_name: str | None = None,  # noqa: ARG002
        **model_kwargs: object,  # noqa: ARG002
    ) -> AsyncGenerator[TextChunkEvent | TextResponse, None]:
        """Simulate streaming response with mock chunks and summary."""
        start_time = time.time()
        chunks: list[TextChunkEvent] = []

        if isinstance(self.mock_responses, list):
            next_response = self.mock_responses.pop(0)
        else:
            next_response = self.mock_responses

        for word in next_response.split():
            chunk = TextChunkEvent(
                content=word + ' ',
                logprob=-1.0,
            )
            chunks.append(chunk)
            yield chunk

        end_time = time.time()

        input_tokens = (sum(len(str(m)) for m in messages) // 4) + 1
        output_tokens = (len(next_response) // 4) + 1

        yield TextResponse(
            response=next_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * (0.03 / 1000),
            output_cost=output_tokens * (0.06 / 1000),
            duration_seconds=end_time - start_time,
        )


@Client.register('MockAsyncOpenAIFunctionWrapper')
class MockAsyncOpenAIFunctionWrapper(Client):
    """Mock wrapper that simulates OpenAI API function calling."""

    def __init__(
        self,
        model_name: str,
        server_url: str | None = None,
        tools: list[Tool] | None = None,
        **model_kwargs: object,
    ) -> None:
        self.model = model_name
        self.server_url = server_url
        self.tools = tools or []
        if 'mock_responses' not in model_kwargs:
            raise ValueError("mock_responses is required in model_kwargs")
        self.mock_responses = model_kwargs.pop('mock_responses')
        self.model_kwargs = model_kwargs or {}

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: list[Tool] | None = None,  # noqa: ARG002
        tool_choice: str = 'required',  # noqa: ARG002
        model_name: str | None = None,  # noqa: ARG002
        **model_kwargs: object,  # noqa: ARG002
    ) -> ToolPredictionResponse:
        """Mock function calling with simulated response."""
        if isinstance(self.mock_responses, list):
            next_response = self.mock_responses.pop(0)
        else:
            next_response = self.mock_responses
        tool_prediction = ToolPrediction(
            name=next_response['name'],
            arguments=next_response['arguments'],
            call_id='mock_call_123',
        )
        # Simulate token counts and costs
        input_tokens = sum(len(str(m)) for m in messages) // 4
        output_tokens = len(str(next_response['arguments'])) // 4
        return ToolPredictionResponse(
            tool_prediction=tool_prediction,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * (0.03 / 1000),
            output_cost=output_tokens * (0.06 / 1000),
            duration_seconds=0.1,
        )


@Client.register('MockAsyncOpenAIStructuredOutput')
class MockAsyncOpenAIStructuredOutput(Client):
    """Mock wrapper that simulates OpenAI API structured output responses."""

    def __init__(
        self,
        model_name: str,
        server_url: str | None = None,
        response_format: BaseModel | None = None,
        **model_kwargs: object,
    ) -> None:
        self.model = model_name
        self.server_url = server_url
        self.response_format = response_format
        if 'mock_responses' not in model_kwargs:
            raise ValueError("mock_responses is required in model_kwargs")
        self.mock_responses = model_kwargs.pop('mock_responses')
        self.model_kwargs = model_kwargs or {}

    async def stream(
        self,
        messages: list[dict[str, str]],  # noqa: ARG002
    ) -> AsyncGenerator[TextResponse, None]:
        """Mock structured output response."""
        # Get mock response
        if isinstance(self.mock_responses, list):
            next_response = self.mock_responses.pop(0)
        else:
            next_response = self.mock_responses
        # Simulate token counts
        input_tokens = 3
        output_tokens = 3
        # Return structured output response
        yield StructuredOutputResponse(
            parsed=next_response.get('parsed', next_response),
            refusal=next_response.get('refusal', None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * (0.03 / 1000),
            output_cost=output_tokens * (0.06 / 1000),
            duration_seconds=0.1,
        )
