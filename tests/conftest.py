"""Fixtures and helper functions for testing."""
from collections.abc import AsyncGenerator
import os
import tempfile
import time
import pytest

from sik_llms import (
    Function,
    FunctionCallResponse,
    FunctionCallResult,
    Client,
    ChatChunkResponse,
    ChatResponseSummary,
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

   async def run_async(
        self,
        messages: list[dict[str, str]],
        model_name: str | None = None,  # noqa: ARG002
        **model_kwargs: object,  # noqa: ARG002
    ) -> AsyncGenerator[ChatChunkResponse | ChatResponseSummary, None]:
        """Simulate streaming response with mock chunks and summary."""
        start_time = time.time()
        chunks: list[ChatChunkResponse] = []

        if isinstance(self.mock_responses, list):
            next_response = self.mock_responses.pop(0)
        else:
            next_response = self.mock_responses

        for word in next_response.split():
            chunk = ChatChunkResponse(
                content=word + ' ',
                logprob=-1.0,
            )
            chunks.append(chunk)
            yield chunk

        end_time = time.time()

        input_tokens = (sum(len(str(m)) for m in messages) // 4) + 1
        output_tokens = (len(next_response) // 4) + 1

        yield ChatResponseSummary(
            content=next_response,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_input_cost=input_tokens * (0.03 / 1000),
            total_output_cost=output_tokens * (0.06 / 1000),
            duration_seconds=end_time - start_time,
        )


@Client.register('MockAsyncOpenAIFunctionWrapper')
class MockAsyncOpenAIFunctionWrapper(Client):
    """Mock wrapper that simulates OpenAI API function calling."""

    def __init__(
        self,
        model_name: str,
        server_url: str | None = None,
        functions: list[Function] | None = None,
        **model_kwargs: object,
    ) -> None:
        self.model = model_name
        self.server_url = server_url
        self.functions = functions or []
        if 'mock_responses' not in model_kwargs:
            raise ValueError("mock_responses is required in model_kwargs")
        self.mock_responses = model_kwargs.pop('mock_responses')
        self.model_kwargs = model_kwargs or {}

    async def run_async(
        self,
        messages: list[dict[str, str]],
        functions: list[Function] | None = None,  # noqa: ARG002
        tool_choice: str = 'required',  # noqa: ARG002
        model_name: str | None = None,  # noqa: ARG002
        **model_kwargs: object,  # noqa: ARG002
    ) -> FunctionCallResponse:
        """Mock function calling with simulated response."""
        if isinstance(self.mock_responses, list):
            next_response = self.mock_responses.pop(0)
        else:
            next_response = self.mock_responses
        function_call = FunctionCallResult(
            name=next_response['name'],
            arguments=next_response['arguments'],
            call_id='mock_call_123',
        )
        # Simulate token counts and costs
        input_tokens = sum(len(str(m)) for m in messages) // 4
        output_tokens = len(str(next_response['arguments'])) // 4
        return FunctionCallResponse(
            function_call=function_call,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_tokens * (0.03 / 1000),
            output_cost=output_tokens * (0.06 / 1000),
        )
