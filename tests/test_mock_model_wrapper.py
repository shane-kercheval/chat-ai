"""Test the mock object to ensure it is working correctly."""
import pytest
from sik_llms import (
    create_client,
    Client,
    TextChunkEvent,
    TextResponse,
    Tool,
    Parameter,
)
from tests.conftest import MockAsyncOpenAICompletionWrapper, MockAsyncOpenAIFunctionWrapper


@pytest.mark.asyncio
async def test_mock_wrapper_model_instantiate() -> None:
    client = Client.instantiate(
        client_type="MockAsyncOpenAICompletionWrapper",
        model_name="Mock",
        mock_responses="My Custom Response",
    )
    assert isinstance(client, MockAsyncOpenAICompletionWrapper)
    assert client.model == "Mock"
    assert client.mock_responses == "My Custom Response"

    client = create_client(
        client_type="MockAsyncOpenAICompletionWrapper",
        model_name="Mock",
        mock_responses="My Custom Response",
    )
    assert isinstance(client, MockAsyncOpenAICompletionWrapper)
    assert client.model == "Mock"
    assert client.mock_responses == "My Custom Response"


@pytest.mark.asyncio
async def test_mock_wrapper_default_response() -> None:
    client = MockAsyncOpenAICompletionWrapper(model_name="gpt-4")
    messages = [{"role": "user", "content": "Hello"}]
    response_text = ""
    async for response in client.stream(messages):
        if isinstance(response, TextChunkEvent):
            response_text += response.content
    assert response_text.strip() == "This is a mock response."


@pytest.mark.asyncio
async def test_mock_wrapper_custom_response() -> None:
    client = MockAsyncOpenAICompletionWrapper(
        model_name="gpt-4",
        mock_responses="Custom mock response",
    )
    messages = [{"role": "user", "content": "Hello"}]
    response_text = ""
    responses = []
    async for response in client.stream(messages):
        responses.append(response)
        if isinstance(response, TextChunkEvent):
            response_text += response.content
    assert response_text.strip() == "Custom mock response"
    assert isinstance(responses[-1], TextResponse)


@pytest.mark.asyncio
async def test_mock_wrapper_multiple_responses() -> None:
    client = MockAsyncOpenAICompletionWrapper(
        model_name="gpt-4",
        mock_responses=[
            "Custom mock response 1",
            "Custom mock response 2",
        ],
    )
    messages = [{"role": "user", "content": "Hello"}]
    response_text = ""
    responses = []
    async for response in client.stream(messages):
        responses.append(response)
        if isinstance(response, TextChunkEvent):
            response_text += response.content
    assert response_text.strip() == "Custom mock response 1"
    assert isinstance(responses[-1], TextResponse)

    response_text = ""
    responses = []
    async for response in client.stream(messages):
        responses.append(response)
        if isinstance(response, TextChunkEvent):
            response_text += response.content
    assert response_text.strip() == "Custom mock response 2"
    assert isinstance(responses[-1], TextResponse)


@pytest.fixture
def test_function() -> Tool:
    return Tool(
        name="test_function",
        description="A test function",
        parameters=[
            Parameter(
                name="param1",
                param_type=str,
                required=True,
                description="A test parameter",
            ),
            Parameter(
                name="param2",
                param_type=int,
                required=False,
                description="Another test parameter",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_mock_function_wrapper_instantiate():
    wrapper = Client.instantiate(
        client_type="MockAsyncOpenAIFunctionWrapper",
        model_name="Mock",
        functions=[Tool(name="test_function", parameters=[])],
        mock_responses={'name': 'test_function', 'arguments': {'param1': 'value1', 'param2': 2}},
    )
    assert isinstance(wrapper, MockAsyncOpenAIFunctionWrapper)
    assert wrapper.model == "Mock"
    assert wrapper.mock_responses == {'name': 'test_function', 'arguments': {'param1': 'value1', 'param2': 2}}  # noqa: E501


@pytest.mark.asyncio
async def test_mock_function_wrapper(
        test_function: Tool,
    ) -> None:
    client = MockAsyncOpenAIFunctionWrapper(
        model_name="gpt-4",
        functions=[test_function],
        mock_responses={'name': 'test_function', 'arguments': {'param1': 'value1', 'param2': 2}},
    )
    messages = [{"role": "user", "content": "Test message"}]
    response = await client.stream(messages)
    assert response.tool_prediction.name == "test_function"
    assert response.tool_prediction.arguments == {"param1": "value1", "param2": 2}
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.input_cost > 0
    assert response.output_cost > 0


@pytest.mark.asyncio
async def test_mock_function_wrapper__multiple_responses(
        test_function: Tool,
    ) -> None:
    client = MockAsyncOpenAIFunctionWrapper(
        model_name="gpt-4",
        functions=[test_function],
        mock_responses=[
            {'name': 'test_function_1', 'arguments': {'param1': 'value1', 'param2': 2}},
            {'name': 'test_function_2', 'arguments': {'param3': 'value2', 'param4': 3}},
        ],
    )
    messages = [{"role": "user", "content": "Test message"}]
    response = await client.stream(messages)
    assert response.tool_prediction.name == "test_function_1"
    assert response.tool_prediction.arguments == {"param1": "value1", "param2": 2}
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.input_cost > 0
    assert response.output_cost > 0

    response = await client.stream(messages)
    assert response.tool_prediction.name == "test_function_2"
    assert response.tool_prediction.arguments == {"param3": "value2", "param4": 3}
