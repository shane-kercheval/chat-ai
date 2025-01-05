"""Test the OpenAI Wrapper."""
import asyncio
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI
from server.models import system_message, user_message
from server.models.openai import (
    AsyncOpenAICompletionWrapper,
    ChatChunkResponse,
    ChatStreamResponseSummary,
    AsyncOpenAIFunctionWrapper,
    Function,
    Parameter,
    FunctionCallResponse,
    FunctionCallResult,
)

load_dotenv()

OPENAI_TEST_MODEL = 'gpt-4o-mini'

class TestClassHelperFucntions:  # noqa: D101

    def test_provider_name(self):
        assert AsyncOpenAICompletionWrapper.provider_name() == 'OpenAI'

    def test_primary_chat_model_names(self):
        primary_names = AsyncOpenAICompletionWrapper.primary_chat_model_names()
        all_names = AsyncOpenAICompletionWrapper.supported_chat_model_names()
        assert OPENAI_TEST_MODEL in AsyncOpenAICompletionWrapper.primary_chat_model_names()
        assert all(name in all_names for name in primary_names)

    def test_supported_chat_model_names(self):
        supported_names = AsyncOpenAICompletionWrapper.supported_chat_model_names()
        assert OPENAI_TEST_MODEL in supported_names

    def test_cost_per_token(self):
        cost = AsyncOpenAICompletionWrapper.cost_per_token(OPENAI_TEST_MODEL, 'input')
        assert isinstance(cost, float)
        assert cost > 0

        cost = AsyncOpenAICompletionWrapper.cost_per_token(OPENAI_TEST_MODEL, 'output')
        assert isinstance(cost, float)
        assert cost > 0


@pytest.mark.asyncio
class TestOpenAICompletionWrapper:
    """Test the OpenAI Completion Wrapper."""

    async def test_async_openai_completion_wrapper_call(self):
        # Create an instance of the wrapper
        model = AsyncOpenAICompletionWrapper(
            client=AsyncOpenAI(),
            model=OPENAI_TEST_MODEL,
        )

        assert AsyncOpenAICompletionWrapper.provider_name() == 'OpenAI'
        assert OPENAI_TEST_MODEL in AsyncOpenAICompletionWrapper.primary_chat_model_names()
        assert OPENAI_TEST_MODEL in AsyncOpenAICompletionWrapper.supported_chat_model_names()

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
            assert all(chunk.logprob is not None for chunk in chunks)
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
        assert sum(passed_tests) / len(passed_tests) >= 0.9, f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."  # noqa: E501



@pytest.mark.asyncio
class TestOpenAIFunctions:
    """Test the OpenAI Function Wrapper."""

    @pytest.fixture
    def simple_weather_function(self):
        """Create a simple weather function with one required parameter."""
        return Function(
            name="get_weather",
            description="Get the weather for a location.",
            parameters=[
                Parameter(
                    name="location",
                    type="string",
                    required=True,
                    description="The city and country for weather info.",
                ),
            ],
        )

    @pytest.fixture
    def complex_weather_function(self):
        """Create a weather function with multiple parameters including an enum."""
        return Function(
            name="get_detailed_weather",
            description="Get detailed weather information for a location.",
            parameters=[
                Parameter(
                    name="location",
                    type="string",
                    required=True,
                    description="The city and country",
                ),
                Parameter(
                    name="unit",
                    type="string",
                    required=False,
                    description="Temperature unit",
                    enum=["celsius", "fahrenheit"],
                ),
                Parameter(
                    name="include_forecast",
                    type="boolean",
                    required=False,
                    description="Whether to include forecast data",
                ),
            ],
        )

    @pytest.fixture
    def restaurant_function(self):
        """Create a restaurant search function with multiple parameters."""
        return Function(
            name="search_restaurants",
            description="Search for restaurants in a location.",
            parameters=[
                Parameter(
                    name="location",
                    type="string",
                    required=True,
                    description="The city to search in",
                ),
                Parameter(
                    name="cuisine",
                    type="string",
                    required=False,
                    description="Type of cuisine",
                    enum=["italian", "chinese", "mexican", "indian"],
                ),
                Parameter(
                    name="price_range",
                    type="string",
                    required=False,
                    description="Price range",
                    enum=["$", "$$", "$$$", "$$$$"],
                ),
                Parameter(
                    name="open_now",
                    type="boolean",
                    required=False,
                    description="Filter for currently open restaurants",
                ),
            ],
        )

    async def test_single_function_single_parameter(self, simple_weather_function: Function):
        """Test calling a simple function with one required parameter."""
        wrapper = AsyncOpenAIFunctionWrapper(
            client=AsyncOpenAI(),
            model=OPENAI_TEST_MODEL,
            functions=[simple_weather_function],
        )
        response = await wrapper(
            messages=[
                {"role": "user", "content": "What's the weather like in Paris?"},
            ],
        )
        assert isinstance(response, FunctionCallResponse)
        assert isinstance(response.function_call, FunctionCallResult)
        assert response.function_call.name == "get_weather"
        assert "location" in response.function_call.arguments
        assert "Paris" in response.function_call.arguments["location"]
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.input_cost > 0
        assert response.output_cost > 0

    @pytest.mark.asyncio
    async def test_single_function_multiple_parameters(self, complex_weather_function: Function):
        """Test calling a function with multiple parameters including optional ones."""
        wrapper = AsyncOpenAIFunctionWrapper(
            client=AsyncOpenAI(),
            model=OPENAI_TEST_MODEL,
            functions=[complex_weather_function],
        )
        response = await wrapper(
            messages=[
                {"role": "user", "content": "What's the weather like in Tokyo in Celsius with forecast?"},  # noqa: E501
            ],
        )
        assert response.function_call.name == "get_detailed_weather"
        args = response.function_call.arguments
        assert "Tokyo" in args["location"]
        assert args.get("unit") in ["celsius", "fahrenheit"]
        assert args.get("include_forecast") is not None

    @pytest.mark.asyncio
    async def test_multiple_functions(self, simple_weather_function: Function, restaurant_function: Function):  # noqa: E501
        """Test providing multiple functions to the model."""
        wrapper = AsyncOpenAIFunctionWrapper(
            client=AsyncOpenAI(),
            model=OPENAI_TEST_MODEL,
            functions=[simple_weather_function, restaurant_function],
        )
        # Weather query
        weather_response = await wrapper(
            messages=[
                {"role": "user", "content": "What's the weather like in London?"},
            ],
        )
        assert weather_response.function_call.name == "get_weather"
        assert "London" in weather_response.function_call.arguments["location"]
        # Restaurant query
        restaurant_response = await wrapper(
            messages=[
                {"role": "user", "content": "Find me an expensive Italian restaurant in New York"},
            ],
        )
        assert restaurant_response.function_call.name == "search_restaurants"
        args = restaurant_response.function_call.arguments
        assert "New York" in args["location"]
        assert args.get("cuisine") == "italian"
        assert args.get("price_range") in ["$$", "$$$", "$$$$"]

    @pytest.mark.asyncio
    async def test_enum_parameters(self, restaurant_function: Function):
        """Test handling of enum parameters."""
        wrapper = AsyncOpenAIFunctionWrapper(
            client=AsyncOpenAI(),
            model=OPENAI_TEST_MODEL,
            functions=[restaurant_function],
        )
        test_cases = [
            (
                "Find me a cheap Chinese restaurant in Boston",
                {"cuisine": "chinese", "price_range": "$"},
            ),
            (
                "I want Mexican food in Chicago at a moderate price",
                {"cuisine": "mexican"},
            ),
            (
                "Find an Indian restaurant in Seattle",
                {"cuisine": "indian"},
            ),
        ]

        for prompt, expected_args in test_cases:
            response = await wrapper(
                messages=[{"role": "user", "content": prompt}],
            )
            args = response.function_call.arguments
            for key, value in expected_args.items():
                assert args.get(key) == value

    @pytest.mark.asyncio
    async def test_concurrent_function_calls(self, simple_weather_function: Function):
        """Test multiple concurrent function calls."""
        wrapper = AsyncOpenAIFunctionWrapper(
            client=AsyncOpenAI(),
            model=OPENAI_TEST_MODEL,
            functions=[simple_weather_function],
        )

        cities = ["Paris", "London", "Tokyo", "New York", "Sydney"]
        messages = [
            [{"role": "user", "content": f"What's the weather like in {city}?"}]
            for city in cities
        ]

        responses = await asyncio.gather(*(
            wrapper(messages=msg) for msg in messages
        ))

        for i, response in enumerate(responses):
            assert response.function_call.name == "get_weather"
            assert cities[i] in response.function_call.arguments["location"]
            assert response.input_tokens > 0
            assert response.output_tokens > 0

    @pytest.mark.asyncio
    async def test_function_override(self, simple_weather_function: Function, complex_weather_function: Function):  # noqa: E501
        """Test overriding functions during the call."""
        wrapper = AsyncOpenAIFunctionWrapper(
            client=AsyncOpenAI(),
            model=OPENAI_TEST_MODEL,
            functions=[simple_weather_function],
        )

        # Override with complex weather function
        response = await wrapper(
            messages=[
                {"role": "user", "content": "What's the weather like in Berlin in Fahrenheit?"},
            ],
            functions=[complex_weather_function],
        )

        assert response.function_call.name == "get_detailed_weather"
        args = response.function_call.arguments
        assert "Berlin" in args["location"]
        assert args.get("unit") in ["celsius", "fahrenheit"]
