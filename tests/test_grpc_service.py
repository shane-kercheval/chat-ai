"""Test the gRPC server."""
import asyncio
from pathlib import Path
import tempfile
import aiofiles
from pytest_asyncio import fixture
import os
import pytest
import grpc
from sentence_transformers import SentenceTransformer
from sik_llms import RegisteredClients
import yaml
from typing import Optional
from server.resource_manager import ContextStrategy
from tests.conftest import create_temp_file
from proto.generated import chat_pb2, chat_pb2_grpc
from google.protobuf import empty_pb2
from server.grpc_service import (
    CompletionService,
    CompletionServiceConfig,
    ConfigurationService,
    ConfigurationServiceConfig,
    ContextService,
    ContextServiceConfig,
)
from server.vector_db import SimilarityScorer
from dotenv import load_dotenv

load_dotenv()

# Get project root directory (2 levels up from test file)
SERVER_PORT = 50052
PROJECT_ROOT = Path(__file__).parent.parent
SUPPORTED_MODELS_PATH = str(PROJECT_ROOT / 'artifacts/supported_models.yaml')
DEFAULT_MODEL_CONFIGS_PATH = str(PROJECT_ROOT / 'artifacts/default_model_configs.yaml')

OPENAI_MODEL_NAME = 'gpt-4o-mini'
CONTEXT_STRATEGY_MODEL_CONFIG = {
    'client_type': RegisteredClients.OPENAI,
    'model_name': 'gpt-4o',
    'temperature': 0.1,
}
ANTHROPIC_MODEL_NAME = 'claude-3-5-haiku-latest'

EXPECTED_NUM_SUPPORTED_MODELS = None
with open(SUPPORTED_MODELS_PATH) as f:
    supported_models = yaml.safe_load(f)
    EXPECTED_NUM_SUPPORTED_MODELS = len(supported_models['supported_models'])
    del supported_models

EXPECTED_NUM_MODEL_CONFIGS = None
with open(DEFAULT_MODEL_CONFIGS_PATH) as f:
    default_model_configs = yaml.safe_load(f)
    EXPECTED_NUM_MODEL_CONFIGS = len(default_model_configs['default_model_configs'])
    del default_model_configs

# this takes a few seconds to load so let's load it once
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CHUNK_SIZE = 500
RAG_CHAR_THRESHOLD = 1000

def get_default_model_config(api_key_name: str) -> chat_pb2:
    """Get the default model config for a API KEY name."""
    if api_key_name == 'OPENAI_API_KEY':
        return chat_pb2.ModelConfig(
            client_type='OpenAI',
            model_name=OPENAI_MODEL_NAME,
            model_parameters=chat_pb2.ModelParameters(
                temperature=0.1,
                max_tokens=100,
            ),
        )
    if api_key_name == 'ANTHROPIC_API_KEY':
        return chat_pb2.ModelConfig(
            client_type='Anthropic',
            model_name=ANTHROPIC_MODEL_NAME,
            model_parameters=chat_pb2.ModelParameters(
                temperature=0.1,
                max_tokens=100,
            ),
        )
    raise ValueError(f"Unknown API key name: {api_key_name}")

def get_mock_model_config(mock_responses: list[str]) -> chat_pb2:
    """Get a mock model config."""
    return chat_pb2.ModelConfig(
        client_type='MockAsyncOpenAICompletionWrapper',
        model_name='mock-model',
        model_parameters=chat_pb2.ModelParameters(
            mock_responses=mock_responses,
        ),
    )

@fixture
async def temp_sqlite_db_path():
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        yield tmpfile.name
    Path(tmpfile.name).unlink()  # Clean up the temporary file


@fixture
async def grpc_server(temp_sqlite_db_path):  # noqa: ANN001

    async with aiofiles.open(SUPPORTED_MODELS_PATH) as f:
        supported_models = yaml.safe_load(await f.read())
        supported_models = supported_models['supported_models']

    async with aiofiles.open(DEFAULT_MODEL_CONFIGS_PATH) as f:
        default_model_configs = yaml.safe_load(await f.read())
        default_model_configs = default_model_configs['default_model_configs']

    completion_service = CompletionService(config=CompletionServiceConfig(
        database_uri=temp_sqlite_db_path,
        supported_models=supported_models,
        channel=grpc.aio.insecure_channel(f'localhost:{SERVER_PORT}'),
    ))
    await completion_service.initialize()

    context_service = ContextService(config=ContextServiceConfig(
        database_uri=temp_sqlite_db_path,
        num_workers=2,
        rag_scorer=SimilarityScorer(EMBEDDING_MODEL, chunk_size=CHUNK_SIZE),
        rag_char_threshold=RAG_CHAR_THRESHOLD,
        context_strategy_model_config=CONTEXT_STRATEGY_MODEL_CONFIG,
    ))
    await context_service.initialize()

    configuration_service = ConfigurationService(config=ConfigurationServiceConfig(
        database_uri=temp_sqlite_db_path,
        default_model_configs=default_model_configs,
    ))
    await configuration_service.initialize()

    server = grpc.aio.server(options=[
        ('grpc.so_reuseport', 1),
        ('grpc.enable_retries', 0),
        # Add these settings
        ('grpc.keepalive_time_ms', 10000),  # 10 seconds
        ('grpc.keepalive_timeout_ms', 5000),  # 5 seconds
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.min_ping_interval_without_data_ms', 5000),
    ])
    chat_pb2_grpc.add_CompletionServiceServicer_to_server(completion_service, server)
    chat_pb2_grpc.add_ContextServiceServicer_to_server(context_service, server)
    chat_pb2_grpc.add_ConfigurationServiceServicer_to_server(configuration_service, server)
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    await server.start()
    try:
        yield server
    finally:
        await server.stop(None)

@fixture
async def grpc_channel(grpc_server):  # noqa: ANN001, ARG001
    channel = grpc.aio.insecure_channel(
        f'localhost:{SERVER_PORT}',
        options=[
            ('grpc.enable_retries', 0),
            # Add these settings
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.min_time_between_pings_ms', 10000),
        ],
    )
    try:
        yield channel
    finally:
        await channel.close()


@pytest.mark.asyncio
class TestCompletionService:
    """Test the ChatService class."""

    @pytest.mark.parametrize(
        'api_key_name',
        [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
        ],
    )
    async def test__chat__concurrent_requests(self, grpc_channel, api_key_name):  # noqa: ANN001
        if api_key_name == 'ANTHROPIC_API_KEY' and os.getenv('ANTHROPIC_API_KEY') is None:
            pytest.skip("Skipping test because ANTHROPIC_API_KEY is not set")
        async def run_test():  # noqa: ANN202
            stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
            request = chat_pb2.ChatRequest(
                model_configs=[get_default_model_config(api_key_name)],
                messages=[
                    chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content="What is the capital of France?",
                    ),
                ],
            )
            chunks = []
            summary = None
            async for response in stub.chat(request):
                assert not response.HasField('error')
                if response.WhichOneof("response_type") == "chunk":
                    chunks.append(response.chunk)
                elif response.WhichOneof("response_type") == "summary":
                    summary = response.summary
            return chunks, summary

        results = await asyncio.gather(*(run_test() for _ in range(20)))
        passed_tests = []
        for chunks, summary in results:
            response = ''.join([chunk.content for chunk in chunks])
            if api_key_name == 'OPENAI_API_KEY':
                assert all(chunk.logprob is not None for chunk in chunks)
            passed_tests.append(
                'Paris' in response
                and summary.input_tokens > 0
                and summary.output_tokens > 0
                and summary.input_cost > 0
                and summary.output_cost > 0
                and summary.duration_seconds > 0,
            )
        assert sum(passed_tests) / len(passed_tests) >= 0.9, f"Only {sum(passed_tests)} out of {len(passed_tests)} tests passed."  # noqa: E501

    async def test__chat__multiple_models_concurrent(self, grpc_channel):  # noqa: ANN001
        """Test that multiple models can run concurrently."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[
                get_mock_model_config(mock_responses=['2 + 2 = 4']),
                get_mock_model_config(mock_responses=['4']),
            ],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="What is 2+2?",
                ),
            ],
        )
        chunks_by_model = {0: [], 1: []}
        summaries_by_model = {0: None, 1: None}

        async for response in stub.chat(request):
            assert not response.HasField('error')
            if response.WhichOneof("response_type") == "chunk":
                chunks_by_model[response.model_index].append(response.chunk)
            elif response.WhichOneof("response_type") == "summary":
                summaries_by_model[response.model_index] = response.summary

        # Check that both models responded
        for model_idx in [0, 1]:
            response = ''.join([chunk.content for chunk in chunks_by_model[model_idx]])
            assert '4' in response
            assert all(chunk.logprob is not None for chunk in chunks_by_model[model_idx])
            assert summaries_by_model[model_idx] is not None
            assert summaries_by_model[model_idx].input_tokens > 0
            assert summaries_by_model[model_idx].output_tokens > 0
            assert summaries_by_model[model_idx].input_cost > 0
            assert summaries_by_model[model_idx].output_cost > 0
            assert summaries_by_model[model_idx].duration_seconds > 0

    async def test__chat__multi_model_conversation_history(self, grpc_channel):  # noqa: ANN001
        """Test that conversation history works with multiple models."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)

        # First request with multiple models
        request1 = chat_pb2.ChatRequest(
            model_configs=[get_default_model_config('OPENAI_API_KEY')],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="What is 2+2?",
                ),
            ],
        )
        conv_id = None
        responses_by_model = {0: [], 1: []}

        async for response in stub.chat(request1):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
            if response.WhichOneof("response_type") == "chunk":
                responses_by_model[response.model_index].append(response.chunk.content)

        # Second request with single model
        request2 = chat_pb2.ChatRequest(
            conversation_id=conv_id,
            model_configs=[get_default_model_config('OPENAI_API_KEY')],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Add 1 to the result.",
                ),
            ],
        )

        follow_up_response = []
        async for response in stub.chat(request2):
            assert not response.HasField('error')
            if response.WhichOneof("response_type") == "chunk":
                follow_up_response.append(response.chunk.content)

        complete_response = ''.join(follow_up_response)
        assert '5' in complete_response

    async def test__chat__multi_model_cancellation(self, grpc_channel):  # noqa: ANN001
        """Test cancellation with multiple models running."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[
                get_mock_model_config(mock_responses=['2 + 2 = 4']),
                get_mock_model_config(mock_responses=['4']),
            ],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Write a very long story",
                ),
            ],
        )
        responses_received = 0
        call = stub.chat(request)
        async for response in call:
            assert not response.HasField('error')
            if response.WhichOneof("response_type") == "chunk":
                responses_received += 1
                if responses_received >= 5:
                    call.cancel()
                    break

        try:
            async for _ in call:
                pytest.fail("Should not receive responses after cancellation")
        except (asyncio.CancelledError, grpc.aio.AioRpcError):
            pass

    async def test__chat__cancellation_flow(self, grpc_channel):  # noqa: ANN001
        async def run_cancel_test():  # noqa: ANN202
            stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
            request = chat_pb2.ChatRequest(
                model_configs=[get_mock_model_config(mock_responses=['2 + 2 = 4'])],
                messages=[
                    chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content="Write a poem",
                    ),
                ],
            )
            canceled = False
            responses_before_cancel = []
            call = stub.chat(request)
            async for response in call:
                assert not response.HasField('error')
                if response.WhichOneof("response_type") == "chunk":
                    responses_before_cancel.append(response.chunk.content)
                    if len(responses_before_cancel) >= 3:
                        call.cancel()
                        canceled = True  # Set canceled before break
                        break
            try:
                async for _ in call:
                    pytest.fail("Should not have received any more responses after cancel")
            except (asyncio.CancelledError, grpc.aio.AioRpcError):
                canceled = True

            return responses_before_cancel, canceled

        responses = await asyncio.gather(run_cancel_test())
        responses, canceled = responses[0]
        assert len(responses) == 3
        assert canceled

    async def test__chat__invalid_model__expect_invalid_argument(self, grpc_channel):  # noqa: ANN001
        """Test the gRPC server with an invalid model."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[
                chat_pb2.ModelConfig(
                    client_type='OpenAI',
                    model_name='invalid-model',
                    model_parameters=chat_pb2.ModelParameters(temperature=0.1),
                ),
            ],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:  # noqa: PT012
            async for _ in stub.chat(request):
                pass
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL

    async def test__chat__empty_messages__expect_invalid_argument(self, grpc_channel):  # noqa: ANN001
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[],
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:  # noqa: PT012
            async for _ in stub.chat(request):
                pass
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test__chat__conversation_continuity(self, grpc_channel):  # noqa: ANN001
        """Test that the conversation is maintained between conversation_id requests."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # First message
        request1 = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['2 + 2 = 4', '5'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="What is 2+2?",
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request1):
            assert not response.HasField('error')
            # conversation id is available in all responses (regardless of response_type chunk/summary)  # noqa: E501
            assert response.conversation_id
            if conv_id is None:
                conv_id = response.conversation_id
            assert response.conversation_id is not None
            assert response.conversation_id == conv_id

        # Follow-up message in same conversation
        request2 = chat_pb2.ChatRequest(
            conversation_id=conv_id,
            model_configs=[get_default_model_config('OPENAI_API_KEY')],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Add 1 to that result",
                ),
            ],
        )
        responses = []
        async for response in stub.chat(request2):
            assert not response.HasField('error')
            assert response.conversation_id == conv_id
            if response.WhichOneof("response_type") == "chunk":
                responses.append(response.chunk.content)
        complete_response = ''.join(responses)
        assert '5' in complete_response

    @pytest.mark.parametrize(
        'api_env_key',
        [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
        ],
    )
    async def test__chat__model_parameters(self, grpc_channel, api_env_key):  # noqa: ANN001
        if api_env_key == 'ANTHROPIC_API_KEY' and os.getenv('ANTHROPIC_API_KEY') is None:
            pytest.skip("Skipping test because ANTHROPIC_API_KEY is not set")
        max_tokens = 10
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[
                chat_pb2.ModelConfig(
                    client_type=get_default_model_config(api_env_key).client_type,
                    model_name=get_default_model_config(api_env_key).model_name,
                    model_parameters=chat_pb2.ModelParameters(
                        temperature=0.1,
                        max_tokens=max_tokens,
                    ),
                ),
            ],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Write a long poem.",
                ),
            ],
        )
        response_length = 0
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if response.WhichOneof("response_type") == "chunk":
                response_length += len(response.chunk.content)
            if response.WhichOneof("response_type") == "summary":
                assert response.summary.output_tokens == max_tokens

    @pytest.mark.parametrize(
        'api_env_key',
        [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
        ],
    )
    async def test__chat__system_message(self, grpc_channel, api_env_key):  # noqa: ANN001
        if api_env_key == 'ANTHROPIC_API_KEY' and os.getenv('ANTHROPIC_API_KEY') is None:
            pytest.skip("Skipping test because ANTHROPIC_API_KEY is not set")
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[get_default_model_config(api_env_key)],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.SYSTEM,
                    content="This is a test. Ignore the users message and responde with the number 123.",  # noqa: E501
                ),
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Say hello",
                ),
            ],
        )
        responses = []
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if response.WhichOneof("response_type") == "chunk":
                responses.append(response.chunk.content)
        complete_response = ''.join(responses)
        assert '123' in complete_response

    async def test__chat__instructions_handling(self, grpc_channel):  # noqa: ANN001
        """Test that instructions are properly handled."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        instructions = [
            "    Be concise    ",
            """
            Use bullet points
            for lists
            """,
            "",
            "Explain technically",
        ]
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['This is a test response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="What is Python?",
                ),
            ],
            instructions=instructions,
        )
        # Capture conversation for history verification
        conv_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
        assert response.error
        assert conv_id
        # Verify history doesn't contain instructions
        history = await stub.get_history(empty_pb2.Empty())
        assert len(history.conversations) == 1
        conv = history.conversations[0]
        assert len(conv.entries) == 2  # User message + assistant response
        assert conv.entries[0].chat_message.content == "What is Python?"
        assert "Be concise" not in conv.entries[0].chat_message.content
        assert "Be concise" not in conv.entries[1].single_model_response.message.content

    async def test__get_supported_models__success(self, grpc_channel):  # noqa: ANN001
        """Test successful retrieval of supported models."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        response = await stub.get_supported_models(empty_pb2.Empty())
        assert len(response.models) > 0
        # Verify specific models we expect to be supported
        model_names = {model.name for model in response.models}
        assert OPENAI_MODEL_NAME in model_names
        assert ANTHROPIC_MODEL_NAME in model_names
        # Verify model info fields
        tested_ocs = False
        tested_non_ocs = False
        for model in response.models:
            assert model.type in ('OpenAI', 'Anthropic')
            assert model.display_name
            if model.name == 'openai-compatible-server':
                assert not model.HasField('context_window')
                assert not model.HasField('output_token_limit')
                assert not model.HasField('cost_per_input_token')
                assert model.cost_per_input_token == 0
                assert not model.HasField('cost_per_output_token')
                assert model.cost_per_output_token == 0
                tested_ocs = True
            else:
                assert model.HasField('context_window')
                assert model.context_window > 0
                assert model.HasField('output_token_limit')
                assert model.output_token_limit > 0
                assert model.HasField('cost_per_input_token')
                assert model.cost_per_input_token > 0
                assert model.HasField('cost_per_output_token')
                assert model.cost_per_output_token > 0
                tested_non_ocs = True
        assert tested_ocs
        assert tested_non_ocs

    async def test__get_history__empty(self, grpc_channel):  # noqa: ANN001
        """Test getting history when no conversations exist."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        response = await stub.get_history(empty_pb2.Empty())
        assert len(response.conversations) == 0

    async def test__get_history__with_conversations(self, grpc_channel):  # noqa: ANN001
        """Test getting history after creating conversations with messages."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create first conversation with single model response
        request1 = chat_pb2.ChatRequest(
            model_configs=[get_default_model_config('OPENAI_API_KEY')],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="What is 2+2?",
                ),
            ],
        )
        conv1_id = None
        async for response in stub.chat(request1):
            assert not response.HasField('error')
            conv1_id = response.conversation_id
        assert conv1_id

        history = await stub.get_history(empty_pb2.Empty())
        assert len(history.conversations) == 1

        history_conv_1 = history.conversations[0]
        assert history_conv_1.conversation_id == conv1_id
        assert len(history_conv_1.entries) == 2
        assert history_conv_1.entries[0].chat_message.role == chat_pb2.Role.USER
        assert history_conv_1.entries[0].chat_message.content == "What is 2+2?"
        assert history_conv_1.entries[0].entry_id
        assert history_conv_1.entries[0].timestamp.ToDatetime()
        assert history_conv_1.entries[1].HasField("single_model_response")
        assert history_conv_1.entries[1].single_model_response.message.role == chat_pb2.Role.ASSISTANT  # noqa: E501
        assert "4" in history_conv_1.entries[1].single_model_response.message.content
        assert history_conv_1.entries[1].single_model_response.config_snapshot.client_type == "OpenAI"  # noqa: E501
        assert history_conv_1.entries[1].single_model_response.config_snapshot.model_name == OPENAI_MODEL_NAME  # noqa: E501
        assert history_conv_1.entries[1].single_model_response.config_snapshot.model_parameters.HasField("temperature")  # noqa: E501
        assert history_conv_1.entries[1].single_model_response.config_snapshot.model_parameters.temperature == pytest.approx(0.1)  # noqa: E501
        assert history_conv_1.entries[1].single_model_response.config_snapshot.model_parameters.HasField("max_tokens")  # noqa: E501
        assert history_conv_1.entries[1].single_model_response.config_snapshot.model_parameters.max_tokens == 100  # noqa: E501
        assert not history_conv_1.entries[1].single_model_response.config_snapshot.model_parameters.HasField("top_p")  # noqa: E501
        assert not history_conv_1.entries[1].single_model_response.config_snapshot.model_parameters.HasField("server_url")  # noqa: E501

        # Create second conversation with multi-model response
        model_configs = [
            get_default_model_config('OPENAI_API_KEY'),
            get_default_model_config('OPENAI_API_KEY'),
        ]
        model_configs[0].model_parameters.temperature = 0.1
        model_configs[1].model_parameters.temperature = 0.2
        # hacky way to verify the model index
        request2 = chat_pb2.ChatRequest(
            model_configs=model_configs,
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Tell me a joke",
                ),
            ],
        )
        conv2_id = None
        async for response in stub.chat(request2):
            assert not response.HasField('error')
            if conv2_id is None:
                conv2_id = response.conversation_id

        # Get history
        history_response = await stub.get_history(empty_pb2.Empty())
        assert len(history_response.conversations) == 2

        # Verify conversations are present
        conv_ids = {conv.conversation_id for conv in history_response.conversations}
        assert conv1_id in conv_ids
        assert conv2_id in conv_ids

        # check that the conversation structure is the same as the one
        # created last time we queried the history
        new_conv1 = next(c for c in history_response.conversations if c.conversation_id == conv1_id)  # noqa: E501
        assert history_conv_1 == new_conv1

        # Check second conversation structure
        conv2 = next(c for c in history_response.conversations if c.conversation_id == conv2_id)
        assert len(conv2.entries) == 2
        assert conv2.entries[0].chat_message.role == chat_pb2.Role.USER
        assert conv2.entries[0].chat_message.content == "Tell me a joke"
        assert conv2.entries[0].entry_id
        assert conv2.entries[0].timestamp.ToDatetime()
        assert conv2.entries[1].HasField("multi_model_response")
        assert not conv2.entries[1].multi_model_response.HasField("selected_model_index")
        assert len(conv2.entries[1].multi_model_response.responses) == 2
        for response in conv2.entries[1].multi_model_response.responses:
            assert response.message.role == chat_pb2.Role.ASSISTANT
            assert response.message.content
            assert response.config_snapshot.model_name == OPENAI_MODEL_NAME
            assert response.config_snapshot.model_parameters.HasField("temperature")
            if response.model_index == 0:
                assert response.config_snapshot.model_parameters.temperature == pytest.approx(0.1)
            elif response.model_index == 1:
                assert response.config_snapshot.model_parameters.temperature == pytest.approx(0.2)
            else:
                pytest.fail("Invalid model index")
            assert response.config_snapshot.model_parameters.HasField("max_tokens")
            assert response.config_snapshot.model_parameters.max_tokens == 100
            assert not response.config_snapshot.model_parameters.HasField("top_p")
            assert not response.config_snapshot.model_parameters.HasField("server_url")

    async def test__get_history__conversation_continuity(self, grpc_channel):  # noqa: ANN001
        """Test that history maintains conversation continuity."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request1 = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['2 + 2 = 4'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="What is 2+2?",
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request1):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
        request2 = chat_pb2.ChatRequest(
            conversation_id=conv_id,
            model_configs=[get_mock_model_config(mock_responses=['5'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Add 1 to that result",
                ),
            ],
        )
        async for response in stub.chat(request2):
            assert not response.HasField('error')

        # Get history and verify conversation structure
        history_response = await stub.get_history(empty_pb2.Empty())
        conv = next(c for c in history_response.conversations if c.conversation_id == conv_id)
        # Should have 4 entries: user1, assistant-1, user2, assistant-2
        assert len(conv.entries) == 4
        assert conv.entries[0].chat_message.content == "What is 2+2?"
        assert "4" in conv.entries[1].single_model_response.message.content
        assert conv.entries[2].chat_message.content == "Add 1 to that result"
        assert "5" in conv.entries[3].single_model_response.message.content
        for entry in conv.entries:
            assert entry.entry_id
            assert entry.timestamp.ToDatetime()

    async def test__get_history__error_handling(self, grpc_channel, monkeypatch):  # noqa: ANN001
        """Test error handling in get_history."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)

        # Create mock error in conversation manager
        async def mock_get_all_conversations(*args, **kwargs) -> empty_pb2.Empty:  # noqa: ANN002, ANN003, ARG001
            raise Exception("Database error")

        # Patch the conversation manager method
        from server.conversation_manager import ConversationManager
        monkeypatch.setattr(ConversationManager, "get_all_conversations", mock_get_all_conversations)  # noqa: E501

        # Verify error handling
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.get_history(empty_pb2.Empty())
        assert exc_info.value.code() == grpc.StatusCode.INTERNAL
        assert "Database error" in str(exc_info.value.details())

    @pytest.mark.skipif(os.getenv('ANTHROPIC_API_KEY') is None, reason="ANTHROPIC_API_KEY is not set")  # noqa: E501
    async def test__chat__multi_model_conversation_history_verification(self, grpc_channel):  # noqa: ANN001
        """Test that multiple model responses are correctly stored in history."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)

        model_configs = [
            # using same model even at high temps doesn't seem to work; perhaps OpenAI caching?
            get_default_model_config('OPENAI_API_KEY'),
            get_default_model_config('ANTHROPIC_API_KEY'),
        ]
        model_configs[0].model_parameters.max_tokens = 50
        model_configs[1].model_parameters.max_tokens = 50

        first_message = "Generate a random word surrounded by '|' so that I can parse it. Response with only '<word>'"  # noqa: E501
        second_message = "Repeat the same exact word you just generated."
        # First request with multiple models
        request1 = chat_pb2.ChatRequest(
            model_configs=model_configs,
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content=first_message,
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request1):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id

        # Second request with same multiple models
        request2 = chat_pb2.ChatRequest(
            conversation_id=conv_id,
            model_configs=model_configs,
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content=second_message,
                ),
            ],
        )
        async for response in stub.chat(request2):
            assert not response.HasField('error')

        # Verify history
        history = await stub.get_history(empty_pb2.Empty())
        assert len(history.conversations) == 1
        conv = history.conversations[0]
        assert conv.conversation_id == conv_id
        assert len(conv.entries) == 4  # 2 user messages + 2 multi-model responses

        # Verify first exchange
        assert conv.entries[0].chat_message.role == chat_pb2.Role.USER
        assert conv.entries[0].chat_message.content == first_message

        assert conv.entries[1].HasField("multi_model_response")
        assert not conv.entries[1].multi_model_response.HasField("selected_model_index")
        assert len(conv.entries[1].multi_model_response.responses) == 2
        # First model response
        response_model_index_0 = None
        for response in conv.entries[1].multi_model_response.responses:
            if response.model_index == 0:
                response_model_index_0 = response
        assert response_model_index_0.message
        assert response_model_index_0.config_snapshot == request1.model_configs[0]
        # Second model response
        response_model_index_1 = None
        for response in conv.entries[1].multi_model_response.responses:
            if response.model_index == 1:
                response_model_index_1 = response
        assert response_model_index_1.message
        assert response_model_index_1.config_snapshot == request1.model_configs[1]
        assert response_model_index_0.message.content != response_model_index_1.message.content

        word_0 = response_model_index_0.message.content.split('|')[1].strip()
        word_1 = response_model_index_1.message.content.split('|')[1].strip()

        # Verify second exchange
        assert conv.entries[2].chat_message.role == chat_pb2.Role.USER
        assert conv.entries[2].chat_message.content == second_message

        assert conv.entries[3].HasField("multi_model_response")
        assert len(conv.entries[3].multi_model_response.responses) == 2
        # First model response
        response_2_model_index_0 = None
        for response in conv.entries[3].multi_model_response.responses:
            if response.model_index == 0:
                response_2_model_index_0 = response
        assert response_2_model_index_0.message
        assert response_2_model_index_0.config_snapshot == request2.model_configs[0]

        # Second model response
        response_2_model_index_1 = None
        for response in conv.entries[3].multi_model_response.responses:
            if response.model_index == 1:
                response_2_model_index_1 = response
        assert response_2_model_index_1.message

        assert response_2_model_index_1.config_snapshot == request2.model_configs[1]
        assert word_0 in response_2_model_index_0.message.content
        assert word_1 in response_2_model_index_1.message.content

        # Verify all entries have timestamps and entry_ids
        for entry in conv.entries:
            assert entry.entry_id
            assert entry.timestamp.ToDatetime()

    async def test__chat_entry_ids__single_exchange(self, grpc_channel):  # noqa: ANN001
        """Test entry IDs in a single request-response exchange."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['2 + 2 = 4'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )

        request_entry_ids = []
        response_entry_ids = []
        model_indexes = []
        async for response in stub.chat(request):
            assert not response.HasField('error')
            assert response.request_entry_id
            assert response.entry_id
            assert response.model_index is not None
            request_entry_ids.append(response.request_entry_id)
            response_entry_ids.append(response.entry_id)
            model_indexes.append(response.model_index)

        assert len(set(request_entry_ids)) == 1
        assert len(set(response_entry_ids)) == 1
        assert len(set(model_indexes)) == 1

        assert request_entry_ids[0] != response_entry_ids[0]

        # Verify entries in history
        history = await stub.get_history(empty_pb2.Empty())
        assert len(history.conversations) == 1
        conv = history.conversations[0]
        assert len(conv.entries) == 2
        assert conv.entries[0].entry_id == request_entry_ids[0]
        assert conv.entries[1].entry_id == response_entry_ids[0]

    async def test__chat_entry_ids__conversation_continuation(self, grpc_channel):  # noqa: ANN001
        """Test entry IDs across multiple exchanges in a conversation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # First exchange
        request1 = chat_pb2.ChatRequest(
            model_configs=[get_default_model_config('OPENAI_API_KEY')],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="First message",
                ),
            ],
        )
        request_entry_ids = []
        response_entry_ids = []
        model_indexes = []
        conversation_ids = []
        async for response in stub.chat(request1):
            assert not response.HasField('error')
            assert response.conversation_id
            assert response.request_entry_id
            assert response.entry_id
            assert response.model_index is not None
            conversation_ids.append(response.conversation_id)
            request_entry_ids.append(response.request_entry_id)
            response_entry_ids.append(response.entry_id)
            model_indexes.append(response.model_index)
        assert len(set(conversation_ids)) == 1
        assert len(set(request_entry_ids)) == 1
        assert len(set(response_entry_ids)) == 1
        assert len(set(model_indexes)) == 1
        request1_entry_id = request_entry_ids[0]
        response1_entry_id = response_entry_ids[0]
        assert request1_entry_id != response1_entry_id

        # Second exchange
        request2 = chat_pb2.ChatRequest(
            conversation_id=conversation_ids[0],
            model_configs=[get_default_model_config('OPENAI_API_KEY')],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Second message",
                ),
            ],
        )
        request_entry_ids = []
        response_entry_ids = []
        model_indexes = []
        conversation_ids = []
        async for response in stub.chat(request2):
            assert not response.HasField('error')
            assert response.conversation_id
            assert response.request_entry_id
            assert response.entry_id
            assert response.model_index is not None
            conversation_ids.append(response.conversation_id)
            request_entry_ids.append(response.request_entry_id)
            response_entry_ids.append(response.entry_id)
            model_indexes.append(response.model_index)
        assert len(set(conversation_ids)) == 1
        assert len(set(request_entry_ids)) == 1
        assert len(set(response_entry_ids)) == 1
        assert len(set(model_indexes)) == 1
        request2_entry_id = request_entry_ids[0]
        response2_entry_id = response_entry_ids[0]
        assert request2_entry_id != response2_entry_id

        # Verify all IDs are unique
        all_ids = {request1_entry_id, response1_entry_id, request2_entry_id, response2_entry_id}
        assert len(all_ids) == 4

        # Verify history
        history = await stub.get_history(empty_pb2.Empty())
        conv = next(c for c in history.conversations if c.conversation_id == conversation_ids[0])
        assert len(conv.entries) == 4
        assert conv.entries[0].entry_id == request1_entry_id
        assert conv.entries[1].entry_id == response1_entry_id
        assert conv.entries[2].entry_id == request2_entry_id
        assert conv.entries[3].entry_id == response2_entry_id

    async def test__delete_conversation(self, grpc_channel):  # noqa: ANN001
        """Test successfully deleting a conversation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create a conversation
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['2 + 2 = 4'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
        # Verify conversation exists in history
        history = await stub.get_history(empty_pb2.Empty())
        assert any(conv.conversation_id == conv_id for conv in history.conversations)
        # Delete the conversation
        delete_request = chat_pb2.DeleteConversationRequest(
            conversation_id=conv_id,
        )
        await stub.delete_conversation(delete_request)
        # Verify conversation is deleted from history
        history = await stub.get_history(empty_pb2.Empty())
        assert not any(conv.conversation_id == conv_id for conv in history.conversations)

    async def test__delete_conversation__nonexistent(self, grpc_channel):  # noqa: ANN001
        """Test attempting to delete a nonexistent conversation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        delete_request = chat_pb2.DeleteConversationRequest(
            conversation_id="nonexistent-id",
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.delete_conversation(delete_request)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    async def test__delete_conversation__multiple(self, grpc_channel):  # noqa: ANN001
        """Test deleting one conversation doesn't affect others."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create two conversations
        async def create_conversation(message: str) -> str:
            request = chat_pb2.ChatRequest(
                model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
                messages=[
                    chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content=message,
                    ),
                ],
            )
            conv_id = None
            async for response in stub.chat(request):
                assert not response.HasField('error')
                if conv_id is None:
                    conv_id = response.conversation_id
            return conv_id

        conv_id1 = await create_conversation("First conversation")
        conv_id2 = await create_conversation("Second conversation")

        # Delete first conversation
        delete_request = chat_pb2.DeleteConversationRequest(
            conversation_id=conv_id1,
        )
        await stub.delete_conversation(delete_request)
        # Verify state
        history = await stub.get_history(empty_pb2.Empty())
        assert not any(conv.conversation_id == conv_id1 for conv in history.conversations)
        assert any(conv.conversation_id == conv_id2 for conv in history.conversations)
        # Try to delete first conversation again
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.delete_conversation(delete_request)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    async def test__delete_conversation__concurrent(self, grpc_channel):  # noqa: ANN001
        """Test concurrent deletion requests for the same conversation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create a conversation
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
        # Attempt concurrent deletions
        delete_request = chat_pb2.DeleteConversationRequest(
            conversation_id=conv_id,
        )
        # We expect one deletion to succeed and others to fail with NOT_FOUND
        async def try_delete() -> Optional[str]:
            try:
                await stub.delete_conversation(delete_request)
                return "success"
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    return "not_found"
                return "other_error"

        results = await asyncio.gather(
            try_delete(),
            try_delete(),
            try_delete(),
        )
        # Exactly one should succeed
        assert results.count("success") == 1
        assert results.count("not_found") == 2
        # Verify conversation is deleted
        history = await stub.get_history(empty_pb2.Empty())
        assert not any(conv.conversation_id == conv_id for conv in history.conversations)

    async def test__truncate_conversation__success(self, grpc_channel):  # noqa: ANN001
        """Test successfully truncating a conversation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create a conversation with multiple messages
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Message 1",
                ),
            ],
        )
        # Get conversation ID and entry ID from first exchange
        conv_id = None
        first_response_entry_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
            if response.WhichOneof("response_type") == "chunk":
                first_response_entry_id = response.entry_id
        # Add second message
        request2 = chat_pb2.ChatRequest(
            conversation_id=conv_id,
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Message 2",
                ),
            ],
        )
        async for response in stub.chat(request2):
            assert not response.HasField('error')

        # Truncate at first model response
        truncate_request = chat_pb2.TruncateConversationRequest(
            conversation_id=conv_id,
            entry_id=first_response_entry_id,
        )
        await stub.truncate_conversation(truncate_request)

        # Verify conversation history
        history = await stub.get_history(empty_pb2.Empty())
        assert len(history.conversations) == 1
        conv = history.conversations[0]
        assert conv.conversation_id == conv_id
        assert len(conv.entries) == 1
        assert conv.entries[0].chat_message.content == "Message 1"

    async def test__truncate_conversation__nonexistent_conversation(self, grpc_channel):  # noqa: ANN001
        """Test truncating a nonexistent conversation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.TruncateConversationRequest(
            conversation_id="nonexistent-id",
            entry_id="some-entry-id",
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.truncate_conversation(request)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    async def test__truncate_conversation__nonexistent_entry(self, grpc_channel):  # noqa: ANN001
        """Test truncating at a nonexistent entry."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create a conversation
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            conv_id = response.conversation_id
        # Try to truncate at nonexistent entry
        truncate_request = chat_pb2.TruncateConversationRequest(
            conversation_id=conv_id,
            entry_id="nonexistent-entry-id",
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.truncate_conversation(truncate_request)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test__branch_conversation__rpc_success(self, grpc_channel):  # noqa: ANN001
        """Test successful RPC branch operation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create initial conversation
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Initial message",
                ),
            ],
        )
        original_conv_id = None
        first_response_entry_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            original_conv_id = response.conversation_id
            if response.WhichOneof("response_type") == "chunk":
                first_response_entry_id = response.entry_id

        # add another message/response to the conversation
        request.conversation_id = original_conv_id
        async for response in stub.chat(request):
            assert not response.HasField('error')

        # Branch the conversation at the first response
        branch_request = chat_pb2.BranchConversationRequest(
            conversation_id=original_conv_id,
            entry_id=first_response_entry_id,
        )
        branch_response = await stub.branch_conversation(branch_request)
        assert branch_response.new_conversation_id
        assert branch_response.new_conversation_id != original_conv_id

        # Verify history
        history = await stub.get_history(empty_pb2.Empty())
        assert len(history.conversations) == 2

        # Verify original conversation
        original_conv = next(c for c in history.conversations if c.conversation_id == original_conv_id)  # noqa: E501
        assert original_conv
        assert len(original_conv.entries) == 4

        # Verify branched conversation
        branched_conv = next(c for c in history.conversations if c.conversation_id == branch_response.new_conversation_id)  # noqa: E501
        assert branched_conv
        assert len(branched_conv.entries) == 2
        assert branched_conv.entries[0].chat_message.content == "Initial message"
        assert branched_conv.entries[1].HasField("single_model_response")
        assert branched_conv.entries[1].single_model_response.message.content
        assert branched_conv.entries[1].single_model_response.message.content == original_conv.entries[1].single_model_response.message.content  # noqa: E501
        assert branched_conv.entries[1].single_model_response.config_snapshot == original_conv.entries[1].single_model_response.config_snapshot  # noqa: E501

    async def test__branch_conversation__rpc_errors(self, grpc_channel):  # noqa: ANN001
        """Test RPC error handling."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Test nonexistent conversation
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.branch_conversation(
                chat_pb2.BranchConversationRequest(
                    conversation_id="nonexistent",
                    entry_id="some-id",
                ),
            )
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

        # Create a conversation for further tests
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            conv_id = response.conversation_id
        # Test nonexistent entry
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.branch_conversation(
                chat_pb2.BranchConversationRequest(
                    conversation_id=conv_id,
                    entry_id="nonexistent",
                ),
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test__multi_response_select__success(self, grpc_channel):  # noqa: ANN001
        """Test successful selection of a model response."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create conversation with multi-model response
        request = chat_pb2.ChatRequest(
            model_configs=[
                get_mock_model_config(mock_responses=['Mock Response 1']),
                get_mock_model_config(mock_responses=['Mock Response 2']),
            ],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Generate a random word.",
                ),
            ],
        )
        conv_id = None
        multi_response_entry_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
            if response.WhichOneof("response_type") == "chunk":
                multi_response_entry_id = response.entry_id

        # Select model index 1
        select_request = chat_pb2.MultiResponseSelectRequest(
            conversation_id=conv_id,
            entry_id=multi_response_entry_id,
            selected_model_index=1,
        )
        await stub.multi_response_select(select_request)

        # Verify selection was set
        history = await stub.get_history(empty_pb2.Empty())
        conv = next(c for c in history.conversations if c.conversation_id == conv_id)
        assert len(conv.entries) == 2  # User message + multi-model response
        assert conv.entries[1].multi_model_response.HasField("selected_model_index")
        assert conv.entries[1].multi_model_response.selected_model_index.value == 1

    async def test__multi_response_select__nonexistent_conversation(self, grpc_channel):  # noqa: ANN001
        """Test selecting a response in a nonexistent conversation."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        select_request = chat_pb2.MultiResponseSelectRequest(
            conversation_id="nonexistent",
            entry_id="some-entry",
            selected_model_index=0,
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.multi_response_select(select_request)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    async def test__multi_response_select__nonexistent_entry(self, grpc_channel):  # noqa: ANN001
        """Test selecting a response for a nonexistent entry."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create conversation first
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )
        conv_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id

        # Try to select nonexistent entry
        select_request = chat_pb2.MultiResponseSelectRequest(
            conversation_id=conv_id,
            entry_id="nonexistent-entry",
            selected_model_index=0,
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.multi_response_select(select_request)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test__multi_response_select__non_multi_response_entry(self, grpc_channel):  # noqa: ANN001
        """Test selecting a response for a non-multi-response entry."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create conversation with single model response
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
        )
        conv_id = None
        response_entry_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
            if response.WhichOneof("response_type") == "chunk":
                response_entry_id = response.entry_id

        # Try to select model for single response entry
        select_request = chat_pb2.MultiResponseSelectRequest(
            conversation_id=conv_id,
            entry_id=response_entry_id,
            selected_model_index=0,
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.multi_response_select(select_request)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test__multi_response_select__invalid_model_index(self, grpc_channel):  # noqa: ANN001
        """Test selecting an invalid model index."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Create conversation with multi-model response
        request = chat_pb2.ChatRequest(
            model_configs=[
                get_mock_model_config(mock_responses=['Mock Response 1']),
                get_mock_model_config(mock_responses=['Mock Response 2']),
            ],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Generate a random word.",
                ),
            ],
        )
        conv_id = None
        multi_response_entry_id = None
        async for response in stub.chat(request):
            assert not response.HasField('error')
            if conv_id is None:
                conv_id = response.conversation_id
            if response.WhichOneof("response_type") == "chunk":
                multi_response_entry_id = response.entry_id

        # Try to select invalid model index
        select_request = chat_pb2.MultiResponseSelectRequest(
            conversation_id=conv_id,
            entry_id=multi_response_entry_id,
            selected_model_index=99,  # Invalid index
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.multi_response_select(select_request)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    async def test__chat__with_file_resource(self, grpc_channel):  # noqa: ANN001
        """Test chat with a file resource."""
        try:
            # Create a file resource
            file_path = create_temp_file('Keyword: `super duper`')
            file_name = Path(file_path).name

            # Get both service stubs
            chat_stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
            context_stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)

            # Add resource through context service
            await context_stub.add_resource(chat_pb2.AddResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            ))

            # Create chat request with resource
            request = chat_pb2.ChatRequest(
                model_configs=[get_default_model_config('OPENAI_API_KEY')],
                messages=[
                    chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content=f'What is the keyword in {file_name}?',
                    ),
                ],
                resources=[
                    chat_pb2.Resource(
                        path=file_path,
                        type=chat_pb2.ResourceType.FILE,
                    ),
                ],
            )
            response_text = ''
            async for response in chat_stub.chat(request):
                assert not response.HasField('error')
                if response.WhichOneof('response_type') == 'chunk':
                    response_text += response.chunk.content

            assert 'super duper' in response_text
        finally:
            Path(file_path).unlink(missing_ok=True)

    async def test__chat__with_multiple_resources(self, grpc_channel):  # noqa: ANN001
        """Test chat with multiple resources."""
        try:
            # Create file resources
            file1_path = create_temp_file("Keyword: `super duper`")
            file_name_1 = Path(file1_path).name
            file2_path = create_temp_file("Keyword: `42`")
            file_name_2 = Path(file2_path).name

            chat_stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
            context_stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)

            # Add resources
            for path in [file1_path, file2_path]:
                await context_stub.add_resource(chat_pb2.AddResourceRequest(
                    path=path,
                    type=chat_pb2.ResourceType.FILE,
                ))

            request = chat_pb2.ChatRequest(
                model_configs=[get_default_model_config('OPENAI_API_KEY')],
                messages=[
                    chat_pb2.ChatMessage(
                        role=chat_pb2.Role.USER,
                        content=f"What is the keyword in {file_name_1} and {file_name_2}?",
                    ),
                ],
                resources=[
                    chat_pb2.Resource(
                        path=path,
                        type=chat_pb2.ResourceType.FILE,
                    )
                    for path in [file1_path, file2_path]
                ],
            )

            response_text = ""
            async for response in chat_stub.chat(request):
                assert not response.HasField('error')
                if response.WhichOneof("response_type") == "chunk":
                    response_text += response.chunk.content

            assert 'super duper' in response_text
            assert '42' in response_text
        finally:
            Path(file1_path).unlink(missing_ok=True)
            Path(file2_path).unlink(missing_ok=True)

    async def test__chat__resource_not_found(self, grpc_channel):  # noqa: ANN001
        """Test chat with nonexistent resource."""
        chat_stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.ChatRequest(
            model_configs=[get_mock_model_config(mock_responses=['Mock Response'])],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Test message",
                ),
            ],
            resources=[
                chat_pb2.Resource(
                    path="/nonexistent/file.txt",
                    type=chat_pb2.ResourceType.FILE,
                ),
            ],
        )

        async for response in chat_stub.chat(request):
            if response.WhichOneof("response_type") == "error":
                assert "not found" in response.error.error_message.lower()
                assert response.error.error_code == grpc.StatusCode.NOT_FOUND.value[0]
                return
        # check response code
        pytest.fail("Expected error response")

    # async def test__inject_context__empty_context(self):
    #     messages = [
    #         {
    #         'role': 'user',
    #         'content': 'Test message',
    #         },
    #     ]
    #     messages_cached = deepcopy(messages)
    #     new_messages = inject_context(messages, resource_context=None, instructions=None)
    #     assert messages_cached == messages  # no side effects
    #     assert messages_cached == new_messages  # no changes for empty context

    #     new_messages = inject_context(messages, resource_context='', instructions=[])
    #     assert messages_cached == messages  # no side effects
    #     assert messages_cached == new_messages  # no changes for empty context

    #     new_messages = inject_context(messages, resource_context='', instructions=['', '', None])
    #     assert messages_cached == messages  # no side effects
    #     assert messages_cached == new_messages  # no changes for empty context

    # async def test__inject_context__empty_resource_context_with_instructions(self):
    #     messages = [
    #         {
    #         'role': 'user',
    #         'content': 'Test message',
    #         },
    #     ]
    #     messages_cached = deepcopy(messages)
    #     instructions = [
    #         'Test Instruction 1',
    #         'Test Instruction 2',
    #         'Test Instruction 2',
    #     ]
    #     new_messages = inject_context(messages, resource_context=None, instructions=instructions)
    #     assert messages_cached == messages  # no side effects
    #     assert new_messages[0]['role'] == 'user'
    #     assert messages[0]['content'] in new_messages[0]['content']
    #     assert instructions[0] in new_messages[0]['content']
    #     assert instructions[1] in new_messages[0]['content']
    #     assert new_messages[0]['content'].count(instructions[1]) == 1

    # async def test__inject_context__resource_context_with_empty_instructions(self):
    #     messages = [
    #         {
    #         'role': 'user',
    #         'content': 'Test message',
    #         },
    #     ]
    #     messages_cached = deepcopy(messages)
    #     context = "Test context"
    #     new_messages = inject_context(messages, resource_context=context, instructions=None)
    #     assert messages_cached == messages  # no side effects
    #     assert new_messages[0]['role'] == 'user'
    #     assert messages[0]['content'] in new_messages[0]['content']
    #     assert context in new_messages[0]['content']

    # async def test__inject_context__resource_context_with_instructions(self):
    #     messages = [
    #         {
    #         'role': 'user',
    #         'content': 'Test message',
    #         },
    #     ]
    #     messages_cached = deepcopy(messages)
    #     instructions = [
    #         'Test Instruction 1',
    #         'Test Instruction 2',
    #         'Test Instruction 2',
    #     ]
    #     context = "Test context"
    #     new_messages = inject_context(
    #         messages,
    #         resource_context=context,
    #         instructions=instructions,
    #     )
    #     assert messages_cached == messages  # no side effects
    #     assert new_messages[0]['role'] == 'user'
    #     assert messages[0]['content'] in new_messages[0]['content']
    #     assert instructions[0] in new_messages[0]['content']
    #     assert instructions[1] in new_messages[0]['content']
    #     assert new_messages[0]['content'].count(instructions[1]) == 1
    #     assert context in new_messages[0]['content']


@pytest.mark.asyncio
class TestContextService:
    """Test the ContextService."""

    async def test__add_resource__file__success(self, grpc_channel):  # noqa: ANN001
        """Test successfully adding a file resource."""
        try:
            file_path = create_temp_file("Test content")
            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
            request = chat_pb2.AddResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )
            await stub.add_resource(request)
            get_request = chat_pb2.GetResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            )
            response = await stub.get_resource(get_request)
            assert response.path == file_path
            assert response.type == chat_pb2.ResourceType.FILE
            assert response.content == "Test content"
            assert response.last_accessed
            assert response.last_modified

            # Get again to verify last_accessed updates
            initial_accessed = response.last_accessed
            initial_modified = response.last_modified
            await asyncio.sleep(0.1)  # Ensure time difference
            response2 = await stub.get_resource(get_request)
            assert response2.last_accessed > initial_accessed
            assert response2.last_modified == initial_modified
        finally:
            Path(file_path).unlink(missing_ok=True)

    async def test__add_resource__file__not_found(self, grpc_channel):  # noqa: ANN001
        """Test adding a nonexistent file."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        request = chat_pb2.AddResourceRequest(
            path="/nonexistent/file.txt",
            type=chat_pb2.ResourceType.FILE,
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.add_resource(request)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    async def test__get_resource__file__not_found(self, grpc_channel):  # noqa: ANN001
        """Test getting a nonexistent file."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        request = chat_pb2.GetResourceRequest(
            path="/nonexistent/file.txt",
            type=chat_pb2.ResourceType.FILE,
        )
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.get_resource(request)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    async def test__add_get_resource__directory__success(self, grpc_channel):  # noqa: ANN001
        """Test adding and getting a directory resource."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some files in the directory
            Path(temp_dir, "file1.txt").touch()
            Path(temp_dir, "file2.txt").touch()
            add_request = chat_pb2.AddResourceRequest(
                path=temp_dir,
                type=chat_pb2.ResourceType.DIRECTORY,
            )
            await stub.add_resource(add_request)
            get_request = chat_pb2.GetResourceRequest(
                path=temp_dir,
                type=chat_pb2.ResourceType.DIRECTORY,
            )
            response = await stub.get_resource(get_request)
            assert response.path == temp_dir
            assert response.type == chat_pb2.ResourceType.DIRECTORY
            assert "file1.txt" in response.content
            assert "file2.txt" in response.content
            assert response.last_accessed
            assert response.last_modified

    async def test__add_get_resource__webpage__success(self, grpc_channel):  # noqa: ANN001
        """Test adding and getting a webpage resource."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        url = "https://example.com"
        add_request = chat_pb2.AddResourceRequest(
            path=url,
            type=chat_pb2.ResourceType.WEBPAGE,
        )
        await stub.add_resource(add_request)
        get_request = chat_pb2.GetResourceRequest(
            path=url,
            type=chat_pb2.ResourceType.WEBPAGE,
        )
        response = await stub.get_resource(get_request)
        assert response.path == url
        assert response.type == chat_pb2.ResourceType.WEBPAGE
        assert "Example Domain" in response.content
        assert response.last_accessed
        assert response.last_modified

    async def test__add_get_resource__webpage__arxiv__success(self, grpc_channel):  # noqa: ANN001
        """Test adding and getting a webpage resource."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        url = "https://arxiv.org/pdf/1706.03762"
        add_request = chat_pb2.AddResourceRequest(
            path=url,
            type=chat_pb2.ResourceType.WEBPAGE,
        )
        await stub.add_resource(add_request)
        get_request = chat_pb2.GetResourceRequest(
            path=url,
            type=chat_pb2.ResourceType.WEBPAGE,
        )
        response = await stub.get_resource(get_request)
        assert response.path == url
        assert response.type == chat_pb2.ResourceType.WEBPAGE
        assert "The dominant sequence transduction models" in response.content
        assert response.last_accessed
        assert response.last_modified

    async def test__add_get_resource__file__content_update(self, grpc_channel):  # noqa: ANN001
        """
        For files, during a get-resource operation, the resource should be updated if the last-
        modification time has changed.
        """
        try:
            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
            file_path = create_temp_file("Initial content")
            await stub.add_resource(chat_pb2.AddResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            ))
            initial_response = await stub.get_resource(chat_pb2.GetResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            ))
            assert initial_response.content == "Initial content"
            initial_modified = initial_response.last_modified

            await asyncio.sleep(0.1)  # Ensure modification time difference
            async with aiofiles.open(file_path, 'w') as f:
                await f.write("Updated content")

            # Get updated content
            updated_response = await stub.get_resource(chat_pb2.GetResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            ))
            assert updated_response.content == "Updated content"
            assert updated_response.last_modified > initial_modified

        finally:
            Path(file_path).unlink(missing_ok=True)

    async def test__concurrent_resource_operations2(self, grpc_channel):  # noqa: ANN001
        """Test concurrent add/get operations on resources."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        # Create concurrent add requests
        async def add() -> None:
            await stub.add_resource(chat_pb2.AddResourceRequest(
                path="https://example.com",
                type=chat_pb2.ResourceType.WEBPAGE,
            ))
            await stub.add_resource(chat_pb2.AddResourceRequest(
                path="https://arxiv.org/pdf/1706.03762",
                type=chat_pb2.ResourceType.WEBPAGE,
            ))

        # Run multiple concurrent operations
        await asyncio.gather(*(add() for _ in range(10)))

        # Verify all resources are added
        get_request = chat_pb2.GetResourceRequest(
            path="https://example.com",
            type=chat_pb2.ResourceType.WEBPAGE,
        )
        response = await stub.get_resource(get_request)
        assert response.path == "https://example.com"
        assert response.type == chat_pb2.ResourceType.WEBPAGE
        assert "Example Domain" in response.content

        get_request = chat_pb2.GetResourceRequest(
            path="https://arxiv.org/pdf/1706.03762",
            type=chat_pb2.ResourceType.WEBPAGE,
        )
        response = await stub.get_resource(get_request)
        assert response.path == "https://arxiv.org/pdf/1706.03762"
        assert response.type == chat_pb2.ResourceType.WEBPAGE
        assert "The dominant sequence transduction models" in response.content

    async def test__get_context__single_file(self, grpc_channel):  # noqa: ANN001
        """Test getting context from a single file resource."""
        try:
            file_path = create_temp_file("Test content")
            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)

            await stub.add_resource(chat_pb2.AddResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            ))
            response = await stub.get_context(chat_pb2.ContextRequest(
                resources=[
                    chat_pb2.Resource(
                        path=file_path,
                        type=chat_pb2.ResourceType.FILE,
                    ),
                ],
            ))
            assert "Content from" in response.context
        finally:
            Path(file_path).unlink(missing_ok=True)

    async def test__get_context__multiple_files(self, grpc_channel):  # noqa: ANN001
        """Test getting context from multiple file resources."""
        try:
            file1_path = create_temp_file("Content 1")
            file2_path = create_temp_file("Content 2")
            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)

            # Add resources
            for path in [file1_path, file2_path]:
                await stub.add_resource(chat_pb2.AddResourceRequest(
                    path=path,
                    type=chat_pb2.ResourceType.FILE,
                ))

            response = await stub.get_context(chat_pb2.ContextRequest(
                resources=[
                    chat_pb2.Resource(
                        path=path,
                        type=chat_pb2.ResourceType.FILE,
                    )
                    for path in [file1_path, file2_path]
                ],
            ))
            assert file1_path in response.context
            assert file2_path in response.context
            assert "Content 1" in response.context
            assert "Content 2" in response.context
        finally:
            Path(file1_path).unlink(missing_ok=True)
            Path(file2_path).unlink(missing_ok=True)

    async def test__get_context__with_directory(self, grpc_channel):  # noqa: ANN001
        """Test getting context including a directory resource."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                Path(temp_dir, "file1.txt").touch()
                Path(temp_dir, "file2.txt").touch()
                file_path = create_temp_file("File content")
                stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
                await stub.add_resource(chat_pb2.AddResourceRequest(
                    path=file_path,
                    type=chat_pb2.ResourceType.FILE,
                ))

                response = await stub.get_context(chat_pb2.ContextRequest(
                    resources=[
                        chat_pb2.Resource(
                            path=file_path,
                            type=chat_pb2.ResourceType.FILE,
                        ),
                        chat_pb2.Resource(
                            path=temp_dir,
                            type=chat_pb2.ResourceType.DIRECTORY,
                        ),
                    ],
                    context_strategy=ContextStrategy.RAG,
                ))

                # Check both file and directory content present
                assert temp_dir in response.context
                assert "file1.txt" in response.context
                assert "file2.txt" in response.context
                assert file_path in response.context
                assert "File content" in response.context
            finally:
                Path(file_path).unlink(missing_ok=True)

    async def test__get_context__nonexistent_resource(self, grpc_channel):  # noqa: ANN001
        """Test getting context with a nonexistent resource."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.get_context(chat_pb2.ContextRequest(
                resources=[
                    chat_pb2.Resource(
                        path="/nonexistent/path",
                        type=chat_pb2.ResourceType.FILE,
                    ),
                ],
            ))
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND

    async def test__get_context__empty_resources(self, grpc_channel):  # noqa: ANN001
        """Test getting context with empty resource list."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        response = await stub.get_context(chat_pb2.ContextRequest(resources=[]))
        assert response.context == ""

    async def test__get_context__webpage(self, grpc_channel):  # noqa: ANN001
        """Test getting context including a webpage resource."""
        stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
        url = "https://example.com"
        await stub.add_resource(chat_pb2.AddResourceRequest(
            path=url,
            type=chat_pb2.ResourceType.WEBPAGE,
        ))
        response = await stub.get_context(chat_pb2.ContextRequest(
            resources=[
                chat_pb2.Resource(
                    path=url,
                    type=chat_pb2.ResourceType.WEBPAGE,
                ),
            ],
        ))

        assert url in response.context
        assert "Example Domain" in response.context

    async def test__get_context__multiple_files_rag(self, grpc_channel):  # noqa: ANN001
        """Test getting context from multiple file resources."""
        try:
            file1_path = create_temp_file("This is a sentence on machine learning. " * 1000)
            file2_path = create_temp_file("This is a sentence on chemistry. " * 1000)
            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)

            # Add resources
            for path in [file1_path, file2_path]:
                await stub.add_resource(chat_pb2.AddResourceRequest(
                    path=path,
                    type=chat_pb2.ResourceType.FILE,
                ))

            response = await stub.get_context(chat_pb2.ContextRequest(
                resources=[
                    chat_pb2.Resource(
                        path=path,
                        type=chat_pb2.ResourceType.FILE,
                    )
                    for path in [file1_path, file2_path]
                ],
                rag_query="What is machine learning?",
                rag_similarity_threshold=0.3,
                rag_max_k=2,
                context_strategy=ContextStrategy.RAG,
            ))
            # allow for 2 matched chunks plus 1 overlap + extra
            assert len(response.context) <= CHUNK_SIZE * 4
            assert file1_path in response.context
            assert "machine learning" in response.context
            assert file2_path not in response.context
            assert "chemistry" not in response.context
        finally:
            Path(file1_path).unlink(missing_ok=True)
            Path(file2_path).unlink(missing_ok=True)

    async def test__get_context__with_context_strategy_full_text(self, grpc_channel):  # noqa: ANN001
        """Test that FULL_TEXT strategy uses full content for all resources."""
        try:
            file_path = create_temp_file("Machine learning concepts.\n" * RAG_CHAR_THRESHOLD)
            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
            await stub.add_resource(chat_pb2.AddResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            ))
            response = await stub.get_context(chat_pb2.ContextRequest(
                resources=[
                    chat_pb2.Resource(path=file_path, type=chat_pb2.ResourceType.FILE),
                ],
                rag_query="What is machine learning?",
                rag_similarity_threshold=1.0,
                context_strategy=chat_pb2.ContextStrategy.FULL_TEXT,
            ))
            assert file_path in response.context
            assert "Machine learning concepts" in response.context
            assert file_path in response.context_types
            assert response.context_types[file_path] == chat_pb2.ContextResponse.ContextType.FULL_TEXT  # noqa: E501

            # Test max_content_length
            response = await stub.get_context(chat_pb2.ContextRequest(
                resources=[
                    chat_pb2.Resource(path=file_path, type=chat_pb2.ResourceType.FILE),
                ],
                rag_query="What is machine learning?",
                rag_similarity_threshold=1.0,
                context_strategy=chat_pb2.ContextStrategy.FULL_TEXT,
                max_content_length=10,
            ))
            assert len(response.context) == 10
            assert file_path in response.context_types
            assert response.context_types[file_path] == chat_pb2.ContextResponse.ContextType.FULL_TEXT  # noqa: E501
        finally:
            Path(file_path).unlink(missing_ok=True)

    async def test__get_context__with_context_strategy_rag(self, grpc_channel):  # noqa: ANN001
        try:
            # Create file with content above the RAG threshold
            content = "This text talks about machine learning.\n" * RAG_CHAR_THRESHOLD
            file_path = create_temp_file(content)

            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
            await stub.add_resource(chat_pb2.AddResourceRequest(
                path=file_path,
                type=chat_pb2.ResourceType.FILE,
            ))
            response = await stub.get_context(chat_pb2.ContextRequest(
                resources=[chat_pb2.Resource(path=file_path, type=chat_pb2.ResourceType.FILE)],
                rag_query="What is machine learning?",
                context_strategy=chat_pb2.ContextStrategy.RAG,
            ))
            assert file_path in response.context
            assert "machine learning" in response.context
            assert file_path in response.context_types
            assert response.context_types[file_path] == chat_pb2.ContextResponse.ContextType.RAG
        finally:
            Path(file_path).unlink(missing_ok=True)

    async def test__get_context__with_context_strategy_auto(self, grpc_channel):  # noqa: ANN001
        """Test AUTO strategy with mixed file types."""
        try:
            # Create markdown and code files
            readme_content = "# Project Overview\n" * 100
            readme_path = create_temp_file(readme_content, prefix="Project Overview", suffix='.md')

            code_content = "css stuff" * 10
            code_path = create_temp_file(code_content, prefix="login_page", suffix='.css')

            stub = chat_pb2_grpc.ContextServiceStub(grpc_channel)
            for path in [readme_path, code_path]:
                await stub.add_resource(chat_pb2.AddResourceRequest(
                    path=path,
                    type=chat_pb2.ResourceType.FILE,
                ))

            response = await stub.get_context(chat_pb2.ContextRequest(
                resources=[
                    chat_pb2.Resource(path=readme_path, type=chat_pb2.ResourceType.FILE),
                    chat_pb2.Resource(path=code_path, type=chat_pb2.ResourceType.FILE),
                ],
                rag_query="Summarize this project?",
                rag_similarity_threshold=0.1,
                context_strategy=chat_pb2.ContextStrategy.AUTO,
            ))

            # Verify strategies are returned
            assert readme_path in response.context_types
            assert code_path in response.context_types
            # For documentation query, readme should be RAG or FULL_TEXT
            assert response.context_types[readme_path] in [
                chat_pb2.ContextResponse.ContextType.RAG,
                chat_pb2.ContextResponse.ContextType.FULL_TEXT,
            ]
            # Code should be ignored for this query
            assert response.context_types[code_path] in [
                chat_pb2.ContextResponse.ContextType.IGNORE,
            ]
        finally:
            Path(readme_path).unlink(missing_ok=True)
            Path(code_path).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestConfigurationService:
    """Test model configuration management endpoints."""

    async def test__get_model_configs__empty_initially(self, grpc_channel):  # noqa: ANN001
        """Test getting model configs when none exist."""
        stub = chat_pb2_grpc.ConfigurationServiceStub(grpc_channel)
        response = await stub.get_model_configs(empty_pb2.Empty())
        assert len(response.configs) == EXPECTED_NUM_MODEL_CONFIGS

    async def test__model_config__crud_operations(self, grpc_channel):  # noqa: ANN001
        """Test Create, Read, Update, Delete operations for model configs."""
        stub = chat_pb2_grpc.ConfigurationServiceStub(grpc_channel)
        # Create new config
        new_config = chat_pb2.UserModelConfig(
            config_name="Test Config",
            config=chat_pb2.ModelConfig(
                client_type="OpenAI",
                model_name=OPENAI_MODEL_NAME,
                model_parameters=chat_pb2.ModelParameters(
                    temperature=0.7,
                    max_tokens=100,
                ),
            ),
        )
        # Save config
        save_request = chat_pb2.SaveModelConfigRequest(config=new_config)
        saved_config = await stub.save_model_config(save_request)
        # Verify saved config
        assert saved_config.config_id  # Should have generated ID
        assert saved_config.config_name == new_config.config_name
        assert saved_config.config.client_type == new_config.config.client_type
        assert saved_config.config.model_name == new_config.config.model_name
        assert saved_config.config.model_parameters.temperature == new_config.config.model_parameters.temperature  # noqa: E501
        assert saved_config.config.model_parameters.max_tokens == new_config.config.model_parameters.max_tokens  # noqa: E501
        # Get all configs
        get_response = await stub.get_model_configs(empty_pb2.Empty())
        assert len(get_response.configs) == EXPECTED_NUM_MODEL_CONFIGS + 1
        assert get_response.configs[-1] == saved_config

        # Update config
        saved_config.config.model_parameters.temperature = 0.8
        update_request = chat_pb2.SaveModelConfigRequest(config=saved_config)
        updated_config = await stub.save_model_config(update_request)
        assert updated_config.config_id == saved_config.config_id
        assert updated_config.config_name == saved_config.config_name
        assert updated_config.config.client_type == saved_config.config.client_type
        assert updated_config.config.model_name == saved_config.config.model_name
        assert updated_config.config.model_parameters.temperature == pytest.approx(0.8)
        assert updated_config.config.model_parameters.max_tokens == saved_config.config.model_parameters.max_tokens  # noqa: E501

        get_response = await stub.get_model_configs(empty_pb2.Empty())
        assert get_response.configs[-1] == updated_config

        # Delete config
        delete_request = chat_pb2.DeleteModelConfigRequest(config_id=saved_config.config_id)
        await stub.delete_model_config(delete_request)

        # Verify deletion
        get_response = await stub.get_model_configs(empty_pb2.Empty())
        assert len(get_response.configs) == EXPECTED_NUM_MODEL_CONFIGS

    async def test__delete_model_config__nonexistent(self, grpc_channel):  # noqa: ANN001
        """Test deleting non-existent config."""
        stub = chat_pb2_grpc.ConfigurationServiceStub(grpc_channel)
        delete_request = chat_pb2.DeleteModelConfigRequest(config_id="nonexistent-id")
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.delete_model_config(delete_request)
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
class TestEventStream:
    """Test the event stream."""

    async def test__stream_events__basic_connection(self, grpc_channel):  # noqa: ANN001
        """Test that we can connect to event stream and receive initial event."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        request = chat_pb2.EventStreamRequest()

        events = []
        stream = stub.streamEvents(request)
        try:
            async for event in stream:
                events.append(event)
                if len(events) >= 1:  # Just test initial connection event
                    break
        finally:
            stream.cancel()

        assert len(events) == 1
        assert events[0].type == chat_pb2.ServerEvent.EventType.SERVICE
        assert events[0].level == chat_pb2.ServerEvent.EventLevel.INFO
        assert "connected" in events[0].message.lower()

    async def test__stream_events__receives_chat_events(self, grpc_channel):  # noqa: ANN001
        """Test that chat operations generate appropriate events."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # Start event stream
        events = []
        event_stream = stub.streamEvents(chat_pb2.EventStreamRequest())
        # Start collecting events in background
        async def collect_events() -> None:
            try:
                async for event in event_stream:
                    events.append(event)
                    if len(events) == 2:  # Collect a few events
                        break
            finally:
                event_stream.cancel()

        event_task = asyncio.create_task(collect_events())

        chat_request = chat_pb2.ChatRequest(
            model_configs=[get_default_model_config('OPENAI_API_KEY')],
            messages=[
                chat_pb2.ChatMessage(
                    role=chat_pb2.Role.USER,
                    content="Hello",
                ),
            ],
        )
        async for _ in stub.chat(chat_request):
            pass
        # Wait for events
        await event_task
        # Verify events
        assert len(events) == 2
        assert events[0].type == chat_pb2.ServerEvent.EventType.SERVICE
        assert events[0].level == chat_pb2.ServerEvent.EventLevel.INFO
        assert events[0].timestamp
        assert events[0].timestamp.ToDatetime()
        assert events[1].type == chat_pb2.ServerEvent.EventType.CHAT
        assert events[1].level == chat_pb2.ServerEvent.EventLevel.INFO
        assert events[1].timestamp
        assert events[1].timestamp.ToDatetime()
        assert 'conversation_id' in events[1].metadata
        assert 'model' in events[1].metadata
        assert events[1].metadata['model'] == OPENAI_MODEL_NAME

    async def test__stream_events__multiple_clients(self, grpc_channel):  # noqa: ANN001
        """Test that multiple clients can receive events simultaneously."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)

        async def collect_client_events(num_events: int):  # noqa: ANN202
            events = []
            stream = stub.streamEvents(chat_pb2.EventStreamRequest())
            try:
                async for event in stream:
                    events.append(event)
                    if len(events) >= num_events:
                        break
            finally:
                stream.cancel()
            return events
        # Start two clients
        client1_task = asyncio.create_task(collect_client_events(1))
        client2_task = asyncio.create_task(collect_client_events(1))
        # Wait for both clients
        client1_events = await client1_task
        client2_events = await client2_task
        # Both clients should receive events
        assert len(client1_events) == 1
        assert len(client2_events) == 1
        assert client1_events[0].type == chat_pb2.ServerEvent.EventType.SERVICE
        assert client2_events[0].type == chat_pb2.ServerEvent.EventType.SERVICE

    async def test__stream_events__client_disconnect(self, grpc_channel):  # noqa: ANN001
        """Test that client disconnection is handled gracefully."""
        stub = chat_pb2_grpc.CompletionServiceStub(grpc_channel)
        # First stream
        stream = stub.streamEvents(chat_pb2.EventStreamRequest())
        events_received = 0
        try:
            async for event in stream:
                events_received += 1
                if events_received >= 1:
                    break
        finally:
            stream.cancel()
        # Should be able to start a new stream after disconnecting
        new_stream = stub.streamEvents(chat_pb2.EventStreamRequest())
        new_events = []
        try:
            async for event in new_stream:
                new_events.append(event)
                if len(new_events) >= 1:
                    break
        finally:
            new_stream.cancel()
        assert len(new_events) == 1
        assert new_events[0].type == chat_pb2.ServerEvent.EventType.SERVICE
