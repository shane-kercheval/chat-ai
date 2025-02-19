"""Main gRPC service for the agent's chat service."""
import argparse
from collections.abc import AsyncGenerator
from copy import deepcopy
from dataclasses import dataclass
import logging
import logging.config
import os
import sys
import textwrap
from uuid import uuid4
import aiofiles
import grpc
import asyncio
from google.protobuf import timestamp_pb2, empty_pb2
from sentence_transformers import SentenceTransformer
import yaml
from server.mcp_server.mcp_manager import MCPClientManager
from server.mcp_server.utilities import convert_mcp_tools_to_functions
from server.mcp_server.functions_agent import (
    AgentEvent,
    Function as AgentFunction,
    FunctionAgent,
    ModelConfiguration,
    ThinkStartEvent,
    ThoughtEvent,
    ToolExecutionResultEvent, ToolExecutionStartEvent,
)
from proto.generated import chat_pb2, chat_pb2_grpc
from server.agents.context_strategy_agent import ContextType
from server.async_merge import AsyncMerge
from server.model_config_manager import ModelConfigManager
from server.supported_models import SupportedModels
from server.models import Model, ChatChunkResponse, ChatStreamResponseSummary
# these need to be imported for the models to be registered
from server.models.anthropic import AsyncAnthropicCompletionWrapper  # noqa: F401
from server.models.openai import (
    AsyncOpenAICompletionWrapper,  # noqa: F401
    AsyncOpenAIFunctionWrapper,  # noqa: F401
    OPENAI_FUNCTIONS,
)
from server.conversation_manager import (
    ConversationManager,
    ConversationNotFoundError,
    convert_proto_messages_to_model_messages,
    convert_to_conversations,
)
from server.resource_manager import ResourceManager, ResourceNotFoundError
from server.vector_db import SimilarityScorer
from dotenv import load_dotenv

load_dotenv()
logging.config.fileConfig('config/logging.conf')


@dataclass
class CompletionServiceConfig:
    """Configuration for the CompletionService."""

    database_uri: str
    supported_models: list[dict]
    channel: grpc.aio.Channel
    initial_conversations: list[chat_pb2.Conversation] | None = None
    rag_similarity_threshold: float = 0.3
    rag_max_k: int = 20
    max_content_length: int = 200_000
    mcp_server_config_path: str | None = None


@dataclass
class ContextServiceConfig:
    """Configuration for the CompletionService."""

    database_uri: str
    num_workers: int
    rag_scorer: SimilarityScorer | None = None
    rag_char_threshold: int = 5000
    context_strategy_model_config: dict | None = None


@dataclass
class ConfigurationServiceConfig:
    """Configuration for the ConfigurationService."""

    database_uri: str
    default_model_configs: list[dict]


def inject_context(
        messages: list[dict],
        resource_context: str | None,
        instructions: list[str] | None,
    ) -> list[dict]:
    """Inject context into the user message."""
    messages = deepcopy(messages)
    final_context = ""
    if resource_context:
        final_context = f"# Use the following context for your response if applicable:\n\n{resource_context}\n\n---\n\n"  # noqa: E501
        logging.info(f"Using resource context with {len(resource_context):,} characters.")

    if instructions:
        instructions = [
            textwrap.dedent(i.strip())
            for i in instructions
            if i and i.strip()  # Filter out empty/whitespace instructions
        ]
        # remove duplicate instructions
        instructions = list(set(instructions))
        if instructions:
            instructions = '\n\n'.join(instructions)
            final_context += f'# Use the following instructions for your response:\n\n{instructions}\n\n---\n\n'  # noqa: E501

    # Create new message with instructions + original content
    # TODO: this assumes the user message is the last message; i don't know why it
    # wouldn't be; but not sure i want to raise an exception if it isn't
    messages[-1]['content'] = f"{final_context}{messages[-1]['content']}"
    return messages


def proto_to_dict(proto_obj) -> dict[str, object]:  # noqa: ANN001
    """Convert protobuf message to dictionary with proper casing."""
    from google.protobuf.json_format import MessageToDict
    return MessageToDict(proto_obj, preserving_proto_field_name=True)


class ToolEventHandler:
    """Handle tool events and build consolidated output string."""

    def __init__(self, conv_id: str, model_index: int, request_id: str, response_id: str):
        self.consolidated_output = []
        self.conv_id = conv_id
        self.model_index = model_index
        self.request_id = request_id
        self.response_id = response_id
        self.response_queue = asyncio.Queue()

    async def handle_event(self, event: AgentEvent) -> None:
        """Handle tool events - build output string and queue response for client."""
        # Create response for client
        response = chat_pb2.ChatStreamResponse()
        response.conversation_id = self.conv_id
        response.model_index = self.model_index
        response.entry_id = self.response_id
        response.request_entry_id = self.request_id

        tool_event = response.tool_event
        if isinstance(event, ThinkStartEvent):
            tool_event.type = chat_pb2.ChatStreamResponse.ToolEvent.THINK_START
            tool_event.iteration = event.iteration
            self.consolidated_output.append(f"\nThinking iteration {event.iteration + 1}...")
        elif isinstance(event, ThoughtEvent):
            tool_event.type = chat_pb2.ChatStreamResponse.ToolEvent.THOUGHT
            tool_event.iteration = event.iteration
            tool_event.thought = event.thought
            self.consolidated_output.append(f"\nThought: {event.thought}")
            if event.tool_name:
                tool_event.tool_name = event.tool_name
                # Convert all values to strings before updating the map
                if event.tool_args:
                    tool_event.tool_args.update({k: str(v) for k, v in event.tool_args.items()})
                self.consolidated_output.append(
                    f"\nUsing tool: {event.tool_name} with args {event.tool_args}",
                )
        elif isinstance(event, ToolExecutionStartEvent):
            tool_event.type = chat_pb2.ChatStreamResponse.ToolEvent.TOOL_EXECUTION_START
            tool_event.iteration = event.iteration
            tool_event.tool_name = event.tool_name
            # Convert all values to strings before updating the map
            if event.tool_args:
                tool_event.tool_args.update({k: str(v) for k, v in event.tool_args.items()})
            self.consolidated_output.append(f"\nExecuting {event.tool_name}...")
        elif isinstance(event, ToolExecutionResultEvent):
            tool_event.type = chat_pb2.ChatStreamResponse.ToolEvent.TOOL_EXECUTION_RESULT
            tool_event.iteration = event.iteration
            tool_event.tool_name = event.tool_name
            tool_event.result = str(event.result)
            self.consolidated_output.append(f"\nResult from {event.tool_name}: {event.result}")

        # Queue the response for the client
        await self.response_queue.put(response)

    def get_output(self) -> str:
        """Get the consolidated output string."""
        return "\n".join(self.consolidated_output)


class CompletionService(chat_pb2_grpc.CompletionServiceServicer):
    """Represents the agent's chat service."""

    def __init__(self, config: CompletionServiceConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.conversation_manager = ConversationManager(
            db_path=config.database_uri,
            initial_conversations=config.initial_conversations,
        )
        self.model_registry = SupportedModels(models=config.supported_models)
        self.event_subscribers = set()  # Track active event streams
        self.channel = config.channel
        self._rag_similarity_threshold = config.rag_similarity_threshold
        self._rag_max_k = config.rag_max_k
        self._max_content_length = config.max_content_length

    async def initialize(self) -> None:
        """Initialize the service. Need to define this in async function."""
        await self.conversation_manager.initialize()

    async def emit_event(
        self,
        type: chat_pb2.ServerEvent.EventType,  # noqa: A002
        level: chat_pb2.ServerEvent.EventLevel,
        message: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Emit event to all active subscribers."""
        timestamp = timestamp_pb2.Timestamp()
        timestamp.GetCurrentTime()
        event = chat_pb2.ServerEvent(
            type=type,
            level=level,
            timestamp=timestamp,
            message=message,
        )
        if metadata:
            event.metadata.update(metadata)

        dead_subscribers = set()
        for subscriber in self.event_subscribers:
            try:
                await subscriber.put(event)
            except asyncio.QueueFull:
                dead_subscribers.add(subscriber)
        self.event_subscribers -= dead_subscribers

    async def streamEvents(  # noqa: N802
        self,
        request: chat_pb2.EventStreamRequest,  # noqa: ARG002
        context: grpc.aio.ServicerContext,
        ) -> AsyncGenerator[chat_pb2.ServerEvent, None]:
        """Stream server events to the client."""
        queue = asyncio.Queue(maxsize=100)
        self.event_subscribers.add(queue)
        try:
            logging.info("Client connected to event stream")
            await self.emit_event(
                chat_pb2.ServerEvent.EventType.SERVICE,
                chat_pb2.ServerEvent.EventLevel.INFO,
                "Client connected to event stream",
            )
            while not context.cancelled():
                try:
                    event = await queue.get()
                    yield event
                except asyncio.CancelledError:
                    break
        finally:
            self.event_subscribers.remove(queue)


    async def _run_function_agent(
        self,
        messages: list[dict],
        model_config: chat_pb2.ModelConfig,
        conv_id: str,
        model_index: int,
        request_id: str,
        response_id: str,
    ) -> str: # type: ignore
        """Run function agent and return consolidated output."""
        # Create event handler that will manage both output string and client responses
        event_handler = ToolEventHandler(conv_id, model_index, request_id, response_id)

        async with MCPClientManager(self.config.mcp_server_config_path) as manager:
            mcp_tools = await manager.list_tools()
            if not mcp_tools:
                raise ValueError("No tools found on any connected MCP servers")

            functions = await convert_mcp_tools_to_functions(mcp_tools, manager)

            # Add list_tools function
            async def list_tools() -> list[dict]:
                return await manager.list_tools()

            list_tools_function = AgentFunction(
                name='list_tools',
                description='List available tools from all connected MCP servers.',
                parameters=[],
                func=list_tools,
            )
            functions.append(list_tools_function)

            agent = FunctionAgent(
                model_config=ModelConfiguration(
                    model="openai/" + model_config.model_name,
                    api_key=os.getenv("OPENAI_API_KEY"),
                ),
                tools=functions,
                callback=event_handler.handle_event,
            )
            agent_task = asyncio.create_task(agent(messages=messages))
            # Stream responses from the queue while agent is running
            while not agent_task.done():
                response = await event_handler.response_queue.get()
                yield response, None

            result = await agent_task
            if result.answer:
                event_handler.consolidated_output.append(f"\n\nFinal tool-based answer: {result.answer}")  # noqa: E501
            # Yield final response with consolidated output
            yield None, event_handler.get_output()

    async def _process_model(
        self,
        model_config: chat_pb2.ModelConfig,
        instructions: list[str],
        conv_id: str,
        model_index: int,
        request_id: str,
        response_id: str,
        resource_context: str | None,
        context: grpc.aio.ServicerContext,
        enable_tools: bool = False,
    ) -> AsyncGenerator[chat_pb2.ChatStreamResponse, None]:
        """Process a single model's chat request."""
        try:
            params = model_config.model_parameters
            logging.info(f"Request from model `{model_config.model_type}`; `{model_config.model_name}`")  # noqa: E501
            logging.info(f"Server URL: `{params.server_url if params.HasField("server_url") else None}`")  # noqa: E501
            logging.info(f"Model index: `{model_index}`")
            logging.info(f"temperature: `{params.temperature if params.HasField("temperature") else None}`")  # noqa: E501
            logging.info(f"max_tokens: `{params.max_tokens if params.HasField("max_tokens") else None}`")  # noqa: E501
            logging.info(f"top_p: `{params.top_p if params.HasField("top_p") else None}`")
            logging.info(f"enable_tools: `{enable_tools}`")

            model_wrapper = Model.instantiate({
                'model_type': model_config.model_type,
                'model_name': model_config.model_name,
                **proto_to_dict(params),
            })

            # Get conversation history
            messages = await self.conversation_manager.get_messages(conv_id)
            model_messages = convert_proto_messages_to_model_messages(
                messages=messages,
                model_config=model_config,
            )
            if not model_messages:
                self.logger.error(f"No messages found for conversation `{conv_id}`")
                if not context.cancelled():
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(f"No messages found for conversation `{conv_id}`")
                raise ValueError(f"No messages found for conversation `{conv_id}`")

            if enable_tools:
                print("Tools enabled")
                # Create queue for tool events
                tool_output = None
                async for response, output in self._run_function_agent(
                    messages=model_messages,
                    model_config=model_config,
                    conv_id=conv_id,
                    model_index=model_index,
                    request_id=request_id,
                    response_id=response_id,
                ):
                    if response:
                        yield response  # Stream the tool events to client
                    if output:
                        tool_output = output  # Capture the final consolidated output

                tool_output = f"# Tool Processing Results:\n\nThe following is additional information from a tool used, please incorporate the information into your response.\n\n{tool_output}\n\n"  # noqa: E501
                print(f"Tool output: {tool_output}")
                model_messages[-1]['content'] = f"{model_messages[-1]['content']}\n\n{tool_output}"

            model_messages = inject_context(
                messages=model_messages,
                resource_context=resource_context,
                instructions=instructions,
            )
            async for response in model_wrapper(messages=model_messages):
                if context.cancelled():
                    return
                if isinstance(response, ChatChunkResponse):
                    yield chat_pb2.ChatStreamResponse(
                        conversation_id=conv_id,
                        chunk=chat_pb2.ChatStreamResponse.Chunk(
                            content=response.content,
                            logprob=response.logprob or 0.0,
                        ),
                        model_index=model_index,
                        entry_id=response_id,
                        request_entry_id=request_id,
                    )
                elif isinstance(response, ChatStreamResponseSummary):
                    yield chat_pb2.ChatStreamResponse(
                        conversation_id=conv_id,
                        summary=chat_pb2.ChatStreamResponse.Summary(
                            input_tokens=response.total_input_tokens,
                            output_tokens=response.total_output_tokens,
                            input_cost=response.total_input_cost,
                            output_cost=response.total_output_cost,
                            duration_seconds=response.duration_seconds,
                        ),
                        model_index=model_index,
                        entry_id=response_id,
                        request_entry_id=request_id,
                    )
                else:
                    logging.error(f"Unknown response type: {response}")

        except Exception as e:
            self.logger.error(f"{conv_id} - Error processing model `{model_config.model_name}`: `{e!s}`")  # noqa: E501
            await self.emit_event(
                chat_pb2.ServerEvent.EventType.CHAT,
                chat_pb2.ServerEvent.EventLevel.ERROR,
                f"Error processing model {model_config.model_name}: {e!s}",
                metadata={
                    'conversation_id': conv_id,
                    'model': model_config.model_name,
                },
            )
            yield chat_pb2.ChatStreamResponse(
                conversation_id=conv_id,
                error=chat_pb2.ChatStreamResponse.ChatError(
                    error_message=str(e),
                    error_code=grpc.StatusCode.INTERNAL.value[0],
                ),
                model_index=model_index,
                entry_id=response_id,
                request_entry_id=request_id,
            )

    async def chat(  # noqa: PLR0912, PLR0915
        self,
        request: chat_pb2.ChatRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[chat_pb2.ChatStreamResponse, None]:
        """Chat with one or more models."""
        conv_id = None
        try:
            conv_id = request.conversation_id
            if not conv_id:
                conv_id = await self.conversation_manager.create_conversation()
            logging.info(f"Request received for conversation {conv_id}")
            await self.emit_event(
                type=chat_pb2.ServerEvent.EventType.CHAT,
                level=chat_pb2.ServerEvent.EventLevel.INFO,
                message="Request received",
                metadata={
                    'conversation_id': conv_id,
                    'model': ', '.join(mc.model_name for mc in request.model_configs),
                },
            )
            # Validate request
            if not request.messages:
                if not context.cancelled():
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("No messages provided")
                return

            # Store user messages in conversation
            timestamp = timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            first_request_id = None
            response_id = str(uuid4())
            for i, msg in enumerate(request.messages):
                entry_id = str(uuid4())
                if i == 0:
                    first_request_id = entry_id
                await self.conversation_manager.add_message(
                    conv_id,
                    chat_pb2.ConversationEntry(
                        entry_id=entry_id,
                        chat_message=msg,
                        timestamp=timestamp,
                    ),
                )

            resource_context = None
            instructions = None
            if request.resources:
                try:
                    context_stub = chat_pb2_grpc.ContextServiceStub(self.channel)
                    context_response = await context_stub.get_context(
                        chat_pb2.ContextRequest(
                            resources=request.resources,
                            rag_query=request.messages[-1].content,
                            rag_similarity_threshold=self._rag_similarity_threshold,
                            rag_max_k=self._rag_max_k,
                            max_content_length=self._max_content_length,
                            context_strategy=request.context_strategy,
                        ),
                    )
                    resource_context = context_response.context
                    logging.info(f"Using resource context with {len(resource_context):,} characters.")  # noqa: E501
                except grpc.aio.AioRpcError as e:
                    # Preserve the original gRPC status code
                    yield chat_pb2.ChatStreamResponse(
                        error=chat_pb2.ChatStreamResponse.ChatError(
                            error_message=str(e),
                            error_code=e.code().value[0],
                        ),
                    )
                    return
                except Exception as e:
                    # For all other exceptions, use INTERNAL status code
                    self.logger.error(f"Error getting resource context: {e}")
                    yield chat_pb2.ChatStreamResponse(
                        error=chat_pb2.ChatStreamResponse.ChatError(
                            error_message=f"Internal error getting resource context: {e!s}",
                            error_code=grpc.StatusCode.INTERNAL.value[0],
                        ),
                    )
                    return

            if request.instructions:
                # remove duplicate instructions
                instructions = list(set(request.instructions))
                logging.info(f"Using {len(instructions)} instructions.")

            # Process all models and collect responses
            responses = []
            generators = [
                self._process_model(
                    model_config=config,
                    instructions=instructions,
                    conv_id=conv_id,
                    model_index=i,
                    request_id=first_request_id,
                    response_id=response_id,
                    resource_context=resource_context,
                    context=context,
                    enable_tools=request.HasField('enable_tools') and request.enable_tools,
                )
                for i, config in enumerate(request.model_configs)
            ]

            async for response in AsyncMerge(generators):
                if context.cancelled():
                    logging.info(f"Request cancelled for conversation {conv_id}")
                    return
                responses.append(response)
                yield response

            # After streaming, organize responses by model
            model_responses = {}  # model_index -> list of chunks
            for response in responses:
                if response.WhichOneof("response_type") == "chunk":
                    if response.model_index not in model_responses:
                        model_responses[response.model_index] = []
                    model_responses[response.model_index].append(response.chunk)

            # Create final ChatModelResponses and store in history
            final_responses = [
                chat_pb2.ChatModelResponse(
                    message=chat_pb2.ChatMessage(
                        role=chat_pb2.Role.ASSISTANT,
                        content="".join(chunk.content for chunk in chunks),
                    ),
                    config_snapshot=request.model_configs[model_index],
                    model_index=model_index,
                )
                for model_index, chunks in model_responses.items()
            ]

            timestamp = timestamp_pb2.Timestamp()
            timestamp.GetCurrentTime()
            if len(final_responses) == 1:
                await self.conversation_manager.add_message(
                    conv_id,
                    chat_pb2.ConversationEntry(
                        entry_id=response_id,
                        single_model_response=final_responses[0],
                        timestamp=timestamp,
                    ),
                )
            elif len(final_responses) > 1:
                await self.conversation_manager.add_message(
                    conv_id,
                    chat_pb2.ConversationEntry(
                        entry_id=response_id,
                        multi_model_response=chat_pb2.MultiChatModelResponse(
                            responses=final_responses,
                        ),
                        timestamp=timestamp,
                    ),
                )
            else:
                logging.error(f"No responses for conversation {conv_id}")
                if not context.cancelled():
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details("No responses")
                return

            logging.info(f"Chat completed `{conv_id}`")
            if resource_context:
                await self.emit_event(
                    type=chat_pb2.ServerEvent.EventType.CHAT,
                    level=chat_pb2.ServerEvent.EventLevel.INFO,
                    message="Using context from resources.",
                    metadata={
                        'conversation_id': conv_id,
                        'Context Length': f"{len(resource_context):,}",
                    },
                )
            if instructions:
                await self.emit_event(
                    type=chat_pb2.ServerEvent.EventType.CHAT,
                    level=chat_pb2.ServerEvent.EventLevel.INFO,
                    message="Using instructions.",
                    metadata={
                        'conversation_id': conv_id,
                        'Number of Instructions': str(len(instructions)),
                    },
                )

        except Exception as e:
            self.logger.error(f"{conv_id} - Error: {e!s}")
            await self.emit_event(
                chat_pb2.ServerEvent.EventType.CHAT,
                chat_pb2.ServerEvent.EventLevel.ERROR,
                f"Error processing chat request: {e!s}",
                metadata={
                    'conversation_id': conv_id,
                },
            )
            if not context.cancelled():
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                yield chat_pb2.ChatStreamResponse(
                    conversation_id=conv_id,
                    error=chat_pb2.ChatStreamResponse.ChatError(
                        error_message=str(e),
                        error_code=grpc.StatusCode.INTERNAL.value[0],
                    ),
                    model_index=0,
                )

    async def get_supported_models(
        self,
        request: empty_pb2.Empty,  # noqa: ARG002
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.GetSupportedModelsResponse:
        """Get list of supported models."""
        try:
            logging.info("Getting supported models")
            models = self.model_registry.get_supported_models()
            return chat_pb2.GetSupportedModelsResponse(models=models)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def get_history(
        self,
        request: empty_pb2.Empty,  # noqa: ARG002
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.GetHistoryResponse:
        """Get all conversation history."""
        try:
            conversations = await self.conversation_manager.get_all_conversations()
            return chat_pb2.GetHistoryResponse(conversations=conversations)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def delete_conversation(
        self,
        request: chat_pb2.DeleteConversationRequest,
        context: grpc.aio.ServicerContext,
    ) -> empty_pb2.Empty:
        """Delete a conversation from history."""
        try:
            logging.info(f"Deleting conversation: {request.conversation_id}")
            await self.conversation_manager.delete_conversation(request.conversation_id)
            return empty_pb2.Empty()
        except ConversationNotFoundError as e:
            self.logger.error(f"Error deleting conversation: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error deleting conversation: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def truncate_conversation(
        self,
        request: chat_pb2.TruncateConversationRequest,
        context: grpc.aio.ServicerContext,
    ) -> empty_pb2.Empty:
        """
        Truncate a conversation at a specific entry/message. Deletes that entry and all subsequent
        entries.
        """
        try:
            logging.info(f"Truncating conversation: `{request.conversation_id}`, entry: `{request.entry_id}`")  # noqa: E501
            await self.conversation_manager.truncate_conversation(
                conv_id=request.conversation_id,
                entry_id=request.entry_id,
            )
            return empty_pb2.Empty()
        except ConversationNotFoundError as e:
            self.logger.error(f"Error truncating conversation: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            raise
        except ValueError as e:
            self.logger.error(f"Error truncating conversation: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error truncating conversation: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def branch_conversation(
        self,
        request: chat_pb2.BranchConversationRequest,
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.BranchConversationResponse:
        """Branch a conversation at a specific entry."""
        try:
            self.logger.info(
                f"Branching conversation: {request.conversation_id} at entry: {request.entry_id}"
                + (f" with model index: {request.model_index}" if request.HasField("model_index") else ""),  # noqa: E501
            )
            new_conv_id = await self.conversation_manager.branch_conversation(
                conv_id=request.conversation_id,
                entry_id=request.entry_id,
                model_index=request.model_index if request.HasField("model_index") else None,
            )
            return chat_pb2.BranchConversationResponse(new_conversation_id=new_conv_id)
        except ConversationNotFoundError as e:
            self.logger.error(f"Error branching conversation: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            raise
        except ValueError as e:
            self.logger.error(f"Error branching conversation: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error branching conversation: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def multi_response_select(
            self,
            request: chat_pb2.MultiResponseSelectRequest,
            context: grpc.aio.ServicerContext,
        ) -> empty_pb2.Empty:
        """Select a specific model response from a multi-model response entry."""
        try:
            self.logger.info(
                f"Selecting model response: {request.selected_model_index}"
                + f" for entry: {request.entry_id} in conversation: {request.conversation_id}",
            )
            await self.conversation_manager.set_multi_response_selected_model(
                conv_id=request.conversation_id,
                entry_id=request.entry_id,
                selected_model_index=request.selected_model_index,
            )
            return empty_pb2.Empty()
        except ConversationNotFoundError as e:
            self.logger.error(f"Error selecting model response: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            raise
        except ValueError as e:
            self.logger.error(f"Error selecting model response: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error selecting model response: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise


class ContextService(chat_pb2_grpc.ContextServiceServicer):
    """Service for managing context resources."""

    def __init__(self, config: ContextServiceConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.resource_manager = ResourceManager(
            db_path=config.database_uri,
            num_workers=config.num_workers,
            rag_scorer=config.rag_scorer,
            rag_char_threshold=config.rag_char_threshold,
            context_strategy_model_config=config.context_strategy_model_config,
        )

    async def initialize(self) -> None:
        """Initialize the service."""
        await self.resource_manager.initialize()

    async def add_resource(
        self,
        request: chat_pb2.AddResourceRequest,
        context: grpc.aio.ServicerContext,
    ) -> empty_pb2.Empty:
        """Add a resource to the context service."""
        try:
            self.logger.info(f"Adding resource (type: `{request.type}`): `{request.path}`")
            await self.resource_manager.add_resource(
                path=request.path,
                type=request.type,
            )
            return empty_pb2.Empty()
        except ResourceNotFoundError as e:
            self.logger.error(f"Resource not found: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            raise
        except ValueError as e:
            self.logger.error(f"Invalid resource request: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error adding resource: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def get_resource(
        self,
        request: chat_pb2.GetResourceRequest,
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.GetResourceResponse:
        """Get a resource from the context service."""
        try:
            self.logger.info(f"Getting resource: {request.path}")
            resource = await self.resource_manager.get_resource(
                path=request.path,
                type=request.type,
            )
            return chat_pb2.GetResourceResponse(
                path=resource.path,
                type=resource.type,
                content=resource.content,
                last_accessed=resource.last_accessed,
                last_modified=resource.last_modified,
            )
        except ResourceNotFoundError as e:
            self.logger.error(f"Resource not found: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            raise
        except ValueError as e:
            self.logger.error(f"Invalid resource request: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error getting resource: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def get_context(
        self,
        request: chat_pb2.ContextRequest,
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.ContextResponse:
        """Get context for the given resources."""
        try:
            logging.info("Getting context for resources:")
            for resource in request.resources:
                logging.info(f"Resource: {resource.path} ({resource.type})")

            query = request.rag_query if request.HasField('rag_query') else None
            sim_threshold = request.rag_similarity_threshold if request.HasField('rag_similarity_threshold') else None  # noqa: E501
            max_k = request.rag_max_k if request.HasField('rag_max_k') else None
            max_content_length = request.max_content_length if request.HasField('max_content_length') else None  # noqa: E501
            strategy = request.context_strategy if request.HasField('context_strategy') else None
            logging.info(f"Context Strategy: {strategy}")
            if sim_threshold is not None:
                logging.info(f"RAG Similarity Threshold: {sim_threshold:.2f}")
            if max_k is not None:
                logging.info(f"RAG Max K: {max_k}")
            if max_content_length is not None:
                logging.info(f"Max Content Length: {max_content_length:,} characters")
            context_str, context_types = await self.resource_manager.create_context(
                resources=request.resources,
                query=query,
                rag_similarity_threshold=sim_threshold,
                rag_max_k=max_k,
                max_content_length=max_content_length,
                context_strategy=strategy,
            )
            logging.info(f"Context context_types: {context_types}")
            # Convert the Python enum values to protobuf enum values
            proto_context_types = {}
            for path, strategy in context_types.items():
                if strategy == ContextType.IGNORE:
                    proto_context_types[path] = chat_pb2.ContextResponse.ContextType.IGNORE
                elif strategy == ContextType.FULL_TEXT:
                    proto_context_types[path] = chat_pb2.ContextResponse.ContextType.FULL_TEXT
                elif strategy == ContextType.RAG:
                    proto_context_types[path] = chat_pb2.ContextResponse.ContextType.RAG
                else:
                    self.logger.error(f"Unknown context strategy: {strategy}")
                    raise ValueError(f"Unknown context strategy: {strategy}")

            return chat_pb2.ContextResponse(
                context=context_str,
                context_types=proto_context_types,
            )
        except ResourceNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error getting context: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise


class ConfigurationService(chat_pb2_grpc.ConfigurationServiceServicer):
    """Represents the agent's chat service."""

    def __init__(self, config: ConfigurationServiceConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_manager = ModelConfigManager(
            db_path=config.database_uri,
            default_model_configs=config.default_model_configs,
        )

    async def initialize(self) -> None:
        """Initialize the service. Need to define this in async function."""
        await self.model_manager.initialize()

    async def get_model_configs(
        self,
        request: empty_pb2.Empty,  # noqa: ARG002
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.GetModelConfigsResponse:
        """Get model configurations for a user."""
        try:
            configs = await self.model_manager.get_model_configs()
            return chat_pb2.GetModelConfigsResponse(configs=configs)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def save_model_config(
        self,
        request: chat_pb2.SaveModelConfigRequest,
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.UserModelConfig:
        """Save a model configuration."""
        try:
            return await self.model_manager.save_model_config(request.config)
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def delete_model_config(
        self,
        request: chat_pb2.DeleteModelConfigRequest,
        context: grpc.aio.ServicerContext,
    ) -> empty_pb2.Empty:
        """Delete a model configuration."""
        try:
            await self.model_manager.delete_model_config(config_id=request.config_id)
            return empty_pb2.Empty()
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            raise
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the gRPC server")
    parser.add_argument(
        "--mcp-config",
        default="/Users/shanekercheval/repos/chat-ai/server/mcp_server/mcp_fake_server_config.json",
        help="Path to MCP server config file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to run the server on",
    )
    return parser.parse_args()


async def serve(mcp_server_config_path: str | None, port: int = 50051) -> None:
    """Start the gRPC server."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'data', 'chat.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async with aiofiles.open(os.path.join(base_dir, 'artifacts', 'supported_models.yaml')) as f:
        content = await f.read()
        supported_models = yaml.safe_load(content)
        supported_models = supported_models['supported_models']

    async with aiofiles.open(os.path.join(base_dir, 'artifacts', 'default_model_configs.yaml')) as f:  # noqa: E501
        content = await f.read()
        default_model_configs = yaml.safe_load(content)
        default_model_configs = default_model_configs['default_model_configs']

    async with aiofiles.open(os.path.join(base_dir, 'artifacts', 'example_history.yaml')) as f:
        content = await f.read()
        initial_conversations = yaml.safe_load(content)
        initial_conversations = initial_conversations['conversations']

    completion_service = CompletionService(
        CompletionServiceConfig(
            database_uri=db_path,
            supported_models=supported_models,
            channel = grpc.aio.insecure_channel(f'[::]:{port}'),
            initial_conversations=convert_to_conversations(initial_conversations),
            mcp_server_config_path=mcp_server_config_path,
        ),
    )
    await completion_service.initialize()

    context_service = ContextService(
        ContextServiceConfig(
            database_uri=db_path,
            num_workers=3,
            rag_scorer=SimilarityScorer(SentenceTransformer('all-MiniLM-L6-v2'), chunk_size=500),
            context_strategy_model_config={
                'model_type': OPENAI_FUNCTIONS,
                'model_name': 'gpt-4o',
            },
        ),
    )
    await context_service.initialize()

    config_service = ConfigurationService(
        ConfigurationServiceConfig(
            database_uri=db_path,
            default_model_configs=default_model_configs,
        ),
    )
    await config_service.initialize()

    server = grpc.aio.server(options=[
        # https://grpc.github.io/grpc/core/group__grpc__arg__keys.html
        ('grpc.so_reuseport', 1),
        # How often to send keepalive pings
        ('grpc.keepalive_time_ms', 60000),  # 60 seconds
        # After a duration of this time the client/server pings its peer to see if the transport is still alive.  # noqa: E501
        ('grpc.keepalive_timeout_ms', 20000),  # 20 seconds
        # Is it permissible to send keepalive pings from the client without any outstanding streams.  # noqa: E501
        # If True, allows keepalive pings even when there's no active RPC
        # Important for long-lived streams with infrequent messages
        ('grpc.keepalive_permit_without_calls', True),
    ])

    chat_pb2_grpc.add_CompletionServiceServicer_to_server(completion_service, server)
    chat_pb2_grpc.add_ConfigurationServiceServicer_to_server(config_service, server)
    chat_pb2_grpc.add_ContextServiceServicer_to_server(context_service, server)

    server.add_insecure_port(f'[::]:{port}')
    try:
        await server.start()
        logging.info(f"Server running on port {port}")
        await server.wait_for_termination()
    except Exception as e:
        logging.error(f"Server error: {e}")
        await server.stop(grace=None)
    finally:
        await server.stop(grace=0)


if __name__ == "__main__":
    logging.info(f"Python version: {sys.version}")
    args = parse_args()
    asyncio.run(serve(mcp_server_config_path=args.mcp_config, port=args.port))
