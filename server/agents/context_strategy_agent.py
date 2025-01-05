
"""
Defines an agent that determines the relevance and context needed from a file based on a
question.
"""
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from textwrap import dedent
from openai import AsyncOpenAI
from server.models.openai import (
    AsyncOpenAIFunctionWrapper,
    Function,
    FunctionCallResult,
    Parameter,
)


FUNCTION_DESCRIPTION = dedent("""
- This function returns the "context" from a file. The context is given to an LLM for answering a specific quetion.
- If the file is not needed to directly address the question, we don't need any context from it.
- If a file is needed, we can either use the entire file or use semantic search (RAG) to get relevant sections.
- The relevance of a file is based on the question asked and the name of the file.
- If the file has an obscure name, then it might be from a research paper and is probably relevant and should be included (either fully or via RAG).
- For example, for code files, if the user asks a question about the server, then client files probably aren't needed. Or if the user asks a UI question, then server files probably aren't needed.
- We need to pass in `skip` if the file is not needed, `retrieve_full` if the entire file is needed, or `retrieve_relevant` if we need to use semantic search to get relevant sections.
""").strip()  # noqa: E501


PARAMETER_DESCRIPTION = dedent("""
Determines the context that will be extracted from a file.

- `skip`: The file is not needed to answer the question.
- `retrieve_full`: The file is needed to answer the question, and the entire file content should be retrieved. For example, the entire file is needed if the user asked to summarize the file or to create an outline. Or, for example, if the file type isn't appropriate for semantic search (e.g. code, .csv, etc.), then the entire file should be retrieved.
- `retrieve_relevant`: The file is needed to answer the question, but only relevant sections should be retrieved. For example, if the user asked a question about a specific topic, only the relevant sections of the file should be retrieved.
""").strip()  # noqa: E501


@dataclass
class FileContextFunction:
    """Uses OpenAI function/tool calling for determining document handling strategy."""

    @classmethod
    def create(cls) -> Function:
        """Create the function definition."""
        return Function(
            name='get_context_from_file',
            description=FUNCTION_DESCRIPTION,
            parameters=[
                Parameter(
                    name='file_name',
                    type='string',
                    required=True,
                    description='The full name and path of the file to get context from, including extension',  # noqa: E501
                ),
                Parameter(
                    name='retrieval_strategy',
                    type='string',
                    required=True,
                    description=PARAMETER_DESCRIPTION,
                    enum=[
                        'skip',
                        'retrieve_full',
                        'retrieve_relevant',
                    ],
                ),
                Parameter(
                    name='reasoning',
                    type='string',
                    required=True,
                    description='For logging purposes; a very short explanation of why this retrieval strategy is appropriate',  # noqa: E501
                ),
            ],
        )


@dataclass
class ContextStrategySummary:
    """Summary of context strategy results for all files."""

    results: list[FunctionCallResult]
    total_input_tokens: int
    total_output_tokens: int
    total_input_cost: float
    total_output_cost: float
    total_cost: float


class ContextStrategyAgent:
    """Agent that determines context strategy for files based on a question."""

    def __init__(
        self,
        model: str,
        **model_kwargs: dict[str, object],
    ) -> None:
        """Initialize the agent."""
        self.model = model
        self.model_kwargs = model_kwargs
        self.function = FileContextFunction.create()
        self.client = AsyncOpenAI()
        self.wrapper = AsyncOpenAIFunctionWrapper(
            client=self.client,
            model=self.model,
            functions=[self.function],
            **self.model_kwargs,
        )

    async def __call__(
        self,
        messages: list[dict[str, str]],
        file_names: list[str],
    ) -> ContextStrategySummary:
        """Get context strategies for all files."""
        # Get responses for each file concurrently
        messages = deepcopy(messages)
        responses = await asyncio.gather(*(
            self.wrapper(
                messages=[
                    *messages[:-1],  # Keep all messages except the last one
                    {
                        # unpack all key-value pairs from last message
                        **messages[-1],
                        # update the content with the file name
                        'content': f"{messages[-1]['content']}\n\nFile: {file_name}",
                    },
                ],
                tool_choice='required',
            )
            for file_name in file_names
        ))

        # Aggregate results and costs
        return ContextStrategySummary(
            results=[response.function_call for response in responses],
            total_input_tokens=sum(r.input_tokens for r in responses),
            total_output_tokens=sum(r.output_tokens for r in responses),
            total_input_cost=sum(r.input_cost for r in responses),
            total_output_cost=sum(r.output_cost for r in responses),
            total_cost=sum(r.input_cost + r.output_cost for r in responses),
        )
