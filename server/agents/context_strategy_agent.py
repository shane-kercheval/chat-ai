
"""
Defines an agent that determines the relevance and context needed from a resource based on a
question.
"""
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from textwrap import dedent
from openai import AsyncOpenAI
from server.models.openai import (
    AsyncOpenAIFunctionWrapper,
    Function,
    Parameter,
)


class ContextType(Enum):
    """Enumeration of context strategies."""

    IGNORE = 'ignore'
    FULL_TEXT = 'retrieve_full'
    RAG = 'retrieve_relevant'


@dataclass
class ContextStrategyResult:
    """Result of context strategy for a resource."""

    resource_name: str
    context_type: ContextType
    reasoning: str


@dataclass
class ContextStrategySummary:
    """Summary of context strategy results for all resources."""

    strategies: list[ContextStrategyResult]
    total_input_tokens: int
    total_output_tokens: int
    total_input_cost: float
    total_output_cost: float
    total_cost: float


FUNCTION_DESCRIPTION = dedent("""
- This function returns the "context" from a resource (e.g. a file, directory, webpage, etc.). The context is given to an LLM for answering a specific quetion.
- The relevance of a resource is based on the question asked and using only the name of the resource. You must infer the contents of the resource based on the name of the resource.
""").strip()  # noqa: E501


PARAMETER_DESCRIPTION = dedent("""
Determines the context that will be extracted from a resource.

- We should ignore resoures if the resource is not needed to **directly** address the question.
- We should carefully determine if a resource is needed and avoid using unnecessary resources.
    - For example, for code resources, if the user asks a question about the server, then client resources probably aren't needed. Or if the user asks a UI question, then server resources probably aren't needed.
    - If the user asks a question that is likely to be found in the readme, then we don't need to include the code files/resources.
    - The one exception is that if the user is a pdf with an obscure name, it could be a research paper that could be relevant, and we should use it.
- If the resource is needed to answer the question, we need to determine the context that should be extracted from the resource (either the entire/full resource or only relevant sections).
- We need to pass in `ignore` if the resource is not needed, `retrieve_full` if the entire resource is needed, or `retrieve_relevant` if we need to use semantic search to get relevant sections.

- `ignore`: The resource is not needed to **directly** answer the question.
- `retrieve_full`: The resource is needed to answer the question, and the entire resource content should be retrieved. For example, the entire resource is needed if the user asked to summarize the resource, or to create an outline, an overview, etc.. Or, for example, if the resource type isn't appropriate for semantic search (e.g. code, .csv, etc.), then the entire resource should be retrieved.
- `retrieve_relevant`: The resource is needed to answer the question, but only relevant sections should be retrieved. For example, if the user asks a question about a specific topic and the answer is likely contained in a specific passage, then only the relevant sections of the resource should be retrieved.
""").strip()  # noqa: E501


@dataclass
class ResourceContextFunction:
    """Uses OpenAI function/tool calling for determining document handling strategy."""

    @classmethod
    def create(cls) -> Function:
        """Create the function definition."""
        return Function(
            name='get_context_from_resource',
            description=FUNCTION_DESCRIPTION,
            parameters=[
                Parameter(
                    name='resource_name',
                    type='string',
                    required=True,
                    description='The resource name being considered. This should return the *exact* resource name that was provided (without spacing or "`" or quotes).',  # noqa: E501
                ),
                Parameter(
                    name='retrieval_strategy',
                    type='string',
                    required=True,
                    description=PARAMETER_DESCRIPTION,
                    enum=[
                        'ignore',
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


class ContextStrategyAgent:
    """Agent that determines context strategy for resources based on a question."""

    def __init__(
        self,
        model: str,
        **model_kwargs: dict[str, object],
    ) -> None:
        """Initialize the agent."""
        self.model = model
        self.model_kwargs = model_kwargs
        self.function = ResourceContextFunction.create()
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
        resource_names: list[str],
    ) -> ContextStrategySummary:
        """
        Get context strategies for all resources.

        Args:
            messages:
                List of messages in the conversation from which the context strategy is determined.
                In other words, the question asked by the user and optional previous messages.

                Messages are formatted in `{role: str, content: str}` format.
            resource_names:
                List of resource names to get context strategies for.
        """
        # Get responses for each resource concurrently
        messages = deepcopy(messages)
        # remove system messages from user inut since that will confuse the agent
        messages = [m for m in messages if m['role'] != 'system']
        responses = await asyncio.gather(*(
            self.wrapper(
                messages=[
                    *messages[:-1],  # Keep all messages except the last one
                    {
                        # unpack all key-value pairs from last message
                        **messages[-1],
                        # update the content with the resource name
                        'content': f"{messages[-1]['content']}\n\nResource name: `{resource_name}`",  # noqa: E501
                    },
                ],
                tool_choice='required',
            )
            for resource_name in resource_names
        ))

        strategies = [
            ContextStrategyResult(
                resource_name=r.function_call.arguments['resource_name'],
                context_type=ContextType(r.function_call.arguments['retrieval_strategy']),
                reasoning=r.function_call.arguments['reasoning'],
            )
            for r in responses
        ]
        # Aggregate results and costs
        return ContextStrategySummary(
            strategies=strategies,
            total_input_tokens=sum(r.input_tokens for r in responses),
            total_output_tokens=sum(r.output_tokens for r in responses),
            total_input_cost=sum(r.input_cost for r in responses),
            total_output_cost=sum(r.output_cost for r in responses),
            total_cost=sum(r.input_cost + r.output_cost for r in responses),
        )
