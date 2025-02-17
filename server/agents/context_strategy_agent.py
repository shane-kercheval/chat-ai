
"""
Defines an agent that determines the relevance and context needed from a resource based on a
question.
"""
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from textwrap import dedent
from server.models import (
    Function,
    Parameter,
    Model,
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
Determine which context strategy (ignore, retrieve_full, or retrieve_relevant) is required for each resource to answer a question.
- Use `ignore` if the resource is unnecessary to answer the question.
- Use `retrieve_full` if the entire resource content is needed (e.g., summarization, outlining, or when semantic search isn't suitable like in code or CSV).
- Use `retrieve_relevant` to extract specific relevant sections (e.g., for targeted or topical questions).
- If the user specifies a resource, only consider that resource.
- Infer resource relevance based on its name alone, even without content visibility.
""").strip()  # noqa: E501

RESOURCE_NAME_PARAMETER_DESCRIPTION = dedent("""
The name of the resource currently being evaluated. Must return the exact name of this resource without additional formatting (e.g., no spaces, quotes, or backticks). Ignore resource names mentioned in the user's message; evaluate only the resource explicitly specified in this context."

The name of the resource currently being evaluated to determine its relevance and how context should be extracted, if needed. This refers to the specific resource being considered and referenced in the "Resource name being considered:" section, not any resources mentioned by the user in their question. The name must exactly match the resource being evaluated (no extra formatting such as spaces, quotes, or backticks)."
""").strip()  # noqa: E501


STRATEGY_PARAMETER_DESCRIPTION = dedent("""
Determines the context that will be extracted from a resource.

- `ignore`: The resource is probably irrelevant to the question. For example, for code resources, if the user asks a question about the server, then the client resources probably aren't needed and should be ignored. Or if the user asks a UI question, then server resources probably aren't needed. Or for example, if the user asks a question that is likely to be found in the readme, then we don't need to include the code files/resources.
- `retrieve_full`: Retrieve the entire resource if the full content probably is necessary to answer the question (e.g., summarization, outlines, or like code or CSV that are not suitable for semantic search).
- `retrieve_relevant`: Extract specific relevant sections if the question targets specific topics or information within the resource.
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
                    description=RESOURCE_NAME_PARAMETER_DESCRIPTION,
                ),
                Parameter(
                    name='retrieval_strategy',
                    type='string',
                    required=True,
                    description=STRATEGY_PARAMETER_DESCRIPTION,
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

    def __init__(self, model_config: dict, **model_kwargs: dict[str, object]):
        """
        Initialize the agent.

        Args:
            model_config:
                Configuration for the model. See Model.instantiate for format.
            **model_kwargs:
                Additional keyword arguments for the model.
        """
        model_config['functions'] = [ResourceContextFunction.create()]
        model_config = {**model_config, **model_kwargs}
        self.wrapper = Model.instantiate(model_config)

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
        messages = deepcopy(messages)
        # remove system messages from user inut since that will confuse the agent
        messages = [m for m in messages if m['role'] != 'system']
        responses = await asyncio.gather(*(
            self.wrapper(
                messages=[
                    *messages,  # Keep all messages
                    {'role': 'user', 'content': f"Resource name being considered: `{resource_name}`"},  # noqa: E501
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
