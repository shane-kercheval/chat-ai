
"""
Defines an agent that determines the relevance and context needed from a resource based on a
question.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from textwrap import dedent
from pydantic import BaseModel
from sik_llms import (
    create_client,
    system_message,
)


class ContextType(Enum):
    """Enumeration of context strategies."""

    IGNORE = 'ignore'
    FULL_TEXT = 'retrieve_full'
    RAG = 'retrieve_relevant'


class ContextStrategy(BaseModel):
    """Context strategy for a resource."""

    resource_name: str
    context_type: ContextType
    reasoning: str


class ContextStrategies(BaseModel):
    """Helper for "structured output" formatting."""

    strategies: list[ContextStrategy]


@dataclass
class ContextStrategySummary:
    """Summary of context strategy results for all resources."""

    strategies: list[ContextStrategy]
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


STRUCTURED_OUTPUT_INSTRUCTIONS = dedent("""
[INSTRUCTIONS]:

Determine the appropriate context strategy for each resource to efficiently answer the provided question:

# Context Strategies

- `FULL_TEXT`: Use when the entire resource content is necessary
    - For document summarization or comprehensive analysis
    - When semantic search is insufficient (e.g., code files, CSV data, structured formats)
    - For broad questions requiring comprehensive document review

- `RAG`: Use to extract specific relevant sections via semantic search
    - When the question is specific and the answer is likely to be found in a specific section of the resource.

- `IGNORE`: Use when the answer is likely not in the resource or the resource is irrelevant.

# Decision Rules

- If the user explicitly specifies resources, ONLY use those resources
- Assess resource relevance based on filename/path even without seeing content.
- If the question is a continuation of a previous question, consider the context strategy that would have been used for the previous question.
- Include a brief 1-2 sentence justification for your selection
- The `resource_name` field MUST EXACTLY match the provided resource name/path
""").strip()  # noqa: E501


class ContextStrategyAgent:
    """Agent that determines context strategy for resources based on a question."""

    def __init__(
            self,
            client_type: str | Enum | None,
            model_name: str,
            **model_kwargs: dict[str, object],
        ):
        """
        Initialize the agent.

        Args:
            client_type:
                Type of model to use. See `sik_llms.create_client` for details.
            model_name:
                Name of the model to use. See `sik_llms.create_client` for details.
            **model_kwargs:
                Additional keyword arguments for the model (e.g. `temperature`, etc.).
        """
        self.model_client = create_client(
            client_type=client_type,
            model_name=model_name,
            response_format=ContextStrategies,
            **model_kwargs,
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
        messages = deepcopy(messages)
        # remove system messages from user inut since that will confuse the agent
        messages = [m for m in messages if m['role'] != 'system']
        prompt = "[RESOURCES]:\n\n" + "\n".join(resource_names)
        messages=[
            *messages,
            system_message(STRUCTURED_OUTPUT_INSTRUCTIONS + "\n\n" + prompt),
        ]
        async for response in self.model_client.stream(messages=messages):
            pass  # response_format only returns one response
        if response.refusal:
            raise ValueError(f"Model refused to provide structured output: '{response.refusal}'")

        strategies = response.parsed.strategies
        # for any missing strategy, add a default IGNORE strategy
        for resource_name in resource_names:
            if not any(s.resource_name == resource_name for s in strategies):
                strategies.append(ContextStrategy(
                    resource_name=resource_name,
                    context_type=ContextType.IGNORE,
                    reasoning="Resource was not mentioned in the conversation.",
                ))
        # order the strategies based on the order of the resources
        strategies = sorted(strategies, key=lambda x: resource_names.index(x.resource_name))
        return ContextStrategySummary(
            strategies=strategies,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            input_cost=response.input_cost,
            output_cost=response.output_cost,
            total_cost=response.input_cost,
        )
