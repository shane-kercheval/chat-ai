"""Helper functions for OpenAI API."""
from functools import cache
import json
import os
import time
from collections.abc import AsyncGenerator
from openai import AsyncOpenAI
import tiktoken
from tiktoken import Encoding
from server.models import (
    Function,
    FunctionCallResponse,
    FunctionCallResult,
    Model,
    ChatChunkResponse,
    ChatStreamResponseSummary,
)


OPENAI = 'OpenAI'
OPENAI_FUNCTIONS = 'OpenAIFunctions'

CHAT_MODEL_COST_PER_TOKEN = {
    # minor versions
    'gpt-4o-2024-05-13': {'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000},
    'gpt-4o-2024-08-06': {'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000},
    'gpt-4o-2024-11-20': {'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000},
    'gpt-4o-mini-2024-07-18':  {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
    # LEGACY MODELS
    'gpt-4-turbo': {'input': 10.00 / 1_000_000, 'output': 30.00 / 1_000_000},
    'gpt-4-turbo-2024-04-09': {'input': 10.00 / 1_000_000, 'output': 30.00 / 1_000_000},
    'gpt-4-0125-preview': {'input': 0.01 / 1_000, 'output': 0.03 / 1_000},
    'gpt-3.5-turbo': {'input': 0.50 / 1_000_000, 'output': 1.50 / 1_000_000},
    'gpt-3.5-turbo-0125': {'input': 0.50 / 1_000_000, 'output': 1.50 / 1_000_000},
    'gpt-4-0613': {'input': 0.03 / 1_000, 'output': 0.06 / 1_000},
}
CHAT_MODEL_COST_PER_TOKEN_PRIMARY = {
    'gpt-4o-mini': CHAT_MODEL_COST_PER_TOKEN['gpt-4o-mini-2024-07-18'],
    'gpt-4o': CHAT_MODEL_COST_PER_TOKEN['gpt-4o-2024-11-20'],
}
CHAT_MODEL_COST_PER_TOKEN.update(CHAT_MODEL_COST_PER_TOKEN_PRIMARY)


EMBEDDING_MODEL_COST_PER_TOKEN = {
    # "Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens
    # is about 750 words. This paragraph is 35 tokens."
    # https://openai.com/pricing
    # https://platform.openai.com/docs/models
    ####
    # Embedding models
    ####
    # LATEST MODELS
    # https://openai.com/blog/new-embedding-models-and-api-updates
    'text-embedding-3-small': 0.02 / 1_000_000,
    'text-embedding-3-large': 0.13 / 1_000_000,
    # LEGACY MODELS
    'text-embedding-ada-002': 0.1 / 1_000_000,
}

MODEL_COST_PER_TOKEN = CHAT_MODEL_COST_PER_TOKEN | EMBEDDING_MODEL_COST_PER_TOKEN


@cache
def _get_encoding_for_model(model_name: str) -> Encoding:
    """Gets the encoding for a given model so that we can calculate the number of tokens."""
    return tiktoken.encoding_for_model(model_name)


def num_tokens(model_name: str, value: str) -> int:
    """For a given model, returns the number of tokens based on the str `value`."""
    return len(_get_encoding_for_model(model_name=model_name).encode(value))


def num_tokens_from_messages(model_name: str, messages: list[dict]) -> int:
    """
    Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    Returns the number of tokens used by a list of messages.
    """
    if model_name in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        # todo: verify once .ipynb is updated
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-1106",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model_name == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model_name:
        # Warning: gpt-3.5-turbo may update over time.
        # Returning num tokens assuming gpt-3.5-turbo-0613
        return num_tokens_from_messages(model_name="gpt-3.5-turbo-0613", messages=messages)
    elif "gpt-4" in model_name:
        # Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.
        return num_tokens_from_messages(model_name="gpt-4-0613", messages=messages)
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model_name}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")  # noqa
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(_get_encoding_for_model(model_name=model_name).encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _parse_completion_chunk(chunk) -> ChatChunkResponse:  # noqa: ANN001
    assert chunk.object == 'chat.completion.chunk'
    log_prob = None
    if chunk.choices[0].logprobs:
        log_prob = chunk.choices[0].logprobs.content[0].logprob
    return ChatChunkResponse(
        content=chunk.choices[0].delta.content,
        logprob=log_prob,
    )


@Model.register(OPENAI)
class AsyncOpenAICompletionWrapper(Model):
    """
    Wrapper for OpenAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.

    The user can specify the model name, timeout, stream, and other parameters for the API call
    either in the constructor or when calling the object. If the latter, the parameters specified
    when calling the object will override the parameters specified in the constructor.
    """

    def __init__(
            self,
            model_name: str,
            server_url: str | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Initialize the wrapper.

        Args:
            client:
                An instance of the AsyncOpenAI client.
            model_name:
                The model name to use for the API call (e.g. 'gpt-4o-mini').
            server_url:
                The base URL for the API call. Required for the `openai-compatible-server` model.
            **model_kwargs: Additional parameters to pass to the API call
        """
        if model_name == 'openai-compatible-server':
            if not server_url:
                raise ValueError("Missing `server_url` for model `openai-compatible-server`")
            api_key = 'None'
            # If max_tokens is not provided, we'll set it;
            # otherwise the server will just output 1 token; not sure if this is llama.cpp issue
            # or something else
            if model_kwargs.get('max_tokens') is None:
                model_kwargs['max_tokens'] = -1
        else:
            server_url = None
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(f"Missing `api_key` for model `{model_name}`")

        self.client = AsyncOpenAI(base_url=server_url, api_key=api_key)
        self.model = model_name
        self.model_parameters = model_kwargs or {}

    async def __call__(
        self,
        messages: list[dict],
        model_name: str | None = None,
        **model_kwargs: dict,
    ) -> AsyncGenerator[ChatChunkResponse | ChatStreamResponseSummary, None]:
        """
        Streams chat chunks and returns a final summary. Note that any parameters passed to this
        method will override the parameters passed to the constructor.
        """
        model_name = model_name or self.model
        model_parameters = {**self.model_parameters, **model_kwargs}

        start_time = time.time()
        chunks = []
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            logprobs=True,
            **model_parameters,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                parsed_chunk = _parse_completion_chunk(chunk)
                yield parsed_chunk
                chunks.append(parsed_chunk)
        end_time = time.time()
        if model_name == 'openai-compatible-server':
            input_tokens = int(len(str(messages)) / 4)
            output_tokens = int(sum(len(chunk.content) for chunk in chunks) / 4)
            total_input_cost=0
            total_output_cost=0
        else:
            input_tokens = num_tokens_from_messages(model_name, messages)
            output_tokens = sum(num_tokens(model_name, chunk.content) for chunk in chunks)
            total_input_cost=input_tokens * MODEL_COST_PER_TOKEN[model_name]['input']
            total_output_cost=output_tokens * MODEL_COST_PER_TOKEN[model_name]['output']
        yield ChatStreamResponseSummary(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_input_cost=total_input_cost,
            total_output_cost=total_output_cost,
            duration_seconds=end_time - start_time,
        )


@Model.register(OPENAI_FUNCTIONS)
class AsyncOpenAIFunctionWrapper(Model):
    """Wrapper for OpenAI API function calling."""

    def __init__(
            self,
            model_name: str,
            server_url: str | None = None,
            functions: list[Function] | None = None,
            **model_kwargs: dict,
            ) -> None:
        """
        Initialize the wrapper.

        Args:
            client:
                An instance of the AsyncOpenAI client.
            model_name:
                The model name to use for the API call (e.g. 'gpt-4').
            server_url:
                The base URL for the API call. Required for the `openai-compatible-server` model.
            functions:
                List of Function objects defining available functions.
            **model_kwargs:
                Additional parameters to pass to the API call
        """
        if model_name == 'openai-compatible-server':
            if not server_url:
                raise ValueError("Missing `server_url` for model `openai-compatible-server`")
            api_key = 'None'
        else:
            server_url = None
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(f"Missing `api_key` for model `{model_name}`")

        self.client = AsyncOpenAI(base_url=server_url, api_key=api_key)
        self.model = model_name
        self.functions = functions or []
        self.model_kwargs = model_kwargs or {}
        if 'temperature' not in self.model_kwargs:
            self.model_kwargs['temperature'] = 0.2


    async def __call__(
        self,
        messages: list[dict[str, str]],
        functions: list[Function] | None = None,
        tool_choice: str = 'required',
        model_name: str | None = None,
        **model_kwargs: dict[str, object],
    ) -> FunctionCallResponse:
        """
        Call the model with functions.

        Args:
            messages: List of messages to send to the model.
            model_name: Optional model override.
            functions: Optional functions override.
            tool_choice:
                Controls which (if any) tool is called by the model. `none` means the model will
                not call any tool and instead generates a message. `auto` means the model can
                pick between generating a message or calling one or more tools. `required` means
                the model must call one or more tools. Specifying a particular tool via
                `{"type": "function", "function": {"name": "my_function"}}` forces the model to
                call that tool.

                `none` is the default when no tools are present. `auto` is the default if tools
                are present.
            **model_kwargs: Additional parameters to override defaults.
        """
        model_name = model_name or self.model
        functions_list = functions or self.functions
        merged_kwargs = {**self.model_kwargs, **model_kwargs}
        tools = [func.to_dict() for func in functions_list]

        completion = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **merged_kwargs,
        )

        # Extract the function call details
        tool_call = completion.choices[0].message.tool_calls[0]
        function_call = FunctionCallResult(
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
            call_id=tool_call.id,
        )

        # Calculate costs
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens

        input_cost = input_tokens * CHAT_MODEL_COST_PER_TOKEN[model_name]['input']
        output_cost = output_tokens * CHAT_MODEL_COST_PER_TOKEN[model_name]['output']

        return FunctionCallResponse(
            function_call=function_call,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
        )
