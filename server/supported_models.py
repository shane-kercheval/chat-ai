"""In-memory registry of supported models loaded from YAML."""
from proto.generated import chat_pb2
from sik_llms import SUPPORTED_MODELS

class SupportedModels:
    """In-memory registry of supported models loaded from YAML."""

    def __init__(self, models: list[dict]):
        """
        Example models.

        ```
        - type: 'OpenAI'
            name: 'gpt-4o-mini'
            display_name: 'GPT-4o-mini'
        - type: 'OpenAI'
            name: 'gpt-4o'
            display_name: 'GPT-4o'
        - type: 'Anthropic'
            name: 'claude-3-5-haiku-latest'
            display_name: 'Claude 3.5 Haiku'
        - type: 'Anthropic'
            name: 'claude-3-5-sonnet-latest'
            display_name: 'Claude 3.5 Sonnet'
        - type: 'OpenAI'
            name: 'openai-compatible-server'
            display_name: 'OpenAI Compatible Server'
        ```
        """
        self._models = []
        for model in models:
            model_name = model['name']
            model_info = SUPPORTED_MODELS.get(model_name)
            self._models.append(chat_pb2.ModelInfo(
                type=model['type'],
                name=model_name,
                display_name=model['display_name'],
                context_window=model_info.context_window_size if model_info else None,
                output_token_limit=model_info.max_output_tokens if model_info else None,
                cost_per_input_token=model_info.pricing['input'] if model_info else None,
                cost_per_output_token=model_info.pricing['output'] if model_info else None,
            ))

    def get_supported_models(self) -> list[chat_pb2.ModelInfo]:
        """Get list of supported models."""
        if not self._models:
            raise ValueError("No models loaded")
        return self._models.copy()
