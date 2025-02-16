"""In-memory registry of supported models loaded from YAML."""
from proto.generated import chat_pb2

class ModelRegistry:
    """In-memory registry of supported models loaded from YAML."""

    def __init__(self, models: list[dict]):
        """
        Example models.

        ```
        - type: 'OpenAI'
            name: 'gpt-4o-mini'
            display_name: 'GPT-4o-mini'
            context_window: 128000
            output_token_limit: 16384
            cost_per_input_token: 0.00000015
            cost_per_output_token: 0.0000006
        - type: 'OpenAI'
            name: 'gpt-4o'
            display_name: 'GPT-4o'
            context_window: 128000
            output_token_limit: 16384
            cost_per_input_token: 0.0000025
            cost_per_output_token: 0.00001
        - type: 'Anthropic'
            name: 'claude-3-5-haiku-latest'
            display_name: 'Claude 3.5 Haiku'
            context_window: 200000
            output_token_limit: 4096
            cost_per_input_token: 0.0000008
            cost_per_output_token: 0.000004
        - type: 'Anthropic'
            name: 'claude-3-5-sonnet-latest'
            display_name: 'Claude 3.5 Sonnet'
            context_window: 200000
            output_token_limit: 4096
            cost_per_input_token: 0.000003
            cost_per_output_token: 0.000015
        - type: 'OpenAI'
            name: 'openai-compatible-server'
            display_name: 'OpenAI Compatible Server'
        ```
        """
        self._models = []
        for model in models:
            self._models.append(chat_pb2.ModelInfo(
                type=model['type'],
                name=model['name'],
                display_name=model['display_name'],
                context_window=model.get('context_window'),
                output_token_limit=model.get('output_token_limit'),
                cost_per_input_token=model.get('cost_per_input_token'),
                cost_per_output_token=model.get('cost_per_output_token'),
            ))

    def get_supported_models(self) -> list[chat_pb2.ModelInfo]:
        """Get list of supported models."""
        if not self._models:
            raise ValueError("No models loaded")
        return self._models.copy()
