"""Test the model registry."""
import pytest
from server.supported_models import SupportedModels


def test_model_registry():
    models = [
        {
            'type': 'OpenAI',
            'name': 'gpt-4o-mini',
            'display_name': 'GPT-4o-mini',
        },
        {
            'type': 'OpenAI',
            'name': 'gpt-4o',
            'display_name': 'GPT-4o',
        },
        {
            'type': 'Anthropic',
            'name': 'claude-3-5-haiku-latest',
            'display_name': 'Claude 3.5 Haiku',
        },
        {
            'type': 'Anthropic',
            'name': 'claude-3-5-sonnet-latest',
            'display_name': 'Claude 3.5 Sonnet',
        },
        {
            'type': 'OpenAI',
            'name': 'openai-compatible-server',
            'display_name': 'OpenAI Compatible Server',
        },
    ]
    registry = SupportedModels(models)
    assert len(registry.get_supported_models()) == 5
    assert registry.get_supported_models()[0].type == 'OpenAI'
    assert registry.get_supported_models()[0].name == 'gpt-4o-mini'
    assert registry.get_supported_models()[0].display_name == 'GPT-4o-mini'
    assert registry.get_supported_models()[0].HasField('context_window')
    assert registry.get_supported_models()[0].context_window == 128000
    assert registry.get_supported_models()[0].HasField('output_token_limit')
    assert registry.get_supported_models()[0].output_token_limit == 16384
    assert registry.get_supported_models()[0].HasField('cost_per_input_token')
    assert registry.get_supported_models()[0].cost_per_input_token == pytest.approx(0.00000015)
    assert registry.get_supported_models()[0].HasField('cost_per_output_token')
    assert registry.get_supported_models()[0].cost_per_output_token == pytest.approx(0.0000006)

    assert registry.get_supported_models()[-1].type == 'OpenAI'
    assert registry.get_supported_models()[-1].name == 'openai-compatible-server'
    assert registry.get_supported_models()[-1].display_name == 'OpenAI Compatible Server'
    assert not registry.get_supported_models()[-1].HasField('context_window')
    assert not registry.get_supported_models()[-1].HasField('output_token_limit')
    assert not registry.get_supported_models()[-1].HasField('cost_per_input_token')
    assert not registry.get_supported_models()[-1].HasField('cost_per_output_token')

