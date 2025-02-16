"""Test the model registry."""
import pytest
from server.models.base import Model
from server.models.openai import OPENAI

def test_registry_registration():
    assert Model.is_registered(OPENAI)
    assert not Model.is_registered('TestModel')

    @Model.register('TestModel')
    class TestModel(Model):
        async def __call__(self, messages, model=None, **kwargs): pass  # noqa
        @classmethod
        def provider_name(cls): return 'test'
        @classmethod
        def primary_chat_model_names(cls): return []
        @classmethod
        def supported_chat_model_names(cls): return []
        @classmethod
        def cost_per_token(cls, model_name, token_type): return 0.0  # noqa

    assert Model.is_registered('TestModel')

    # we should get Assertion error if we try to register the same model again
    with pytest.raises(AssertionError):
        Model.registry.register(type_name='TestModel', item=TestModel)
