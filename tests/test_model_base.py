"""Test the model registry."""
import pytest
from server.models.base import Model
from server.models.openai import OPENAI

def test_registry_registration():
    assert Model.is_registered(OPENAI)
    assert not Model.is_registered('TestModel')

    @Model.register('TestModel')
    class TestModel(Model):
        def __init__(self, model_name, **kwargs):  # noqa: ANN001, ANN003
            self.model_name = model_name
            self.kwargs = kwargs

        async def __call__(self, messages, model=None, **kwargs): pass  # noqa

    assert Model.is_registered('TestModel')

    # we should get Assertion error if we try to register the same model again
    with pytest.raises(AssertionError):
        Model.registry.register(type_name='TestModel', item=TestModel)

    model = Model.instantiate({
        'model_type': 'TestModel',
        'model_name': 'test-model',
        'temperature': 0.5,
        'max_tokens': 100,
    })
    assert isinstance(model, TestModel)
    assert model.model_name == 'test-model'
    assert model.kwargs['temperature'] == 0.5
    assert model.kwargs['max_tokens'] == 100
