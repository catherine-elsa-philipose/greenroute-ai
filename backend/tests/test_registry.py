from app.registry.base import BaseLLMProvider
from app.registry.mock_provider import MockLLMProvider
from app.schemas.provider import ModelCapabilities, ProviderResponse

def test_mock_provider_implements_interface():
    provider = MockLLMProvider()
    assert isinstance(provider, BaseLLMProvider)
    
    info = provider.get_info()
    assert info["provider_name"] == "mock_provider"
    assert info["model_name"] == "mock-model-v1"
    assert info["available"] is True

def test_mock_provider_generate():
    provider = MockLLMProvider()
    prompt = "Hello GreenRoute"
    response = provider.generate(prompt)
    
    assert isinstance(response, ProviderResponse)
    assert response.provider_name == "mock_provider"
    assert response.model_name == "mock-model-v1"
    assert "Hello GreenRoute" in response.content

def test_mock_provider_availability_toggle():
    provider = MockLLMProvider()
    assert provider.is_available() is True
    
    provider.set_availability(False)
    assert provider.is_available() is False
    assert provider.get_info()["available"] is False
