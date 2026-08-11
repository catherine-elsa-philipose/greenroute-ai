import pytest
from unittest.mock import patch, MagicMock
from app.registry.openai_provider import OpenAIProvider, OpenAIProviderError

def test_openai_is_available(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    provider = OpenAIProvider()
    assert provider.is_available() is True
    
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()
    assert provider.is_available() is False

def test_openai_generate_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    provider = OpenAIProvider(model_name="test-gpt")
    
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello World"}}]
    }
    
    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        result = provider.generate("Say hello")
        
        assert result.content == "Hello World"
        assert result.provider_name == "openai"
        assert result.model_name == "test-gpt"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["messages"][0]["content"] == "Say hello"

def test_openai_generate_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()
    with pytest.raises(OpenAIProviderError, match="OpenAI API key missing"):
        provider.generate("test")
