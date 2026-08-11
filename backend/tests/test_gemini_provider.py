import pytest
from unittest.mock import patch, MagicMock
from app.registry.gemini_provider import GeminiProvider, GeminiProviderError

def test_gemini_is_available(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    provider = GeminiProvider()
    assert provider.is_available() is True
    
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider()
    assert provider.is_available() is False

def test_gemini_generate_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    provider = GeminiProvider(model_name="test-gemini")
    
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hello Gemini"}]}}]
    }
    
    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        result = provider.generate("Say hello")
        
        assert result.content == "Hello Gemini"
        assert result.provider_name == "google"
        assert result.model_name == "test-gemini"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["contents"][0]["parts"][0]["text"] == "Say hello"

def test_gemini_generate_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider()
    with pytest.raises(GeminiProviderError, match="GEMINI_API_KEY missing"):
        provider.generate("test")
