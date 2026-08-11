import os
import httpx
from typing import Any, Optional
from app.registry.base import BaseLLMProvider
from app.schemas.provider import ModelCapabilities, ProviderResponse

class GeminiProviderError(Exception):
    pass

class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash", capabilities: Optional[ModelCapabilities] = None):
        super().__init__(provider_name="google", model_name=model_name, capabilities=capabilities)
        self._api_key = os.getenv("GEMINI_API_KEY")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        if not self.is_available():
            raise GeminiProviderError(f"GEMINI_API_KEY missing for model {self.model_name}")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self._api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                # Parse Gemini response structure
                try:
                    content = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    content = "Could not parse response content."
                    
                return ProviderResponse(
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    content=content,
                    raw_response=data
                )
        except Exception as e:
            raise GeminiProviderError(f"Failed to generate response from Gemini: {str(e)}")
