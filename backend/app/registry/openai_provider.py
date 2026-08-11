import os
import httpx
from typing import Any, Optional
from app.registry.base import BaseLLMProvider
from app.schemas.provider import ModelCapabilities, ProviderResponse

class OpenAIProviderError(Exception):
    pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gpt-4o-mini", capabilities: Optional[ModelCapabilities] = None):
        super().__init__(provider_name="openai", model_name=model_name, capabilities=capabilities)
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self._api_key = os.getenv("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        if not self.is_available():
            raise OpenAIProviderError(f"OpenAI API key missing for model {self.model_name}")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        # We ensure not to pass arbitrary unsupported kwargs if they cause API failures, 
        # but for a simple integration, we pass prompt to messages.
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.api_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                content = data["choices"][0]["message"]["content"]
                return ProviderResponse(
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    content=content,
                    raw_response=data
                )
        except Exception as e:
            raise OpenAIProviderError(f"Failed to generate response from OpenAI: {str(e)}")
