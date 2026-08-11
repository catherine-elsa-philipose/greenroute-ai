from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.schemas.provider import ModelCapabilities, ModelProfile, ProviderResponse

class BaseLLMProvider(ABC):
    """
    Abstract Base Class for all LLM provider adapters in GreenRoute AI.
    All future providers (OpenAI, Anthropic, Gemini, Ollama, etc.) must implement this interface.
    """

    def __init__(self, provider_name: str, model_name: str, capabilities: Optional[ModelCapabilities] = None):
        self.provider_name = provider_name
        self.model_name = model_name
        self.capabilities = capabilities or ModelCapabilities()

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """
        Generate a text response from the given prompt.
        Must return a standardized ProviderResponse instance.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider/model is currently available and configured.
        """
        pass

    def get_profile(self) -> ModelProfile:
        """
        Returns a complete ModelProfile instance for CapabilityRegistry integration.
        """
        return ModelProfile(
            model_id=f"{self.provider_name}:{self.model_name}",
            provider_name=self.provider_name,
            is_available=self.is_available(),
            capabilities=self.capabilities
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Returns metadata identifying the provider, model, and configured capabilities.
        """
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "available": self.is_available(),
            "capabilities": self.capabilities.model_dump()
        }
