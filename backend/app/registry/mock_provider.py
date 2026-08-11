from typing import Any, Optional

class MockLLMProviderError(Exception):
    """Exception raised by MockLLMProvider when configured to simulate execution failure."""
    pass

from app.registry.base import BaseLLMProvider
from app.schemas.provider import ModelCapabilities, ProviderResponse

class MockLLMProvider(BaseLLMProvider):
    """
    Lightweight test provider implementation for verification and local testing.
    Does NOT call any external API or incur cost.
    """

    def __init__(
        self,
        provider_name: str = "mock_provider",
        model_name: str = "mock-model-v1",
        default_response: str = "This is a simulated response from MockLLMProvider.",
        capabilities: Optional[ModelCapabilities] = None
    ):
        super().__init__(provider_name=provider_name, model_name=model_name, capabilities=capabilities)
        self.default_response = default_response
        self._is_active = True
        self._should_fail = False
        self._failure_message = "Simulated provider failure"

    def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        if self._should_fail:
            raise MockLLMProviderError(f"Provider '{self.provider_name}:{self.model_name}' error: {self._failure_message}")
            
        custom_response = kwargs.get("mock_response", self.default_response)
        return ProviderResponse(
            provider_name=self.provider_name,
            model_name=self.model_name,
            content=f"Echo ['{prompt}']: {custom_response}",
            raw_response={"prompt": prompt, "simulated": True}
        )

    def is_available(self) -> bool:
        return self._is_active

    def set_availability(self, available: bool) -> None:
        self._is_active = available

    def simulate_failure(self, should_fail: bool = True, failure_message: str = "Simulated failure") -> None:
        """Configure mock provider to simulate runtime failure during generate()."""
        self._should_fail = should_fail
        self._failure_message = failure_message
