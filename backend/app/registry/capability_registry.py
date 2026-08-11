from typing import Dict, List, Optional
from app.schemas.provider import ModelProfile, ModelCapabilities

class DuplicateModelError(Exception):
    """Raised when registering a model_id that already exists in the registry."""
    pass

class ModelNotFoundError(Exception):
    """Raised when looking up a model_id that is not registered."""
    pass

class CapabilityRegistry:
    """
    Capability Registry (Phase 5).
    
    Manages registration, retrieval, availability, and metadata updates for LLM models/providers.
    This component ONLY stores and serves model capability profiles; it does NOT perform routing decisions.
    
    Values in `ModelCapabilities` are initial configuration metadata that can be updated over time
    via empirical benchmark updates or historical feedback.
    """

    def __init__(self):
        # In-memory repository mapping model_id -> ModelProfile
        self._models: Dict[str, ModelProfile] = {}

    def register_model(self, profile: ModelProfile, overwrite: bool = False) -> ModelProfile:
        """
        Register a model profile in the registry.
        If overwrite is False and model_id exists, raises DuplicateModelError.
        """
        if profile.model_id in self._models and not overwrite:
            raise DuplicateModelError(f"Model '{profile.model_id}' is already registered.")
        
        self._models[profile.model_id] = profile
        return profile

    def get_model(self, model_id: str) -> ModelProfile:
        """
        Retrieve a registered model profile by its model_id.
        Raises ModelNotFoundError if not found.
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in CapabilityRegistry.")
        return self._models[model_id]

    def list_models(self, available_only: bool = False) -> List[ModelProfile]:
        """
        List all registered model profiles.
        If available_only is True, filters out unavailable models.
        """
        profiles = list(self._models.values())
        if available_only:
            return [p for p in profiles if p.is_available]
        return profiles

    def set_availability(self, model_id: str, is_available: bool) -> ModelProfile:
        """
        Update the availability status of a registered model.
        """
        profile = self.get_model(model_id)
        profile.is_available = is_available
        return profile

    def update_capabilities(self, model_id: str, capabilities: ModelCapabilities) -> ModelProfile:
        """
        Update capability scores/metadata for an existing registered model.
        """
        profile = self.get_model(model_id)
        profile.capabilities = capabilities
        return profile

    def unregister_model(self, model_id: str) -> None:
        """
        Remove a model profile from the registry.
        """
        if model_id in self._models:
            del self._models[model_id]
