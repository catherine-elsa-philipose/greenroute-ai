import pytest
from pydantic import ValidationError
from app.schemas.provider import ModelProfile, ModelCapabilities
from app.registry.capability_registry import (
    CapabilityRegistry,
    DuplicateModelError,
    ModelNotFoundError
)

def test_register_and_get_model():
    registry = CapabilityRegistry()
    profile = ModelProfile(
        model_id="mock-model-a",
        provider_name="mock_provider",
        capabilities=ModelCapabilities(coding_strength=0.9, reasoning_strength=0.8)
    )
    
    registered = registry.register_model(profile)
    assert registered.model_id == "mock-model-a"
    
    retrieved = registry.get_model("mock-model-a")
    assert retrieved.capabilities.coding_strength == 0.9
    assert retrieved.capabilities.reasoning_strength == 0.8

def test_duplicate_registration_raises_error():
    registry = CapabilityRegistry()
    profile = ModelProfile(model_id="model-b", provider_name="mock")
    registry.register_model(profile)
    
    with pytest.raises(DuplicateModelError):
        registry.register_model(profile)
        
    # Overwrite mode should succeed
    updated_profile = ModelProfile(
        model_id="model-b",
        provider_name="mock",
        capabilities=ModelCapabilities(coding_strength=0.95)
    )
    registry.register_model(updated_profile, overwrite=True)
    assert registry.get_model("model-b").capabilities.coding_strength == 0.95

def test_list_models_and_availability_filter():
    registry = CapabilityRegistry()
    m1 = ModelProfile(model_id="model-1", provider_name="mock", is_available=True)
    m2 = ModelProfile(model_id="model-2", provider_name="mock", is_available=False)
    
    registry.register_model(m1)
    registry.register_model(m2)
    
    assert len(registry.list_models()) == 2
    
    available_models = registry.list_models(available_only=True)
    assert len(available_models) == 1
    assert available_models[0].model_id == "model-1"

def test_set_availability():
    registry = CapabilityRegistry()
    m1 = ModelProfile(model_id="model-1", provider_name="mock", is_available=True)
    registry.register_model(m1)
    
    registry.set_availability("model-1", False)
    assert registry.get_model("model-1").is_available is False

def test_update_capabilities():
    registry = CapabilityRegistry()
    m1 = ModelProfile(model_id="model-1", provider_name="mock")
    registry.register_model(m1)
    
    new_caps = ModelCapabilities(speed_score=0.99, math_strength=0.85)
    registry.update_capabilities("model-1", new_caps)
    
    retrieved = registry.get_model("model-1")
    assert retrieved.capabilities.speed_score == 0.99
    assert retrieved.capabilities.math_strength == 0.85

def test_model_not_found_raises_error():
    registry = CapabilityRegistry()
    with pytest.raises(ModelNotFoundError):
        registry.get_model("non-existent-model")

def test_invalid_capability_scores_validation():
    # Capability score must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        ModelCapabilities(coding_strength=1.5)

    with pytest.raises(ValidationError):
        ModelCapabilities(reasoning_strength=-0.1)
