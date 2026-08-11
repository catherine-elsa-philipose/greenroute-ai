import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api import deps
from app.memory.outcome_memory import AdaptiveOutcomeMemory

client = TestClient(app)

# Override the memory dependency for testing to use in-memory SQLite instead of file
# Use a single shared instance for testing history across requests
_test_memory = AdaptiveOutcomeMemory(db_path=":memory:")

def override_get_memory():
    return _test_memory

app.dependency_overrides[deps.get_memory] = override_get_memory

def test_api_route_empty_prompt():
    response = client.post("/api/v1/route", json={"prompt": ""})
    assert response.status_code == 400
    assert "Prompt cannot be empty" in response.json()["detail"]

def test_api_route_invalid_request():
    response = client.post("/api/v1/route", json={"wrong_field": "test"})
    assert response.status_code == 422 # Pydantic validation error

def test_api_route_success():
    response = client.post("/api/v1/route", json={"prompt": "Write a binary search function in Python"})
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert data["prompt"] == "Write a binary search function in Python"
    
    # Check that structural components are present
    assert "profile" in data
    assert data["profile"]["task_type"] == "coding"
    
    assert "routing_result" in data
    assert data["routing_result"]["selected_model_id"] == "coder-v1"
    
    assert "confidence_gap_analysis" in data
    
    assert "execution_result" in data
    assert data["execution_result"]["execution_mode"] in ["direct_route", "compare_or_escalate"]
    assert data["execution_result"]["overall_success"] is True
    
    assert "quality_result" in data
    assert data["final_selected_model"] == "coder-v1"
    
    assert "trace" in data
    assert "coder-v1" in data["trace"]["decision_summary"]

def test_api_route_history_signal():
    # Send multiple requests to verify memory accumulates correctly in the in-memory test DB
    prompt = "Write a quick sort function"
    
    # First request - no history
    r1 = client.post("/api/v1/route", json={"prompt": prompt})
    assert r1.status_code == 200
    
    # Second request - should have history from the first
    r2 = client.post("/api/v1/route", json={"prompt": prompt})
    assert r2.status_code == 200
    
    data2 = r2.json()
    assert data2["adaptive_signal"] is not None
    assert data2["adaptive_signal"]["total_samples"] >= 1
    assert data2["adaptive_signal"]["historical_success_rate"] == 1.0
