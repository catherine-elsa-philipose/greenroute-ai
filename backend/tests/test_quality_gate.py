import pytest
from app.schemas.profile import PromptProfile, TaskType, Domain
from app.schemas.execution import MultiModelExecutionResult, ModelExecutionResult, ExecutionMode
from app.schemas.quality import QualityLabel
from app.quality.quality_gate import ResponseQualityGate

def test_strong_relevant_response_receives_good_quality():
    profile = PromptProfile(
        prompt="Write a python function for binary search.",
        task_type=TaskType.CODING,
        domain=Domain.COMPUTER_SCIENCE,
        expected_output="Code"
    )
    exec_res = MultiModelExecutionResult(
        prompt=profile.prompt,
        execution_mode=ExecutionMode.DIRECT_ROUTE,
        overall_success=True,
        executions=[
            ModelExecutionResult(
                model_id="coder-v1",
                provider_name="mock",
                success=True,
                content="def binary_search(arr, target):\n    low, high = 0, len(arr)-1\n    return low",
                latency_ms=10.0
            )
        ],
        status_message="Done"
    )
    
    gate = ResponseQualityGate()
    result = gate.evaluate_execution(profile, exec_res)
    
    assert result.selected_model_id == "coder-v1"
    assert result.selected_quality_score >= 0.75
    assert result.is_weak_response is False
    assert result.recommendation == "ACCEPT"

def test_empty_response_detected_as_weak():
    profile = PromptProfile(prompt="Explain gravity")
    exec_res = MultiModelExecutionResult(
        prompt=profile.prompt,
        execution_mode=ExecutionMode.DIRECT_ROUTE,
        overall_success=True,
        executions=[
            ModelExecutionResult(
                model_id="m1", provider_name="mock", success=True, content="   ", latency_ms=5.0
            )
        ],
        status_message="Done"
    )
    gate = ResponseQualityGate()
    result = gate.evaluate_execution(profile, exec_res)
    
    assert result.is_weak_response is True
    assert result.recommendation == "RE_EVALUATE_OR_FALLBACK"

def test_multi_model_best_response_selected():
    profile = PromptProfile(
        prompt="Write a Python recursion function with explanation.",
        task_type=TaskType.CODING,
        expected_output="Code + Explanation"
    )
    exec_res = MultiModelExecutionResult(
        prompt=profile.prompt,
        execution_mode=ExecutionMode.COMPARE_OR_ESCALATE,
        overall_success=True,
        executions=[
            ModelExecutionResult(
                model_id="m_basic", provider_name="mock", success=True,
                content="Just code: def recur(): pass", latency_ms=10.0
            ),
            ModelExecutionResult(
                model_id="m_detailed", provider_name="mock", success=True,
                content="def recur(n):\n    if n<=1: return 1\n    return n*recur(n-1)\n\nExplanation: This function uses recursion to compute factorials.", latency_ms=15.0
            )
        ],
        status_message="Done"
    )
    gate = ResponseQualityGate()
    result = gate.evaluate_execution(profile, exec_res)
    
    assert result.selected_model_id == "m_detailed"
    assert result.selected_quality_score > result.evaluations[1].overall_quality_score

def test_deterministic_tie_breaking():
    profile = PromptProfile(prompt="Hello world")
    exec_res = MultiModelExecutionResult(
        prompt=profile.prompt,
        execution_mode=ExecutionMode.COMPARE_OR_ESCALATE,
        overall_success=True,
        executions=[
            ModelExecutionResult(model_id="beta-model", provider_name="mock", success=True, content="Hello world from assistant", latency_ms=10.0),
            ModelExecutionResult(model_id="alpha-model", provider_name="mock", success=True, content="Hello world from assistant", latency_ms=10.0),
        ],
        status_message="Done"
    )
    gate = ResponseQualityGate()
    result = gate.evaluate_execution(profile, exec_res)
    # Tie broken alphabetically: 'alpha-model' < 'beta-model'
    assert result.selected_model_id == "alpha-model"
