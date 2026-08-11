import pytest
from app.registry.mock_provider import MockLLMProvider
from app.schemas.routing import (
    RoutingResult,
    CandidateScore,
    ConfidenceGapAnalysis,
    ConfidenceLevel,
    RoutingActionRecommendation
)
from app.schemas.execution import ExecutionMode
from app.services.orchestrator import MultiModelOrchestrator

def test_direct_route_executes_single_provider():
    p1 = MockLLMProvider(provider_name="mock_a", model_name="m1", default_response="Ans 1")
    p2 = MockLLMProvider(provider_name="mock_b", model_name="m2", default_response="Ans 2")
    providers = {"m1": p1, "m2": p2}

    routing_result = RoutingResult(
        selected_model_id="m1",
        selected_provider_name="mock_a",
        top_score=0.9,
        candidate_scores=[
            CandidateScore(model_id="m1", provider_name="mock_a", is_available=True, capability_fit_score=0.9, operational_score=0.9, total_score=0.9),
            CandidateScore(model_id="m2", provider_name="mock_b", is_available=True, capability_fit_score=0.6, operational_score=0.6, total_score=0.6),
        ],
        confidence_gap_analysis=ConfidenceGapAnalysis(
            top_model_id="m1", top_score=0.9, second_model_id="m2", second_score=0.6,
            confidence_gap=0.30, confidence_level=ConfidenceLevel.HIGH,
            eligible_candidate_count=2, recommendation=RoutingActionRecommendation.DIRECT_ROUTE,
            explanation="High gap"
        )
    )

    orchestrator = MultiModelOrchestrator()
    result = orchestrator.execute("Hello", routing_result, providers)

    assert result.execution_mode == ExecutionMode.DIRECT_ROUTE
    assert result.overall_success is True
    assert len(result.executions) == 1
    assert result.executions[0].model_id == "m1"
    assert "Ans 1" in result.executions[0].content
    assert result.executions[0].latency_ms >= 0.0

def test_compare_or_escalate_executes_top_candidates():
    p1 = MockLLMProvider(provider_name="mock_a", model_name="m1", default_response="Ans 1")
    p2 = MockLLMProvider(provider_name="mock_b", model_name="m2", default_response="Ans 2")
    p3 = MockLLMProvider(provider_name="mock_c", model_name="m3", default_response="Ans 3")
    providers = {"m1": p1, "m2": p2, "m3": p3}

    routing_result = RoutingResult(
        selected_model_id="m1",
        selected_provider_name="mock_a",
        top_score=0.85,
        candidate_scores=[
            CandidateScore(model_id="m1", provider_name="mock_a", is_available=True, capability_fit_score=0.85, operational_score=0.85, total_score=0.85),
            CandidateScore(model_id="m2", provider_name="mock_b", is_available=True, capability_fit_score=0.83, operational_score=0.83, total_score=0.83),
            CandidateScore(model_id="m3", provider_name="mock_c", is_available=True, capability_fit_score=0.40, operational_score=0.40, total_score=0.40),
        ],
        confidence_gap_analysis=ConfidenceGapAnalysis(
            top_model_id="m1", top_score=0.85, second_model_id="m2", second_score=0.83,
            confidence_gap=0.02, confidence_level=ConfidenceLevel.LOW,
            eligible_candidate_count=3, recommendation=RoutingActionRecommendation.COMPARE_OR_ESCALATE,
            explanation="Low gap"
        )
    )

    orchestrator = MultiModelOrchestrator()
    result = orchestrator.execute("Explain quantum", routing_result, providers, top_k_candidates=2)

    assert result.execution_mode == ExecutionMode.COMPARE_OR_ESCALATE
    assert result.overall_success is True
    assert len(result.executions) == 2
    executed_ids = [e.model_id for e in result.executions]
    assert executed_ids == ["m1", "m2"]
    assert "m3" not in executed_ids  # Lower-ranked model not executed

def test_partial_provider_failure_resilience():
    p1 = MockLLMProvider(provider_name="mock_a", model_name="m1", default_response="Ans 1")
    p2 = MockLLMProvider(provider_name="mock_b", model_name="m2")
    p2.simulate_failure(should_fail=True, failure_message="Connection timeout")
    providers = {"m1": p1, "m2": p2}

    routing_result = RoutingResult(
        selected_model_id="m1",
        selected_provider_name="mock_a",
        top_score=0.85,
        candidate_scores=[
            CandidateScore(model_id="m1", provider_name="mock_a", is_available=True, capability_fit_score=0.85, operational_score=0.85, total_score=0.85),
            CandidateScore(model_id="m2", provider_name="mock_b", is_available=True, capability_fit_score=0.84, operational_score=0.84, total_score=0.84),
        ],
        confidence_gap_analysis=ConfidenceGapAnalysis(
            top_model_id="m1", top_score=0.85, second_model_id="m2", second_score=0.84,
            confidence_gap=0.01, confidence_level=ConfidenceLevel.LOW,
            eligible_candidate_count=2, recommendation=RoutingActionRecommendation.COMPARE_OR_ESCALATE,
            explanation="Low gap"
        )
    )

    orchestrator = MultiModelOrchestrator()
    result = orchestrator.execute("Query", routing_result, providers)

    assert result.overall_success is True
    assert len(result.executions) == 2
    assert result.executions[0].success is True
    assert result.executions[1].success is False
    assert "Connection timeout" in result.executions[1].error_message

def test_all_provider_failure_handled():
    p1 = MockLLMProvider(provider_name="mock_a", model_name="m1")
    p1.simulate_failure(should_fail=True, failure_message="API key expired")
    providers = {"m1": p1}

    routing_result = RoutingResult(
        selected_model_id="m1",
        selected_provider_name="mock_a",
        top_score=0.9,
        candidate_scores=[
            CandidateScore(model_id="m1", provider_name="mock_a", is_available=True, capability_fit_score=0.9, operational_score=0.9, total_score=0.9),
        ],
        confidence_gap_analysis=ConfidenceGapAnalysis(
            top_model_id="m1", top_score=0.9, second_model_id=None, second_score=None,
            confidence_gap=0.0, confidence_level=ConfidenceLevel.HIGH,
            eligible_candidate_count=1, recommendation=RoutingActionRecommendation.DIRECT_ROUTE,
            explanation="Solo"
        )
    )

    orchestrator = MultiModelOrchestrator()
    result = orchestrator.execute("Query", routing_result, providers)

    assert result.overall_success is False
    assert len(result.executions) == 1
    assert result.executions[0].success is False
    assert "API key expired" in result.executions[0].error_message
