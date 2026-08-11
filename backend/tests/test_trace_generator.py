import pytest
from app.schemas.profile import PromptProfile, TaskType, Domain
from app.schemas.routing import CandidateScore, RoutingResult, ConfidenceGapAnalysis, ConfidenceLevel, RoutingActionRecommendation
from app.schemas.execution import MultiModelExecutionResult, ModelExecutionResult, ExecutionMode
from app.schemas.quality import QualityGateResult, ResponseQualityEvaluation, QualityLabel
from app.schemas.memory import AdaptiveRoutingSignal
from app.services.trace_generator import DecisionTraceGenerator

@pytest.fixture
def sample_pipeline_data():
    profile = PromptProfile(prompt="Write a Python binary search function", task_type=TaskType.CODING, domain=Domain.COMPUTER_SCIENCE)
    
    candidates = [
        CandidateScore(model_id="coder-v1", provider_name="mock", is_available=True, capability_fit_score=0.9, operational_score=0.9, total_score=0.9),
        CandidateScore(model_id="general-v1", provider_name="mock", is_available=True, capability_fit_score=0.7, operational_score=0.7, total_score=0.7)
    ]
    gap_analysis = ConfidenceGapAnalysis(
        top_model_id="coder-v1", second_model_id="general-v1", top_score=0.9, second_score=0.7, confidence_gap=0.2, confidence_level=ConfidenceLevel.HIGH, recommendation=RoutingActionRecommendation.DIRECT_ROUTE, explanation="High gap"
    )
    routing_result = RoutingResult(selected_model_id="coder-v1", selected_provider_name="mock", top_score=0.9, candidate_scores=candidates, confidence_gap_analysis=gap_analysis, status_message="Routing complete")
    
    exec_result = MultiModelExecutionResult(
        prompt=profile.prompt,
        execution_mode=ExecutionMode.DIRECT_ROUTE,
        overall_success=True,
        executions=[ModelExecutionResult(model_id="coder-v1", provider_name="mock", success=True, content="def binary_search(): pass", latency_ms=12.0)],
        status_message="Direct route completed"
    )
    
    qual_eval = ResponseQualityEvaluation(
        model_id="coder-v1", provider_name="mock", relevance_score=0.9, completeness_score=0.8, compliance_score=0.9, consistency_score=1.0, overall_quality_score=0.9, quality_label=QualityLabel.GOOD, evaluation_notes="Good code"
    )
    quality_result = QualityGateResult(evaluations=[qual_eval], selected_model_id="coder-v1", selected_content="def binary_search(): pass", selected_quality_score=0.9, is_weak_response=False, recommendation="ACCEPT", status_message="Good quality")
    
    return profile, routing_result, exec_result, quality_result

def test_complete_direct_route_trace(sample_pipeline_data):
    profile, routing_result, exec_result, quality_result = sample_pipeline_data
    generator = DecisionTraceGenerator()
    
    trace = generator.generate_trace(profile, routing_result, exec_result, quality_result)
    
    assert trace.final_selected_model_id == "coder-v1"
    assert trace.final_selected_content == "def binary_search(): pass"
    assert "coder-v1" in trace.decision_summary
    assert "highest routing score (0.9000)" in trace.decision_summary
    assert "No prior historical outcomes" in trace.decision_summary

def test_low_confidence_compare_escalate_trace(sample_pipeline_data):
    profile, routing_result, exec_result, quality_result = sample_pipeline_data
    
    # Modify for COMPARE_OR_ESCALATE
    routing_result.confidence_gap_analysis.confidence_gap = 0.02
    routing_result.confidence_gap_analysis.confidence_level = ConfidenceLevel.LOW
    routing_result.confidence_gap_analysis.recommendation = RoutingActionRecommendation.COMPARE_OR_ESCALATE
    
    exec_result.execution_mode = ExecutionMode.COMPARE_OR_ESCALATE
    exec_result.executions.append(ModelExecutionResult(model_id="general-v1", provider_name="mock", success=True, content="binary search text", latency_ms=15.0))
    
    generator = DecisionTraceGenerator()
    trace = generator.generate_trace(profile, routing_result, exec_result, quality_result)
    
    assert "compare_or_escalate" in trace.decision_summary
    assert "LOW confidence" in trace.decision_summary

def test_trace_with_provider_failure(sample_pipeline_data):
    profile, routing_result, exec_result, quality_result = sample_pipeline_data
    
    exec_result.executions.append(ModelExecutionResult(model_id="failing-v1", provider_name="mock", success=False, error_message="Provider crash", latency_ms=5.0))
    
    generator = DecisionTraceGenerator()
    trace = generator.generate_trace(profile, routing_result, exec_result, quality_result)
    
    assert "Provider execution failure occurred for model(s): failing-v1" in trace.decision_summary

def test_trace_with_weak_response(sample_pipeline_data):
    profile, routing_result, exec_result, quality_result = sample_pipeline_data
    
    quality_result.is_weak_response = True
    quality_result.selected_quality_score = 0.35
    
    generator = DecisionTraceGenerator()
    trace = generator.generate_trace(profile, routing_result, exec_result, quality_result)
    
    assert "flagged as WEAK" in trace.decision_summary
    assert "Fallback or re-evaluation is recommended" in trace.decision_summary

def test_trace_with_historical_memory_signal(sample_pipeline_data):
    profile, routing_result, exec_result, quality_result = sample_pipeline_data
    
    signal = AdaptiveRoutingSignal(
        model_id="coder-v1",
        task_type=TaskType.CODING,
        domain=Domain.COMPUTER_SCIENCE,
        historical_quality_score=0.92,
        historical_success_rate=1.0,
        historical_fallback_rate=0.0,
        adaptive_weight_multiplier=1.2,
        total_samples=5
    )
    
    generator = DecisionTraceGenerator()
    trace = generator.generate_trace(profile, routing_result, exec_result, quality_result, adaptive_signal=signal)
    
    assert "Historical memory recorded 5 prior outcome sample(s)" in trace.decision_summary
    assert "1.2000" in trace.decision_summary

def test_deterministic_trace_generation(sample_pipeline_data):
    profile, routing_result, exec_result, quality_result = sample_pipeline_data
    generator = DecisionTraceGenerator()
    
    t1 = generator.generate_trace(profile, routing_result, exec_result, quality_result, trace_id="fixed-id-1")
    t2 = generator.generate_trace(profile, routing_result, exec_result, quality_result, trace_id="fixed-id-1")
    
    assert t1.trace_id == t2.trace_id
    assert t1.decision_summary == t2.decision_summary
