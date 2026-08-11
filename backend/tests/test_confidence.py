import pytest
from app.schemas.routing import CandidateScore, ConfidenceLevel, RoutingActionRecommendation
from app.router.confidence import ConfidenceGapEvaluator

def test_large_confidence_gap():
    evaluator = ConfidenceGapEvaluator(high_threshold=0.15, medium_threshold=0.05)
    candidates = [
        CandidateScore(model_id="model-a", provider_name="p", is_available=True, capability_fit_score=0.9, operational_score=0.92, total_score=0.91),
        CandidateScore(model_id="model-b", provider_name="p", is_available=True, capability_fit_score=0.6, operational_score=0.60, total_score=0.60),
    ]
    
    result = evaluator.evaluate(candidates)
    assert result.confidence_gap == 0.31
    assert result.confidence_level == ConfidenceLevel.HIGH
    assert result.recommendation == RoutingActionRecommendation.DIRECT_ROUTE
    assert result.top_model_id == "model-a"
    assert result.second_model_id == "model-b"

def test_small_confidence_gap():
    evaluator = ConfidenceGapEvaluator(high_threshold=0.15, medium_threshold=0.05)
    candidates = [
        CandidateScore(model_id="model-a", provider_name="p", is_available=True, capability_fit_score=0.84, operational_score=0.84, total_score=0.84),
        CandidateScore(model_id="model-b", provider_name="p", is_available=True, capability_fit_score=0.82, operational_score=0.82, total_score=0.82),
    ]
    
    result = evaluator.evaluate(candidates)
    assert result.confidence_gap == 0.02
    assert result.confidence_level == ConfidenceLevel.LOW
    assert result.recommendation == RoutingActionRecommendation.COMPARE_OR_ESCALATE

def test_more_than_two_candidates():
    evaluator = ConfidenceGapEvaluator()
    candidates = [
        CandidateScore(model_id="model-a", provider_name="p", is_available=True, capability_fit_score=0.90, operational_score=0.90, total_score=0.90),
        CandidateScore(model_id="model-b", provider_name="p", is_available=True, capability_fit_score=0.70, operational_score=0.70, total_score=0.70),
        CandidateScore(model_id="model-c", provider_name="p", is_available=True, capability_fit_score=0.50, operational_score=0.50, total_score=0.50),
    ]
    
    result = evaluator.evaluate(candidates)
    assert result.eligible_candidate_count == 3
    assert result.top_model_id == "model-a"
    assert result.second_model_id == "model-b"
    assert result.confidence_gap == 0.20

def test_only_one_eligible_candidate():
    evaluator = ConfidenceGapEvaluator()
    candidates = [
        CandidateScore(model_id="model-solo", provider_name="p", is_available=True, capability_fit_score=0.85, operational_score=0.85, total_score=0.85),
        CandidateScore(model_id="model-offline", provider_name="p", is_available=False, capability_fit_score=0.99, operational_score=0.99, total_score=0.99),
    ]
    
    result = evaluator.evaluate(candidates)
    assert result.eligible_candidate_count == 1
    assert result.top_model_id == "model-solo"
    assert result.second_model_id is None
    assert result.second_score is None
    assert result.confidence_gap == 0.0
    assert result.confidence_level == ConfidenceLevel.HIGH

def test_zero_eligible_candidates():
    evaluator = ConfidenceGapEvaluator()
    candidates = [
        CandidateScore(model_id="offline-1", provider_name="p", is_available=False, capability_fit_score=0.9, operational_score=0.9, total_score=0.9),
    ]
    
    result = evaluator.evaluate(candidates)
    assert result.eligible_candidate_count == 0
    assert result.top_model_id is None
    assert result.recommendation == RoutingActionRecommendation.NO_MODEL_AVAILABLE

def test_configurable_threshold_boundaries():
    evaluator = ConfidenceGapEvaluator(high_threshold=0.20, medium_threshold=0.10)
    
    # Gap = 0.12 (between 0.10 and 0.20 -> MEDIUM)
    c1 = [
        CandidateScore(model_id="m1", provider_name="p", is_available=True, capability_fit_score=0.80, operational_score=0.80, total_score=0.82),
        CandidateScore(model_id="m2", provider_name="p", is_available=True, capability_fit_score=0.70, operational_score=0.70, total_score=0.70),
    ]
    res1 = evaluator.evaluate(c1)
    assert res1.confidence_level == ConfidenceLevel.MEDIUM
    assert res1.recommendation == RoutingActionRecommendation.DIRECT_ROUTE

def test_deterministic_repeated_evaluations():
    evaluator = ConfidenceGapEvaluator()
    candidates = [
        CandidateScore(model_id="m1", provider_name="p", is_available=True, capability_fit_score=0.85, operational_score=0.85, total_score=0.85),
        CandidateScore(model_id="m2", provider_name="p", is_available=True, capability_fit_score=0.80, operational_score=0.80, total_score=0.80),
    ]
    res1 = evaluator.evaluate(candidates)
    res2 = evaluator.evaluate(candidates)
    assert res1.confidence_gap == res2.confidence_gap
    assert res1.confidence_level == res2.confidence_level
