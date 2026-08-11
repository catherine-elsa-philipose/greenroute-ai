import pytest
import uuid
from app.schemas.profile import TaskType, Domain
from app.schemas.quality import QualityLabel
from app.schemas.memory import ExecutionOutcomeRecord
from app.memory.outcome_memory import AdaptiveOutcomeMemory

@pytest.fixture
def memory_db():
    # In-memory SQLite database isolated per test
    return AdaptiveOutcomeMemory(db_path=":memory:")

def test_record_and_retrieve_successful_outcome(memory_db):
    record = ExecutionOutcomeRecord(
        outcome_id=str(uuid.uuid4()),
        task_type=TaskType.CODING,
        domain=Domain.COMPUTER_SCIENCE,
        model_id="coder-v1",
        provider_name="mock",
        routing_score=0.92,
        routing_confidence="high",
        confidence_gap=0.25,
        overall_quality_score=0.88,
        quality_label=QualityLabel.GOOD,
        latency_ms=45.0,
        estimated_cost=0.001,
        success=True,
        is_fallback=False
    )
    
    memory_db.record_outcome(record)
    recent = memory_db.get_recent_outcomes()
    
    assert len(recent) == 1
    assert recent[0].outcome_id == record.outcome_id
    assert recent[0].model_id == "coder-v1"
    assert recent[0].success is True

def test_record_failed_and_fallback_outcomes(memory_db):
    r_fail = ExecutionOutcomeRecord(
        outcome_id=str(uuid.uuid4()),
        task_type=TaskType.REASONING,
        domain=Domain.GENERAL,
        model_id="model-fail",
        provider_name="mock",
        routing_score=0.7,
        routing_confidence="low",
        confidence_gap=0.01,
        overall_quality_score=0.0,
        quality_label=QualityLabel.WEAK,
        latency_ms=120.0,
        estimated_cost=0.002,
        success=False,
        is_fallback=True
    )
    memory_db.record_outcome(r_fail)
    
    stats = memory_db.calculate_model_stats("model-fail")
    assert stats.total_attempts == 1
    assert stats.successful_attempts == 0
    assert stats.success_rate == 0.0
    assert stats.fallback_rate == 1.0

def test_filter_by_model_and_task_domain(memory_db):
    r1 = ExecutionOutcomeRecord(
        outcome_id=str(uuid.uuid4()),
        task_type=TaskType.CODING,
        domain=Domain.COMPUTER_SCIENCE,
        model_id="model-a",
        provider_name="mock",
        routing_score=0.8,
        routing_confidence="high",
        confidence_gap=0.2,
        overall_quality_score=0.9,
        quality_label=QualityLabel.GOOD,
        latency_ms=30.0,
        success=True
    )
    r2 = ExecutionOutcomeRecord(
        outcome_id=str(uuid.uuid4()),
        task_type=TaskType.TRANSLATION,
        domain=Domain.LANGUAGE,
        model_id="model-b",
        provider_name="mock",
        routing_score=0.85,
        routing_confidence="high",
        confidence_gap=0.3,
        overall_quality_score=0.85,
        quality_label=QualityLabel.GOOD,
        latency_ms=25.0,
        success=True
    )
    memory_db.record_outcome(r1)
    memory_db.record_outcome(r2)
    
    outcomes_a = memory_db.get_outcomes_for_model("model-a")
    assert len(outcomes_a) == 1
    assert outcomes_a[0].model_id == "model-a"
    
    coding_outcomes = memory_db.get_outcomes_for_task_and_domain(TaskType.CODING, Domain.COMPUTER_SCIENCE)
    assert len(coding_outcomes) == 1
    assert coding_outcomes[0].task_type == TaskType.CODING

def test_calculate_model_stats_includes_all_outcomes_quality(memory_db):
    # 1 successful outcome (quality 0.8) and 1 failed outcome (quality 0.0)
    r_succ = ExecutionOutcomeRecord(
        outcome_id=str(uuid.uuid4()), task_type=TaskType.CODING, domain=Domain.COMPUTER_SCIENCE,
        model_id="m-mix", provider_name="mock", routing_score=0.8, routing_confidence="high",
        confidence_gap=0.2, overall_quality_score=0.8, quality_label=QualityLabel.GOOD,
        latency_ms=40.0, success=True
    )
    r_fail = ExecutionOutcomeRecord(
        outcome_id=str(uuid.uuid4()), task_type=TaskType.CODING, domain=Domain.COMPUTER_SCIENCE,
        model_id="m-mix", provider_name="mock", routing_score=0.8, routing_confidence="high",
        confidence_gap=0.2, overall_quality_score=0.0, quality_label=QualityLabel.WEAK,
        latency_ms=10.0, success=False
    )
    memory_db.record_outcome(r_succ)
    memory_db.record_outcome(r_fail)
    
    stats = memory_db.calculate_model_stats("m-mix")
    assert stats.total_attempts == 2
    assert stats.successful_attempts == 1
    assert stats.average_quality_score == 0.4  # (0.8 + 0.0) / 2 = 0.4 (accurately includes failed outcome)

def test_empty_history_adaptive_signal(memory_db):
    signal = memory_db.generate_adaptive_signal("model-new")
    assert signal.total_samples == 0
    assert signal.adaptive_weight_multiplier == 1.0  # Neutral baseline multiplier
    assert signal.historical_success_rate is None  # Reported as None / UNKNOWN rather than 1.0
    assert signal.historical_quality_score is None  # Reported as None / UNKNOWN

def test_adaptive_routing_signal_multiplier(memory_db):
    # High quality outcomes raise adaptive_weight_multiplier above 1.0
    r = ExecutionOutcomeRecord(
        outcome_id=str(uuid.uuid4()),
        task_type=TaskType.CODING,
        domain=Domain.COMPUTER_SCIENCE,
        model_id="model-star",
        provider_name="mock",
        routing_score=0.9,
        routing_confidence="high",
        confidence_gap=0.3,
        overall_quality_score=0.95,
        quality_label=QualityLabel.GOOD,
        latency_ms=20.0,
        success=True,
        is_fallback=False
    )
    memory_db.record_outcome(r)
    
    signal = memory_db.generate_adaptive_signal("model-star", TaskType.CODING, Domain.COMPUTER_SCIENCE)
    assert signal.total_samples == 1
    assert signal.historical_success_rate == 1.0
    assert signal.historical_quality_score == 0.95
    assert signal.adaptive_weight_multiplier > 1.0
