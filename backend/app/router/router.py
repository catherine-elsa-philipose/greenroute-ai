from typing import List, Dict
from app.schemas.profile import PromptProfile, TaskType, Domain
from app.schemas.provider import ModelProfile
from app.schemas.routing import CandidateScore, RoutingResult
from app.registry.capability_registry import CapabilityRegistry
from app.router.confidence import ConfidenceGapEvaluator

class ConfidenceAwareRouter:
    """
    Confidence-Aware Router Engine (Phase 6 & 7).
    
    Calculates suitability scores for available candidate models registered in CapabilityRegistry
    by comparing PromptProfile requirements against ModelCapabilities metadata, and performs
    Phase 7 Semantic Confidence Gap Analysis.
    """

    def __init__(
        self,
        capability_weight: float = 0.6,
        operational_weight: float = 0.4,
        confidence_evaluator: ConfidenceGapEvaluator = None
    ):
        self.capability_weight = capability_weight
        self.operational_weight = operational_weight
        self.confidence_evaluator = confidence_evaluator or ConfidenceGapEvaluator()

    def route(self, profile: PromptProfile, registry: CapabilityRegistry) -> RoutingResult:
        """
        Evaluate candidate models in registry against prompt profile, compute confidence gap,
        and select the highest scoring available model.
        """
        all_models: List[ModelProfile] = registry.list_models()

        if not all_models:
            gap_analysis = self.confidence_evaluator.evaluate([])
            return RoutingResult(
                no_available_model=True,
                confidence_gap_analysis=gap_analysis,
                status_message="No models are registered in the CapabilityRegistry."
            )

        candidate_scores: List[CandidateScore] = []
        available_candidates: List[CandidateScore] = []

        for model in all_models:
            score_entry = self._evaluate_model(profile, model)
            candidate_scores.append(score_entry)
            if model.is_available:
                available_candidates.append(score_entry)

        gap_analysis = self.confidence_evaluator.evaluate(candidate_scores)

        if not available_candidates:
            return RoutingResult(
                candidate_scores=candidate_scores,
                no_available_model=True,
                confidence_gap_analysis=gap_analysis,
                status_message="No registered models are currently available."
            )

        # Sort available candidates descending by total_score
        available_candidates.sort(key=lambda c: c.total_score, reverse=True)
        winner = available_candidates[0]

        return RoutingResult(
            selected_model_id=winner.model_id,
            selected_provider_name=winner.provider_name,
            top_score=winner.total_score,
            candidate_scores=candidate_scores,
            no_available_model=False,
            confidence_gap_analysis=gap_analysis,
            status_message=f"Selected model '{winner.model_id}' with top score {winner.total_score:.4f} (Gap: {gap_analysis.confidence_gap:.4f})."
        )

    def _evaluate_model(self, profile: PromptProfile, model: ModelProfile) -> CandidateScore:
        caps = model.capabilities

        # 1. Capability Fit Score calculation
        task_score = 0.5  # base neutral fit
        if profile.domain == Domain.MATHEMATICS:
            task_score = caps.math_strength
        elif profile.task_type == TaskType.CODING or profile.domain == Domain.COMPUTER_SCIENCE:
            task_score = caps.coding_strength
        elif profile.task_type == TaskType.REASONING:
            task_score = caps.reasoning_strength
        elif profile.task_type == TaskType.TRANSLATION or profile.domain == Domain.LANGUAGE:
            task_score = caps.multilingual_strength
        else:
            task_score = (caps.coding_strength + caps.reasoning_strength) / 2.0

        # 2. Operational Score calculation
        # Cost factor normalized: $0.00/1k tokens = 1.0 score; $0.03/1k tokens = ~0.0 score
        cost_score = max(0.0, 1.0 - (caps.estimated_cost_per_1k_tokens / 0.03))
        
        op_score = (caps.speed_score * 0.4) + (caps.reliability_score * 0.4) + (cost_score * 0.2)

        # 3. Final Total Weighted Score
        total_score = (task_score * self.capability_weight) + (op_score * self.operational_weight)

        return CandidateScore(
            model_id=model.model_id,
            provider_name=model.provider_name,
            is_available=model.is_available,
            capability_fit_score=round(task_score, 4),
            operational_score=round(op_score, 4),
            total_score=round(total_score, 4),
            scoring_details={
                "task_score": round(task_score, 4),
                "speed_score": round(caps.speed_score, 4),
                "reliability_score": round(caps.reliability_score, 4),
                "cost_score": round(cost_score, 4)
            }
        )
