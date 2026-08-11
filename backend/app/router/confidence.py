from typing import List, Optional
from app.schemas.routing import (
    CandidateScore,
    ConfidenceGapAnalysis,
    ConfidenceLevel,
    RoutingActionRecommendation
)

class ConfidenceGapEvaluator:
    """
    Semantic Confidence Gap Evaluator (Phase 7).
    
    Evaluates the score margin (gap = top_score - second_score) between the top two available models.
    Provides a routing confidence signal to indicate whether a direct route is clear (HIGH gap)
    or whether candidate models are tightly competing (LOW gap).
    
    Note: Thresholds (high_threshold, medium_threshold) are initial heuristic configuration parameters.
    """

    def __init__(
        self,
        high_threshold: float = 0.15,
        medium_threshold: float = 0.05
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def evaluate(self, candidate_scores: List[CandidateScore]) -> ConfidenceGapAnalysis:
        """
        Analyze candidate scores list and calculate the semantic confidence gap.
        Filter for eligible (available) models.
        """
        eligible_candidates = [c for c in candidate_scores if c.is_available]
        # Ensure candidates are sorted descending by total_score
        eligible_candidates.sort(key=lambda c: c.total_score, reverse=True)

        count = len(eligible_candidates)

        if count == 0:
            return ConfidenceGapAnalysis(
                top_model_id=None,
                top_score=0.0,
                second_model_id=None,
                second_score=None,
                confidence_gap=0.0,
                confidence_level=ConfidenceLevel.LOW,
                eligible_candidate_count=0,
                recommendation=RoutingActionRecommendation.NO_MODEL_AVAILABLE,
                explanation="No eligible available models were present for confidence gap evaluation."
            )

        top_candidate = eligible_candidates[0]

        if count == 1:
            return ConfidenceGapAnalysis(
                top_model_id=top_candidate.model_id,
                top_score=top_candidate.total_score,
                second_model_id=None,
                second_score=None,
                confidence_gap=0.0,
                confidence_level=ConfidenceLevel.HIGH,
                eligible_candidate_count=1,
                recommendation=RoutingActionRecommendation.DIRECT_ROUTE,
                explanation=f"Single eligible model '{top_candidate.model_id}' available. Direct route recommended."
            )

        # Count >= 2
        second_candidate = eligible_candidates[1]
        gap = round(top_candidate.total_score - second_candidate.total_score, 4)

        if gap >= self.high_threshold:
            conf_level = ConfidenceLevel.HIGH
            recommendation = RoutingActionRecommendation.DIRECT_ROUTE
            explanation = (
                f"Clear winner '{top_candidate.model_id}' (score: {top_candidate.total_score:.4f}) "
                f"leads '{second_candidate.model_id}' (score: {second_candidate.total_score:.4f}) by gap {gap:.4f} >= {self.high_threshold}. "
                f"Direct route recommended."
            )
        elif gap >= self.medium_threshold:
            conf_level = ConfidenceLevel.MEDIUM
            recommendation = RoutingActionRecommendation.DIRECT_ROUTE
            explanation = (
                f"Moderate confidence winner '{top_candidate.model_id}' leads by gap {gap:.4f}. Direct route recommended."
            )
        else:
            conf_level = ConfidenceLevel.LOW
            recommendation = RoutingActionRecommendation.COMPARE_OR_ESCALATE
            explanation = (
                f"Tight competition between '{top_candidate.model_id}' ({top_candidate.total_score:.4f}) "
                f"and '{second_candidate.model_id}' ({second_candidate.total_score:.4f}) with gap {gap:.4f} < {self.medium_threshold}. "
                f"Multi-model comparison/escalation recommended."
            )

        return ConfidenceGapAnalysis(
            top_model_id=top_candidate.model_id,
            top_score=top_candidate.total_score,
            second_model_id=second_candidate.model_id,
            second_score=second_candidate.total_score,
            confidence_gap=gap,
            confidence_level=conf_level,
            eligible_candidate_count=count,
            recommendation=recommendation,
            explanation=explanation
        )
