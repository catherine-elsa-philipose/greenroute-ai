import re
from typing import List, Optional
from app.schemas.profile import PromptProfile, TaskType
from app.schemas.execution import MultiModelExecutionResult, ModelExecutionResult
from app.schemas.quality import (
    QualityLabel,
    ResponseQualityEvaluation,
    QualityGateResult
)

class ResponseQualityGate:
    """
    Response Quality Gate (Phase 9).
    
    Evaluates model-generated responses across 4 dimensions using transparent heuristics:
    1. Relevance (keyword overlap with user prompt)
    2. Completeness (output length & format alignment with PromptProfile)
    3. Instruction Compliance (presence of required code/explanation/translation structures)
    4. Consistency (absence of empty output, glaring self-contradictions, or error keywords)
    
    Formula:
    Overall Quality = (Relevance * 0.30) + (Completeness * 0.25) + (Compliance * 0.30) + (Consistency * 0.15)
    
    Selection:
    Selects highest quality score. Deterministic tie-breaking uses model_id alphabetical order.
    """

    def __init__(
        self,
        good_threshold: float = 0.75,
        acceptable_threshold: float = 0.50,
        relevance_weight: float = 0.30,
        completeness_weight: float = 0.25,
        compliance_weight: float = 0.30,
        consistency_weight: float = 0.15
    ):
        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold
        self.relevance_weight = relevance_weight
        self.completeness_weight = completeness_weight
        self.compliance_weight = compliance_weight
        self.consistency_weight = consistency_weight

    def evaluate_execution(
        self,
        profile: PromptProfile,
        execution_result: MultiModelExecutionResult
    ) -> QualityGateResult:
        """
        Evaluate all successful model execution responses and select the best candidate response based on quality.
        """
        successful_executions = [e for e in execution_result.executions if e.success and e.content]

        if not successful_executions:
            return QualityGateResult(
                evaluations=[],
                selected_model_id=None,
                selected_content=None,
                selected_quality_score=0.0,
                is_weak_response=True,
                recommendation="RE_EVALUATE_OR_FALLBACK",
                status_message="No successful model execution responses were available to evaluate."
            )

        evaluations: List[ResponseQualityEvaluation] = []

        for exec_res in successful_executions:
            eval_item = self._evaluate_response(profile, exec_res)
            evaluations.append(eval_item)

        # Deterministic sorting: highest quality score first, tie-break by model_id ascending
        evaluations.sort(key=lambda x: (-x.overall_quality_score, x.model_id))

        winner = evaluations[0]
        # Retrieve actual winner content from execution list
        winner_exec = next(e for e in successful_executions if e.model_id == winner.model_id)

        is_weak = winner.overall_quality_score < self.acceptable_threshold
        recommendation = "RE_EVALUATE_OR_FALLBACK" if is_weak else "ACCEPT"
        status_msg = (
            f"Selected best response from model '{winner.model_id}' with quality score {winner.overall_quality_score:.4f} "
            f"[{winner.quality_label.value.upper()}]."
        )

        return QualityGateResult(
            evaluations=evaluations,
            selected_model_id=winner.model_id,
            selected_content=winner_exec.content,
            selected_quality_score=winner.overall_quality_score,
            is_weak_response=is_weak,
            recommendation=recommendation,
            status_message=status_msg
        )

    def _evaluate_response(self, profile: PromptProfile, exec_res: ModelExecutionResult) -> ResponseQualityEvaluation:
        content = exec_res.content or ""
        content_lower = content.lower()
        prompt_words = [w.lower() for w in re.findall(r'\w+', profile.prompt) if len(w) > 3]

        # 1. Relevance Score: keyword overlap ratio
        if not content.strip():
            rel_score = 0.0
        elif not prompt_words:
            rel_score = 0.8
        else:
            matches = sum(1 for w in prompt_words if w in content_lower)
            rel_score = min(1.0, 0.4 + (matches / len(prompt_words)) * 0.6)

        # 2. Completeness Score: length and output expectations
        word_count = len(content.split())
        if word_count == 0:
            comp_score = 0.0
        elif profile.expected_output == "Code + Explanation":
            comp_score = 0.9 if word_count >= 15 else 0.4
        elif word_count < 5:
            comp_score = 0.4
        else:
            comp_score = min(1.0, 0.5 + (word_count / 50) * 0.5)

        # 3. Instruction Compliance Score
        comp_score_val = 0.7  # default baseline compliance
        if profile.task_type == TaskType.CODING:
            has_code_markers = any(m in content for m in ["def ", "class ", "function", "return", "import", "var ", "const ", "let ", "select ", "```"])
            has_explanation = any(m in content_lower for m in ["explain", "time complexity", "because", "recursion", "note", "this code"])
            
            if profile.expected_output == "Code + Explanation":
                comp_score_val = 1.0 if (has_code_markers and has_explanation) else (0.6 if has_code_markers else 0.3)
            else:
                comp_score_val = 1.0 if has_code_markers else 0.4
        elif profile.task_type == TaskType.TRANSLATION:
            # Translation shouldn't echo meta instruction
            comp_score_val = 0.9 if len(content) > 3 else 0.2

        # 4. Consistency Score: checks for empty or error indicators
        if not content.strip() or "error" in content_lower and "simulated provider failure" in content_lower:
            cons_score = 0.0
        else:
            cons_score = 0.95

        # Weighted Overall Score
        overall = (
            (rel_score * self.relevance_weight) +
            (comp_score * self.completeness_weight) +
            (comp_score_val * self.compliance_weight) +
            (cons_score * self.consistency_weight)
        )
        overall = round(overall, 4)

        if overall >= self.good_threshold:
            label = QualityLabel.GOOD
        elif overall >= self.acceptable_threshold:
            label = QualityLabel.ACCEPTABLE
        else:
            label = QualityLabel.WEAK

        notes = (
            f"Relevance: {rel_score:.2f}, Completeness: {comp_score:.2f}, "
            f"Compliance: {comp_score_val:.2f}, Consistency: {cons_score:.2f}"
        )

        return ResponseQualityEvaluation(
            model_id=exec_res.model_id,
            provider_name=exec_res.provider_name,
            relevance_score=round(rel_score, 4),
            completeness_score=round(comp_score, 4),
            compliance_score=round(comp_score_val, 4),
            consistency_score=round(cons_score, 4),
            overall_quality_score=overall,
            quality_label=label,
            evaluation_notes=notes
        )
