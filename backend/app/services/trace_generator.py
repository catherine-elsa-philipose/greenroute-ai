import uuid
from typing import Optional
from app.schemas.profile import PromptProfile
from app.schemas.routing import RoutingResult
from app.schemas.execution import MultiModelExecutionResult
from app.schemas.quality import QualityGateResult
from app.schemas.memory import AdaptiveRoutingSignal
from app.schemas.trace import ExplainableDecisionTrace

class DecisionTraceGenerator:
    """
    Explainable Decision Trace Generator (Phase 11).
    
    Synthesizes facts from all phases of the GreenRoute canonical pipeline into a structured,
    transparent, and human-readable decision trace.
    Does NOT perform routing or provider execution itself; explains facts produced by other modules.
    """

    def generate_trace(
        self,
        profile: PromptProfile,
        routing_result: RoutingResult,
        execution_result: MultiModelExecutionResult,
        quality_result: QualityGateResult,
        adaptive_signal: Optional[AdaptiveRoutingSignal] = None,
        trace_id: Optional[str] = None
    ) -> ExplainableDecisionTrace:
        """
        Generate structured ExplainableDecisionTrace from pipeline outputs.
        """
        t_id = trace_id or str(uuid.uuid4())
        gap_analysis = routing_result.confidence_gap_analysis

        # Build transparent human-readable explanation text
        explanation = self._build_human_explanation(
            profile=profile,
            routing_result=routing_result,
            execution_result=execution_result,
            quality_result=quality_result,
            adaptive_signal=adaptive_signal
        )

        return ExplainableDecisionTrace(
            trace_id=t_id,
            profile=profile,
            routing_result=routing_result,
            confidence_gap_analysis=gap_analysis,
            execution_result=execution_result,
            quality_result=quality_result,
            adaptive_signal=adaptive_signal,
            final_selected_model_id=quality_result.selected_model_id,
            final_selected_content=quality_result.selected_content,
            decision_summary=explanation
        )

    def _build_human_explanation(
        self,
        profile: PromptProfile,
        routing_result: RoutingResult,
        execution_result: MultiModelExecutionResult,
        quality_result: QualityGateResult,
        adaptive_signal: Optional[AdaptiveRoutingSignal]
    ) -> str:
        parts = []

        # 1. Profile summary
        parts.append(
            f"Prompt was profiled as task '{profile.task_type.value}' in domain '{profile.domain.value}' "
            f"with {profile.complexity.value} complexity and {profile.reasoning_need.value} reasoning need."
        )

        # 2. Routing summary
        if routing_result.no_available_model or not routing_result.selected_model_id:
            parts.append("Routing failed because no registered models were available.")
            return " ".join(parts)

        top_model = routing_result.selected_model_id
        top_score = routing_result.top_score

        gap_analysis = routing_result.confidence_gap_analysis
        gap_val = gap_analysis.confidence_gap if gap_analysis else 0.0
        conf_level = gap_analysis.confidence_level.value if gap_analysis else "unknown"

        parts.append(
            f"Based on configured capability metadata, model '{top_model}' received the highest routing score ({top_score:.4f}). "
            f"The confidence gap was {gap_val:.4f} ({conf_level.upper()} confidence)."
        )

        # 3. Execution & Multi-model mode summary
        exec_mode = execution_result.execution_mode.value
        exec_count = len(execution_result.executions)
        succ_count = sum(1 for e in execution_result.executions if e.success)
        failed_count = exec_count - succ_count

        if exec_count > 1:
            parts.append(
                f"Due to '{exec_mode}' mode, top {exec_count} candidate models were executed. "
                f"{succ_count} succeeded, {failed_count} failed."
            )
        else:
            parts.append(f"Model '{top_model}' was directly executed.")

        if failed_count > 0:
            failed_models = [e.model_id for e in execution_result.executions if not e.success]
            parts.append(f"Provider execution failure occurred for model(s): {', '.join(failed_models)}.")

        # 4. Quality evaluation summary
        if quality_result.selected_model_id:
            sel_qual = quality_result.selected_quality_score
            parts.append(
                f"Response Quality Gate evaluated candidate output(s) and selected model '{quality_result.selected_model_id}' "
                f"with an overall quality score of {sel_qual:.4f}."
            )
            if quality_result.is_weak_response:
                parts.append(
                    "WARNING: The selected response fell below acceptable quality threshold and is flagged as WEAK. "
                    "Fallback or re-evaluation is recommended."
                )
        else:
            parts.append("Response Quality Gate reported no eligible model outputs.")

        # 5. Historical Memory Signal summary
        if adaptive_signal and adaptive_signal.total_samples > 0:
            q_hist = (
                f"{adaptive_signal.historical_quality_score:.4f}"
                if adaptive_signal.historical_quality_score is not None else "N/A"
            )
            s_hist = (
                f"{adaptive_signal.historical_success_rate:.4f}"
                if adaptive_signal.historical_success_rate is not None else "N/A"
            )
            parts.append(
                f"Historical memory recorded {adaptive_signal.total_samples} prior outcome sample(s) for model '{top_model}' "
                f"(Avg Quality: {q_hist}, Success Rate: {s_hist}, Adaptive Multiplier: {adaptive_signal.adaptive_weight_multiplier:.4f})."
            )
        else:
            parts.append(f"No prior historical outcomes were recorded for model '{top_model}'. Adaptive multiplier remains neutral (1.0).")

        return " ".join(parts)
