import time
from typing import Dict, List, Optional
from app.registry.base import BaseLLMProvider
from app.schemas.routing import RoutingResult, RoutingActionRecommendation
from app.schemas.execution import (
    ExecutionMode,
    ModelExecutionResult,
    MultiModelExecutionResult
)

class MultiModelOrchestrator:
    """
    Multi-Model Execution & Orchestration Service (Phase 8).
    
    Receives prompt, RoutingResult, and a map of instantiated provider adapters (Dict[model_id, BaseLLMProvider]).
    Executes the designated provider adapter(s) based on recommendation:
    - DIRECT_ROUTE: Executes the single top candidate.
    - COMPARE_OR_ESCALATE: Executes top candidate models for multi-model comparison.
    
    Robustness: Handles provider exceptions gracefully, recording failures without crashing orchestration.
    """

    def execute(
        self,
        prompt: str,
        routing_result: RoutingResult,
        providers: Dict[str, BaseLLMProvider],
        top_k_candidates: int = 2
    ) -> MultiModelExecutionResult:
        """
        Execute provider(s) based on routing decision.
        """
        if routing_result.no_available_model or not routing_result.selected_model_id:
            return MultiModelExecutionResult(
                prompt=prompt,
                execution_mode=ExecutionMode.DIRECT_ROUTE,
                overall_success=False,
                executions=[],
                status_message="Orchestration aborted: No eligible model available in routing result."
            )

        gap_analysis = routing_result.confidence_gap_analysis
        is_compare_mode = (
            gap_analysis is not None and
            gap_analysis.recommendation == RoutingActionRecommendation.COMPARE_OR_ESCALATE
        )

        mode = ExecutionMode.COMPARE_OR_ESCALATE if is_compare_mode else ExecutionMode.DIRECT_ROUTE

        # Target candidate model IDs to execute
        target_model_ids: List[str] = []
        if mode == ExecutionMode.DIRECT_ROUTE:
            target_model_ids = [routing_result.selected_model_id]
        else:
            # Select top K available candidate models from routing result
            available_scores = [c for c in routing_result.candidate_scores if c.is_available]
            available_scores.sort(key=lambda c: c.total_score, reverse=True)
            target_model_ids = [c.model_id for c in available_scores[:top_k_candidates]]

        executions: List[ModelExecutionResult] = []

        for model_id in target_model_ids:
            provider = providers.get(model_id)
            if not provider:
                executions.append(
                    ModelExecutionResult(
                        model_id=model_id,
                        provider_name="unknown",
                        success=False,
                        error_message=f"Provider adapter for model_id '{model_id}' was not supplied to orchestrator.",
                        latency_ms=0.0
                    )
                )
                continue

            # Execute provider with local latency measurement
            start_time = time.perf_counter()
            try:
                provider_response = provider.generate(prompt)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                
                executions.append(
                    ModelExecutionResult(
                        model_id=model_id,
                        provider_name=provider.provider_name,
                        success=True,
                        content=provider_response.content,
                        latency_ms=round(elapsed_ms, 2),
                        raw_response=provider_response.raw_response
                    )
                )
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                executions.append(
                    ModelExecutionResult(
                        model_id=model_id,
                        provider_name=provider.provider_name,
                        success=False,
                        error_message=str(e),
                        latency_ms=round(elapsed_ms, 2)
                    )
                )

        any_success = any(e.success for e in executions)
        status_msg = f"Orchestration completed in {mode.value} mode. Successes: {sum(1 for e in executions if e.success)}/{len(executions)}."

        return MultiModelExecutionResult(
            prompt=prompt,
            execution_mode=mode,
            overall_success=any_success,
            executions=executions,
            status_message=status_msg
        )
