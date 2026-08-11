from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict
from app.registry.base import BaseLLMProvider

from app.schemas.api import RouteRequest, RouteResponse
from app.schemas.memory import ExecutionOutcomeRecord
from app.api import deps
from app.profiler.profiler import PromptProfiler
from app.registry.capability_registry import CapabilityRegistry
from app.router.router import ConfidenceAwareRouter
from app.services.orchestrator import MultiModelOrchestrator
from app.quality.quality_gate import ResponseQualityGate
from app.memory.outcome_memory import AdaptiveOutcomeMemory
from app.services.trace_generator import DecisionTraceGenerator

import uuid
import time

router = APIRouter()

@router.post("/route", response_model=RouteResponse)
def execute_greenroute(
    request: RouteRequest,
    profiler: PromptProfiler = Depends(deps.get_profiler),
    registry: CapabilityRegistry = Depends(deps.get_registry),
    providers: Dict[str, BaseLLMProvider] = Depends(deps.get_providers),
    confidence_router: ConfidenceAwareRouter = Depends(deps.get_router),
    orchestrator: MultiModelOrchestrator = Depends(deps.get_orchestrator),
    quality_gate: ResponseQualityGate = Depends(deps.get_quality_gate),
    memory: AdaptiveOutcomeMemory = Depends(deps.get_memory),
    trace_generator: DecisionTraceGenerator = Depends(deps.get_trace_generator)
):
    """
    Execute the full GreenRoute canonical loop for a given prompt:
    PROFILE -> ROUTE -> EXECUTE -> EVALUATE -> LEARN
    """
    # 1. Validate prompt
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        # 2. Profile the prompt
        profile = profiler.analyze(prompt)

        # 3 & 4. Route the prompt (includes Semantic Confidence Gap evaluation)
        routing_result = confidence_router.route(profile, registry)
        
        if routing_result.no_available_model or not routing_result.selected_model_id:
            raise HTTPException(status_code=400, detail="No available models matched the prompt profile")
            
        gap_analysis = routing_result.confidence_gap_analysis

        # 5. Get historical adaptive signal (Phase 10)
        # Note: In a production system, we'd iterate over candidates or get the top candidate's signal
        adaptive_signal = memory.generate_adaptive_signal(
            model_id=routing_result.selected_model_id,
            task_type=profile.task_type,
            domain=profile.domain
        )

        # 6. Execute candidates using MultiModelOrchestrator
        execution_result = orchestrator.execute(prompt, routing_result, providers)
        
        # Note: if there's a complete failure of provider execution, we still evaluate it (or skip quality gate)
        # but execution_result will record the failure.

        # 7. Evaluate response quality
        quality_result = quality_gate.evaluate_execution(profile, execution_result)

        # 8. Record outcome to Adaptive Outcome Memory
        if quality_result.selected_model_id:
            # Find the latency and cost of the selected execution
            selected_execution = next(
                (e for e in execution_result.executions if e.model_id == quality_result.selected_model_id), 
                None
            )
            latency_ms = selected_execution.latency_ms if selected_execution else 0.0
            
            outcome_record = ExecutionOutcomeRecord(
                outcome_id=str(uuid.uuid4()),
                task_type=profile.task_type,
                domain=profile.domain,
                model_id=quality_result.selected_model_id,
                provider_name=selected_execution.provider_name if selected_execution else "unknown",
                routing_score=routing_result.top_score,
                routing_confidence=gap_analysis.confidence_level.value if gap_analysis else "low",
                confidence_gap=gap_analysis.confidence_gap if gap_analysis else 0.0,
                overall_quality_score=quality_result.selected_quality_score,
                quality_label=quality_result.evaluations[0].quality_label if quality_result.evaluations else None,
                latency_ms=latency_ms,
                estimated_cost=0.0, # Placeholder
                success=execution_result.overall_success,
                is_fallback=quality_result.is_weak_response
            )
            memory.record_outcome(outcome_record)

        # 9. Generate Explainable Decision Trace
        trace = trace_generator.generate_trace(
            profile=profile,
            routing_result=routing_result,
            execution_result=execution_result,
            quality_result=quality_result,
            adaptive_signal=adaptive_signal
        )

        # 10. Return Structured Result
        return RouteResponse(
            prompt=prompt,
            profile=profile,
            routing_result=routing_result,
            confidence_gap_analysis=gap_analysis,
            execution_result=execution_result,
            quality_result=quality_result,
            adaptive_signal=adaptive_signal,
            trace=trace,
            final_selected_model=quality_result.selected_model_id,
            final_selected_content=quality_result.selected_content,
            status="success" if execution_result.overall_success else "provider_error"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Catch unforeseen orchestrator/provider logic failures
        # Do not expose raw tracebacks, but provide a clean 500 error
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
