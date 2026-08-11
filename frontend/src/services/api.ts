// GreenRoute API Data Contracts

export interface PromptProfile {
  prompt: string;
  task_type: string;
  domain: string;
  complexity: string;
  reasoning_need: string;
  expected_output: string;
  language: string;
}

export interface CandidateScore {
  model_id: string;
  provider_name: string;
  is_available: boolean;
  capability_fit_score: number;
  operational_score: number;
  total_score: number;
  scoring_details: Record<string, number>;
}

export interface ConfidenceGapAnalysis {
  top_model_id: string;
  top_score: number;
  second_model_id: string;
  second_score: number;
  confidence_gap: number;
  confidence_level: string;
  eligible_candidate_count: number;
  recommendation: string;
  explanation: string;
}

export interface RoutingResult {
  selected_model_id?: string;
  selected_provider_name?: string;
  top_score: number;
  candidate_scores: CandidateScore[];
  no_available_model: boolean;
  confidence_gap_analysis?: ConfidenceGapAnalysis;
  status_message: string;
}

export interface ModelExecutionResult {
  model_id: string;
  provider_name: string;
  success: boolean;
  content?: string;
  error_message?: string;
  latency_ms: number;
  raw_response?: any;
}

export interface MultiModelExecutionResult {
  prompt: string;
  execution_mode: string;
  overall_success: boolean;
  executions: ModelExecutionResult[];
  status_message: string;
}

export interface ResponseQualityEvaluation {
  model_id: string;
  provider_name: string;
  relevance_score: number;
  completeness_score: number;
  compliance_score: number;
  consistency_score: number;
  overall_quality_score: number;
  quality_label: string;
  evaluation_notes: string;
}

export interface QualityGateResult {
  evaluations: ResponseQualityEvaluation[];
  selected_model_id?: string;
  selected_content?: string;
  selected_quality_score: number;
  is_weak_response: boolean;
  recommendation: string;
  status_message: string;
}

export interface AdaptiveRoutingSignal {
  model_id: string;
  task_type?: string;
  domain?: string;
  historical_quality_score?: number;
  historical_success_rate?: number;
  historical_fallback_rate: number;
  adaptive_weight_multiplier: number;
  total_samples: number;
}

export interface ExplainableDecisionTrace {
  trace_id: string;
  timestamp: string;
  profile?: PromptProfile;
  routing_result?: RoutingResult;
  execution_result?: MultiModelExecutionResult;
  quality_result?: QualityGateResult;
  adaptive_signal?: AdaptiveRoutingSignal;
  final_selected_model_id?: string;
  final_selected_content?: string;
  decision_summary: string;
}

export interface RouteResponse {
  status: string;
  prompt: string;
  profile?: PromptProfile;
  routing_result?: RoutingResult;
  confidence_gap_analysis?: ConfidenceGapAnalysis;
  execution_result?: MultiModelExecutionResult;
  quality_result?: QualityGateResult;
  adaptive_signal?: AdaptiveRoutingSignal;
  trace?: ExplainableDecisionTrace;
  final_selected_model?: string;
  final_selected_content?: string;
  error_message?: string;
}

const API_BASE_URL = 'http://localhost:8020/api/v1';

export async function analyzePrompt(prompt: string): Promise<RouteResponse> {
  const response = await fetch(`${API_BASE_URL}/route`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ prompt }),
  });

  if (!response.ok) {
    let errorDetail = 'An unknown error occurred';
    try {
      const errData = await response.json();
      errorDetail = errData.detail || errorDetail;
    } catch {
      errorDetail = response.statusText || errorDetail;
    }
    throw new Error(errorDetail);
  }

  return response.json();
}
