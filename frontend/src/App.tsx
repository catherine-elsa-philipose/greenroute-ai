import { useState, useEffect } from 'react'
import './App.css'
import { analyzePrompt, type RouteResponse } from './services/api'

type UIState = 'idle' | 'loading' | 'success' | 'error'

const LOADING_MESSAGES = [
  "Analyzing prompt...",
  "Evaluating model capabilities...",
  "Generating routing decision...",
  "Executing selected models...",
  "Running quality gate analysis...",
  "Finalizing explainable trace..."
]

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [uiState, setUiState] = useState<UIState>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  const [result, setResult] = useState<RouteResponse | null>(null)
  
  const [loadingMsgIdx, setLoadingMsgIdx] = useState(0)

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (uiState === 'loading') {
      interval = setInterval(() => {
        setLoadingMsgIdx((prev) => (prev + 1) % LOADING_MESSAGES.length)
      }, 1500)
    } else {
      setLoadingMsgIdx(0)
    }
    return () => clearInterval(interval)
  }, [uiState])

  const handleRoute = async () => {
    if (!prompt.trim()) return
    
    setUiState('loading')
    setErrorMessage('')
    setResult(null)

    try {
      const data = await analyzePrompt(prompt)
      setResult(data)
      setUiState('success')
    } catch (err: any) {
      setErrorMessage(err.message || 'An unexpected error occurred')
      setUiState('error')
    }
  }

  const renderProfilePanel = () => {
    if (!result?.profile) return null;
    const { profile } = result;
    return (
      <div className="section-card">
        <h3 className="section-title">Prompt Intelligence Profile</h3>
        <div className="badge-grid">
          <div className="badge-item">
            <span className="badge-label">Task Type</span>
            <span className="badge-value">{profile.task_type}</span>
          </div>
          <div className="badge-item">
            <span className="badge-label">Domain</span>
            <span className="badge-value">{profile.domain}</span>
          </div>
          <div className="badge-item">
            <span className="badge-label">Complexity</span>
            <span className="badge-value">{profile.complexity}</span>
          </div>
          <div className="badge-item">
            <span className="badge-label">Reasoning</span>
            <span className="badge-value">{profile.reasoning_need}</span>
          </div>
          <div className="badge-item">
            <span className="badge-label">Expected Output</span>
            <span className="badge-value">{profile.expected_output || 'Neutral'}</span>
          </div>
          {profile.language && profile.language !== 'unknown' && (
            <div className="badge-item">
              <span className="badge-label">Language</span>
              <span className="badge-value">{profile.language}</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  const renderRoutingPanel = () => {
    if (!result?.routing_result) return null;
    const { routing_result, confidence_gap_analysis } = result;
    return (
      <div className="section-card section-card-highlight">
        <h3 className="section-title">Routing Decision</h3>
        <div className="routing-highlight-grid">
          <div className="highlight-box">
            <div className="badge-label">Selected Model</div>
            <div className="badge-value">{routing_result.selected_model_id || 'None'}</div>
          </div>
          <div className="highlight-box">
            <div className="badge-label">Confidence</div>
            <div className="badge-value">{confidence_gap_analysis?.confidence_level || 'Unknown'}</div>
          </div>
          <div className="highlight-box">
            <div className="badge-label">Routing Action</div>
            <div className="badge-value">{confidence_gap_analysis?.recommendation || 'Unknown'}</div>
          </div>
          <div className="highlight-box neutral">
            <div className="badge-label">Confidence Gap</div>
            <div className="badge-value">{confidence_gap_analysis?.confidence_gap?.toFixed(4) ?? 'N/A'}</div>
          </div>
        </div>
      </div>
    );
  }

  const renderCandidatesPanel = () => {
    if (!result?.routing_result?.candidate_scores) return null;
    const candidates = [...result.routing_result.candidate_scores].sort((a, b) => b.total_score - a.total_score);
    
    return (
      <div className="section-card">
        <h3 className="section-title">Candidate Models</h3>
        <div className="score-list">
          {candidates.map(c => {
            const pct = Math.max(0, Math.min(100, c.total_score * 100));
            return (
              <div key={c.model_id} className="score-row">
                <div className="score-label">{c.model_id}</div>
                <div className="score-bar-container">
                  <div 
                    className={`score-bar-fill ${c.total_score > 0.8 ? 'high' : ''}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="score-value">{c.total_score.toFixed(4)}</div>
              </div>
            )
          })}
        </div>
      </div>
    );
  }

  const renderExecutionPanel = () => {
    if (!result?.execution_result) return null;
    const { execution_result } = result;
    return (
      <div className="section-card">
        <h3 className="section-title">Model Execution ({execution_result.execution_mode})</h3>
        <div className="micro-card-grid">
          {execution_result.executions.map((e, idx) => (
            <div key={idx} className={`micro-card ${e.success ? 'success' : 'error'}`}>
              <div className="micro-card-header">
                <span className="micro-card-title">{e.model_id}</span>
                <span className={`status-badge ${e.success ? 'success' : 'error'}`}>
                  {e.success ? 'Success' : 'Failed'}
                </span>
              </div>
              <div className="micro-card-stat">
                <span>Latency</span>
                <span>{e.latency_ms.toFixed(2)} ms</span>
              </div>
              {!e.success && e.error_message && (
                <div className="error-message-box">
                  {e.error_message}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }

  const renderQualityPanel = () => {
    if (!result?.quality_result?.evaluations || result.quality_result.evaluations.length === 0) return null;
    
    return (
      <div className="section-card">
        <h3 className="section-title">Response Quality Analysis</h3>
        <div className="micro-card-grid">
          {result.quality_result.evaluations.map((evalItem, idx) => {
            const metrics = [
              { label: 'Relevance', score: evalItem.relevance_score },
              { label: 'Completeness', score: evalItem.completeness_score },
              { label: 'Compliance', score: evalItem.compliance_score },
              { label: 'Consistency', score: evalItem.consistency_score }
            ];
            
            return (
              <div key={idx} className="micro-card">
                <div className="micro-card-header" style={{marginBottom: '1rem'}}>
                  <span className="micro-card-title">{evalItem.model_id}</span>
                  <span className="status-badge" style={{background: 'rgba(59, 130, 246, 0.15)', color: '#93c5fd'}}>
                    {evalItem.quality_label}
                  </span>
                </div>
                
                {metrics.map(m => (
                  <div key={m.label} className="quality-row">
                    <div className="quality-header">
                      <span>{m.label}</span>
                      <span>{m.score.toFixed(2)}</span>
                    </div>
                    <div className="score-bar-container" style={{height: '4px', background: 'rgba(0,0,0,0.3)'}}>
                      <div className="score-bar-fill" style={{width: `${Math.max(0, Math.min(100, m.score * 100))}%`}} />
                    </div>
                  </div>
                ))}
                
                <div className="quality-row" style={{marginTop: '1rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(45,60,90,0.5)'}}>
                  <div className="quality-header" style={{color: '#fff', fontWeight: 600}}>
                    <span>Overall Quality</span>
                    <span>{evalItem.overall_quality_score.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    );
  }

  const renderMemoryPanel = () => {
    if (!result?.adaptive_signal) return null;
    const { adaptive_signal } = result;
    const hasData = adaptive_signal.total_samples > 0;
    
    return (
      <div className="section-card">
        <h3 className="section-title">Adaptive Outcome Memory</h3>
        <p style={{fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem'}}>
          Historical outcomes influence future routing preferences dynamically.
        </p>
        <div className="badge-grid">
          <div className="badge-item">
            <span className="badge-label">Sample Count</span>
            <span className="badge-value">{adaptive_signal.total_samples}</span>
          </div>
          <div className="badge-item">
            <span className="badge-label">Historical Quality</span>
            <span className="badge-value">{hasData && adaptive_signal.historical_quality_score != null ? adaptive_signal.historical_quality_score.toFixed(4) : 'No historical data'}</span>
          </div>
          <div className="badge-item">
            <span className="badge-label">Success Rate</span>
            <span className="badge-value">{hasData && adaptive_signal.historical_success_rate != null ? (adaptive_signal.historical_success_rate * 100).toFixed(1) + '%' : 'No historical data'}</span>
          </div>
          <div className="badge-item">
            <span className="badge-label">Adaptive Multiplier</span>
            <span className="badge-value">{adaptive_signal.adaptive_weight_multiplier.toFixed(4)}</span>
          </div>
        </div>
      </div>
    );
  }

  const renderTracePanel = () => {
    if (!result?.trace) return null;
    return (
      <div className="section-card">
        <h3 className="section-title">Explainable Decision Trace</h3>
        <div className="trace-content">
          {result.trace.decision_summary}
        </div>
      </div>
    );
  }

  const renderFinalResponsePanel = () => {
    if (!result?.final_selected_content) return null;
    return (
      <div className="section-card section-card-highlight" style={{borderLeft: '4px solid var(--accent-blue)'}}>
        <h3 className="section-title">Final Response ({result.final_selected_model})</h3>
        <div className="final-response-content">
          {result.final_selected_content}
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-shell">
      <header className="dashboard-header">
        <div className="logo-group">
          <div className="logo-badge">GR</div>
          <div>
            <h1 className="brand-title">GreenRoute AI</h1>
            <p className="brand-tagline">Semantic LLM Routing & Optimization Platform</p>
          </div>
        </div>
        <div className="status-indicator">
          <span className="status-dot"></span>
          <span>Foundation Active</span>
        </div>
      </header>

      <main className="main-content">
        <section className="panel" style={{position: 'sticky', top: '2.5rem'}}>
          <h2 className="panel-title">Prompt Profiler & Router</h2>
          <p className="panel-subtitle">
            Enter a prompt below to evaluate model capabilities, compute confidence gaps, and route execution.
          </p>

          <textarea
            className="prompt-textarea"
            placeholder="e.g. Write a Python function to perform binary search on a sorted array..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          <div className="button-group">
            <button className="btn-route" onClick={handleRoute} disabled={uiState === 'loading'}>
              Analyze & Route Prompt
            </button>
          </div>

          {uiState === 'error' && (
            <div className="placeholder-notice">
              ❌ {errorMessage}
            </div>
          )}
        </section>

        <section className="panel" style={{background: 'transparent', border: 'none', padding: 0}}>
          {uiState === 'idle' && (
            <div className="panel">
              <div className="empty-state">
                <div className="empty-icon">🧭</div>
                <h3 className="empty-title">Awaiting Routing Input</h3>
                <p className="empty-desc">
                  Submit a prompt on the left panel to trigger profiler classification, candidate scoring, and response evaluation.
                </p>
              </div>
            </div>
          )}

          {uiState === 'loading' && (
            <div className="panel">
              <div className="loading-container">
                <div className="spinner"></div>
                <div className="loading-text">{LOADING_MESSAGES[loadingMsgIdx]}</div>
              </div>
            </div>
          )}

          {uiState === 'success' && result && (
            <div className="results-container">
              {renderFinalResponsePanel()}
              {renderTracePanel()}
              {renderRoutingPanel()}
              {renderProfilePanel()}
              {renderCandidatesPanel()}
              {renderExecutionPanel()}
              {renderQualityPanel()}
              {renderMemoryPanel()}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
