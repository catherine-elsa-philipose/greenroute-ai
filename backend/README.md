# GreenRoute AI

GreenRoute AI is a Semantic LLM Routing & Optimization Platform. It dynamically profiles incoming prompts, evaluates multiple candidate Large Language Models (LLMs), calculates a semantic confidence gap, and routes the prompt to the most suitable model. 

## The Problem
Modern GenAI applications often rely on a single, expensive, or heavily-constrained model (e.g., GPT-4 or Gemini 1.5 Pro) for every query. This results in:
1. **Inefficiency**: Simple tasks (like text formatting) consume high-cost reasoning tokens.
2. **Brittle Architectures**: If a single provider has an outage, the entire system degrades.
3. **Suboptimal Quality**: Certain models excel at coding while others excel at translation. 

## The Solution: GreenRoute Architecture
GreenRoute implements a multi-model orchestration pipeline:

1. **Prompt Profiler**: Analyzes the prompt to determine the `TaskType` (coding, reasoning, translation, etc.), domain, and complexity.
2. **Capability Registry**: Maintains a registry of available models (Mock, OpenAI, Google) and their known capabilities.
3. **Confidence-Aware Router**: Scores each candidate model against the prompt profile using a **Routing Formula**:
   - Matches capability fit (domain, task, reasoning).
   - Accounts for operational factors (latency, cost, reliability).
   - Calculates a **Confidence Gap** between the top candidate and the runner-up to determine the routing recommendation (e.g., *Direct Route* vs *Compare / Escalate*).
4. **Multi-Model Orchestrator**: Executes the prompt against the selected model(s). If multiple models are recommended, it orchestrates parallel execution.
5. **Quality Gate**: Evaluates the resulting execution against relevance, completeness, consistency, and compliance metrics.
6. **Adaptive Outcome Memory**: Records the successful/failed outcomes to a persistent SQLite database, dynamically influencing future routing weights based on historical success rates.
7. **Explainable Decision Trace**: Generates a transparent, human-readable summary of *why* a specific model was chosen and how it performed.

## Provider Abstraction
The system uses a `BaseLLMProvider` interface. V1 implements:
- `MockLLMProvider`: For local testing without incurring costs.
- `OpenAIProvider`: Lightweight REST integration for OpenAI models (`gpt-4o-mini`).
- `GeminiProvider`: Lightweight REST integration for Google Gemini models (`gemini-2.5-flash`).

Providers safely downgrade to unavailable if API keys are missing.

## Setup & Testing
1. Configure environment variables (see `.env.example`):
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```
2. Setup the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/Scripts/activate # Windows
   pip install -r requirements.txt
   pytest
   uvicorn app.main:app --host 0.0.0.0 --port 8020
   ```
3. Setup the frontend:
   ```bash
   cd frontend
   npm install
   npm run build
   npm run dev -- --port 5174
   ```

## Limitations
- **V1 Completion**: This is the V1 completion. RouteLab, user authentication, streaming, Kubernetes deployment, and advanced fine-tuning metrics are deliberately out of scope for V1.
