import re
from typing import Tuple, Optional
from app.schemas.profile import (
    PromptProfile,
    TaskType,
    Domain,
    ComplexityLevel,
    ReasoningRequirement
)

class PromptProfiler:
    """
    Semantic Prompt Intelligence Profiler (Phase 4).
    
    This is a lightweight, rule-based, deterministic profiler that analyzes prompt text
    to extract semantic intent, task type, domain, complexity, reasoning requirements,
    and expected output format.
    
    Design Note:
    This initial version intentionally uses transparent keyword/phrase heuristics and regex
    rules rather than heavy ML/NLP models to ensure high speed, zero external dependencies,
    and predictable testing. Future versions can extend or replace internal classifier methods
    with semantic embedding or LLM-based classifiers without altering the `analyze(prompt)` interface.
    """

    # Keyword rules for task identification
    CODING_KEYWORDS = {
        "python", "javascript", "code", "debug", "recursion", "algorithm", "function",
        "class", "syntax", "refactor", "bug", "exception", "compile", "script", "sql"
    }
    
    REASONING_KEYWORDS = {
        "compare", "tradeoff", "tradeoffs", "analyze", "evaluate", "why", "contrast",
        "difference", "proof", "explain the relationship", "pros and cons"
    }

    MATH_KEYWORDS = {
        "math", "calculus", "algebra", "theorem", "equation", "solve", "integral", "derivative"
    }

    TRANSLATION_KEYWORDS = {
        "translate", "translation", "into french", "into spanish", "into german",
        "into japanese", "into chinese", "into hindi"
    }

    CREATIVE_KEYWORDS = {
        "write a story", "poem", "essay", "creative", "brainstorm", "generate a draft"
    }

    # Language target detection
    LANGUAGE_TARGETS = [
        "french", "spanish", "german", "japanese", "chinese", "hindi", "italian", "russian", "portuguese"
    ]

    def analyze(self, prompt: str) -> PromptProfile:
        """
        Main entry point for profiling a user prompt.
        Converts raw prompt string into a structured PromptProfile.
        """
        cleaned_prompt = prompt.strip()
        lower_prompt = cleaned_prompt.lower()

        # 1. Determine Task Type and Domain
        task_type, domain = self._classify_task_and_domain(lower_prompt)

        # 2. Determine Complexity and Reasoning Need
        complexity, reasoning_need = self._assess_complexity_and_reasoning(lower_prompt, task_type)

        # 3. Determine Expected Output Format
        expected_output = self._determine_expected_output(lower_prompt, task_type)

        # 4. Detect Target Language
        language = self._detect_language(lower_prompt)

        return PromptProfile(
            prompt=cleaned_prompt,
            task_type=task_type,
            domain=domain,
            complexity=complexity,
            reasoning_need=reasoning_need,
            expected_output=expected_output,
            language=language
        )

    def _classify_task_and_domain(self, lower_prompt: str) -> Tuple[TaskType, Domain]:
        # Translation check
        if any(kw in lower_prompt for kw in self.TRANSLATION_KEYWORDS):
            return TaskType.TRANSLATION, Domain.LANGUAGE

        # Coding check
        if any(re.search(r'\b' + re.escape(kw) + r'\b', lower_prompt) for kw in self.CODING_KEYWORDS):
            return TaskType.CODING, Domain.COMPUTER_SCIENCE

        # Direct Math check
        if any(re.search(r'\b' + re.escape(kw) + r'\b', lower_prompt) for kw in self.MATH_KEYWORDS) and any(term in lower_prompt for term in ["math", "calculus", "algebra", "theorem", "equation", "solve"]):
            return TaskType.REASONING, Domain.MATHEMATICS

        # Reasoning / Comparative check
        if any(kw in lower_prompt for kw in self.REASONING_KEYWORDS):
            if any(term in lower_prompt for term in self.MATH_KEYWORDS):
                return TaskType.REASONING, Domain.MATHEMATICS
            return TaskType.REASONING, Domain.GENERAL

        # Creative check
        if any(kw in lower_prompt for kw in self.CREATIVE_KEYWORDS):
            return TaskType.CREATIVE, Domain.GENERAL

        # Simple arithmetic / basic Q&A check
        if re.search(r'\b(what is|who is|where is|when was)\b', lower_prompt) or re.search(r'^\d+\s*[\+\-\*\/]\s*\d+', lower_prompt):
            if re.search(r'[\+\-\*\/]', lower_prompt):
                return TaskType.GENERAL_QA, Domain.MATHEMATICS
            return TaskType.GENERAL_QA, Domain.GENERAL

        return TaskType.UNKNOWN, Domain.UNKNOWN

    def _assess_complexity_and_reasoning(
        self, lower_prompt: str, task_type: TaskType
    ) -> Tuple[ComplexityLevel, ReasoningRequirement]:
        word_count = len(lower_prompt.split())

        # High complexity signals
        has_deep_reasoning = any(kw in lower_prompt for kw in ["explain", "tradeoff", "tradeoffs", "why", "compare", "recursion", "complexity", "architecture"])
        is_long = word_count > 30

        if task_type in [TaskType.CODING, TaskType.REASONING] and (has_deep_reasoning or is_long):
            return ComplexityLevel.HIGH, ReasoningRequirement.HIGH

        if task_type == TaskType.TRANSLATION:
            if is_long:
                return ComplexityLevel.MEDIUM, ReasoningRequirement.LOW
            return ComplexityLevel.LOW, ReasoningRequirement.LOW

        if task_type == TaskType.GENERAL_QA and word_count < 10:
            return ComplexityLevel.LOW, ReasoningRequirement.LOW

        # Default fallback
        if is_long or has_deep_reasoning:
            return ComplexityLevel.MEDIUM, ReasoningRequirement.MEDIUM

        return ComplexityLevel.LOW, ReasoningRequirement.LOW

    def _determine_expected_output(self, lower_prompt: str, task_type: TaskType) -> str:
        if task_type == TaskType.CODING:
            if "explain" in lower_prompt or "time complexity" in lower_prompt or "why" in lower_prompt:
                return "Code + Explanation"
            return "Code"
        if task_type == TaskType.TRANSLATION:
            return "Translated Text"
        if task_type == TaskType.REASONING:
            return "Detailed Analysis / Explanation"
        return "Text"

    def _detect_language(self, lower_prompt: str) -> str:
        for lang in self.LANGUAGE_TARGETS:
            if f"into {lang}" in lower_prompt or f"in {lang}" in lower_prompt or f"to {lang}" in lower_prompt:
                return lang.capitalize()
        return "English"
