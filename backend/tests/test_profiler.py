from app.profiler.profiler import PromptProfiler
from app.schemas.profile import (
    TaskType,
    Domain,
    ComplexityLevel,
    ReasoningRequirement
)

def test_simple_general_qa():
    profiler = PromptProfiler()
    profile = profiler.analyze("What is 2 + 2?")
    
    assert profile.task_type in [TaskType.GENERAL_QA, TaskType.REASONING]
    assert profile.complexity == ComplexityLevel.LOW
    assert profile.reasoning_need == ReasoningRequirement.LOW

def test_coding_debugging_prompt():
    profiler = PromptProfiler()
    profile = profiler.analyze("Debug this Python recursion problem and explain the time complexity.")
    
    assert profile.task_type == TaskType.CODING
    assert profile.domain == Domain.COMPUTER_SCIENCE
    assert profile.complexity == ComplexityLevel.HIGH
    assert profile.reasoning_need == ReasoningRequirement.HIGH
    assert profile.expected_output == "Code + Explanation"

def test_reasoning_comparison_prompt():
    profiler = PromptProfiler()
    profile = profiler.analyze("Compare these two algorithms and explain their tradeoffs.")
    
    assert profile.task_type == TaskType.REASONING
    assert profile.complexity == ComplexityLevel.HIGH
    assert profile.reasoning_need == ReasoningRequirement.HIGH
    assert profile.expected_output == "Detailed Analysis / Explanation"

def test_translation_prompt():
    profiler = PromptProfiler()
    profile = profiler.analyze("Translate this paragraph into French.")
    
    assert profile.task_type == TaskType.TRANSLATION
    assert profile.domain == Domain.LANGUAGE
    assert profile.complexity == ComplexityLevel.LOW
    assert profile.reasoning_need == ReasoningRequirement.LOW
    assert profile.expected_output == "Translated Text"
    assert profile.language == "French"

def test_complex_technical_prompt():
    profiler = PromptProfiler()
    prompt = "Refactor this SQL query with window functions to optimize execution plan and reduce latency."
    profile = profiler.analyze(prompt)
    
    assert profile.task_type == TaskType.CODING
    assert profile.domain == Domain.COMPUTER_SCIENCE

def test_ambiguous_general_prompt():
    profiler = PromptProfiler()
    profile = profiler.analyze("Hello there")
    
    assert profile.task_type == TaskType.UNKNOWN
    assert profile.domain == Domain.UNKNOWN
    assert profile.complexity == ComplexityLevel.LOW
    assert profile.reasoning_need == ReasoningRequirement.LOW
