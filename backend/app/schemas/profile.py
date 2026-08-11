from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    CREATIVE = "creative"
    GENERAL_QA = "general_qa"
    UNKNOWN = "unknown"

class Domain(str, Enum):
    COMPUTER_SCIENCE = "computer_science"
    MATHEMATICS = "mathematics"
    LANGUAGE = "language"
    GENERAL = "general"
    UNKNOWN = "unknown"

class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ReasoningRequirement(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class PromptProfile(BaseModel):
    """
    Structured representation of a user prompt produced by the Semantic Prompt Intelligence Profiler.
    """
    prompt: str = Field(..., description="Original user prompt evaluated by the profiler")
    task_type: TaskType = Field(default=TaskType.UNKNOWN, description="Identified task classification")
    domain: Domain = Field(default=Domain.UNKNOWN, description="Identified subject domain")
    complexity: ComplexityLevel = Field(default=ComplexityLevel.LOW, description="Estimated prompt complexity")
    reasoning_need: ReasoningRequirement = Field(default=ReasoningRequirement.LOW, description="Estimated reasoning requirement")
    expected_output: str = Field(default="text", description="Expected format/nature of output (e.g. text, code, code+explanation)")
    language: Optional[str] = Field(default="English", description="Detected language or target language if specified")
