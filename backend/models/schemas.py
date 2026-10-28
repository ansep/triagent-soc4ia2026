from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


# message types flowing between the three agents
class MessageType(str, Enum):
    KNOWLEDGE_REQUEST = "knowledge_request"
    ACTION_REQUEST = "action_request"
    ACTION_RESULT = "action_result"
    ESCALATION = "escalation"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    MODERATE = "moderate"
    SEVERE = "severe"
    EARLY = "early"
    THRESHOLD = "threshold"
    SUSPECT = "suspect"
    CONFIRMED = "confirmed"
    SPREADING = "spreading"
    EPIDEMIC = "epidemic"


class AgentMessage(BaseModel):
    type: MessageType
    content: str
    process_state: Optional[dict] = {}
    confidence: Optional[float] = None
    step: Optional[int] = None


# what the user sends to the API
class ProcessRequest(BaseModel):
    description: str
    crop: Optional[str] = None
    location: Optional[str] = None
    growth_stage: Optional[str] = None
    severity: Optional[Severity] = None


# the rule retrieved from the vector store
class RuleMatch(BaseModel):
    rule_id: str
    trigger: str
    content: str
    similarity_score: float


# one step in the reasoning chain
class ReasoningStep(BaseModel):
    step: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float
    gateway_decision: Optional[str] = None


# full response back to the frontend
class ProcessResult(BaseModel):
    success: bool
    process_id: str
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    actions_taken: List[str] = []
    reasoning_trace: List[ReasoningStep] = []
    rule_used: Optional[RuleMatch] = None
    hallucination_flags: List[str] = []
    escalated: bool = False
    cost_usd: Optional[float] = None
    llm_model: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    rules_loaded: int
    llm_model: str
