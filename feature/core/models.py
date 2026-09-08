from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Learner:
    id: str = ""
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseInput:
    text: str
    source: str = "text"
    source_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Criterion:
    id: str
    description: str
    weight: float = 1.0
    required: bool = False
    semantic_variants: List[str] = field(default_factory=list)
    accepted_values: List[str] = field(default_factory=list)
    contradictory_values: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentSpec:
    id: str = ""
    name: str = ""
    language: str = "auto"
    domain: str = "general"
    criteria: List[Criterion] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    scoring_policy: Dict[str, Any] = field(default_factory=dict)
    feedback_policy: Dict[str, Any] = field(default_factory=dict)
    reliability_policy: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    prompt: str
    task_type: str = "constructed_response"
    max_score: float = 1.0
    language: str = "auto"
    domain: str = "general"
    assessment_spec: Optional[AssessmentSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationInput:
    assessment_id: str
    task: Task
    response: ResponseInput
    learner: Learner = field(default_factory=Learner)
    language: str = "auto"
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepresentationResult:
    language: str = "unknown"
    concepts_detected: List[str] = field(default_factory=list)
    concepts_missing: List[str] = field(default_factory=list)
    conceptual_relations: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    conceptual_coverage: float = 0.0
    response_profile: str = ""
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriterionJudgment:
    criterion_id: str
    satisfied: bool
    score: float
    max_score: float
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentResult:
    score: float = 0.0
    max_score: float = 0.0
    criterion_judgments: List[CriterionJudgment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosisResult:
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    misconceptions: List[str] = field(default_factory=list)
    error_type: str = ""
    error_severity: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityResult:
    confidence: float = 0.0
    disagreement_risk: float = 0.0
    reliability_class: str = "unknown"
    risk_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DelegationResult:
    decision: str = "HUMAN_REVIEW"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackResult:
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    needs_improvement: List[str] = field(default_factory=list)
    next_step: str = ""
    audience: str = "learner"
    language: str = "es"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaliaResult:
    assessment_id: str
    task_id: str
    learner_id: str
    language: str
    representation: RepresentationResult
    judgment: AssessmentResult
    diagnosis: DiagnosisResult
    reliability: ReliabilityResult
    delegation: DelegationResult
    feedback: FeedbackResult
    traceability: Dict[str, Any] = field(default_factory=dict)
