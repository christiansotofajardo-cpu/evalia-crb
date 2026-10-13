from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class AdjudicationResult:
    relation: str
    score: float
    confidence: float
    rationale: str
    evidence: str = ""
    contradiction: bool = False
    partial: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticAdjudicator(Protocol):
    def judge(
        self,
        response_text: str,
        reference_text: str,
        *,
        task_type: Optional[str] = None,
        language: str = "auto",
        context: Optional[Dict[str, Any]] = None,
    ) -> AdjudicationResult:
        ...


class BaselineAdjudicator:
    """
    Conservative placeholder adjudicator.

    It does not attempt deep semantic reasoning.
    Its purpose is to define the contract that future
    LLM, NLI, or hybrid adjudicators must implement.
    """

    def judge(
        self,
        response_text: str,
        reference_text: str,
        *,
        task_type: Optional[str] = None,
        language: str = "auto",
        context: Optional[Dict[str, Any]] = None,
    ) -> AdjudicationResult:

        response = (response_text or "").strip()
        reference = (reference_text or "").strip()

        if not response or not reference:
            return AdjudicationResult(
                relation="insufficient_evidence",
                score=0.0,
                confidence=1.0,
                rationale="Empty response or reference.",
                metadata={
                    "semantic_reasoning": False,
                    "adjudicator": "baseline",
                },
            )

        return AdjudicationResult(
            relation="undetermined",
            score=0.0,
            confidence=0.0,
            rationale=(
                "Deep conceptual adjudication has not yet been applied."
            ),
            metadata={
                "semantic_reasoning": False,
                "adjudicator": "baseline",
                "task_type": task_type,
                "language": language,
            },
        )


DEFAULT_ADJUDICATOR: SemanticAdjudicator = BaselineAdjudicator()


def adjudicate(
    response_text: str,
    reference_text: str,
    *,
    task_type: Optional[str] = None,
    language: str = "auto",
    context: Optional[Dict[str, Any]] = None,
    adjudicator: Optional[SemanticAdjudicator] = None,
) -> AdjudicationResult:
    """
    Public interface for conceptual adjudication.

    Future engines may classify relations such as:
    - correct
    - partially_correct
    - contradictory
    - misconception
    - irrelevant
    - insufficient_evidence

    The rest of Evalia should depend on this interface,
    not on a specific provider or model.
    """

    engine = adjudicator or DEFAULT_ADJUDICATOR

    return engine.judge(
        response_text=response_text,
        reference_text=reference_text,
        task_type=task_type,
        language=language,
        context=context,
    )
