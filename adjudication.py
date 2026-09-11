from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from .semantic import (
    EmbeddingSemanticMatcher,
    SemanticMatcher,
    normalize_semantic_text,
)


# ============================================================
# Result contract
# ============================================================

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


# ============================================================
# Adjudicator interface
# ============================================================

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


# ============================================================
# Conservative baseline
# ============================================================

class BaselineAdjudicator:
    """
    Conservative placeholder adjudicator.

    It defines the public contract without pretending to perform
    deep conceptual reasoning.
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


# ============================================================
# Negation utilities
# ============================================================

NEGATION_MARKERS = {
    # Spanish
    "no",
    "nunca",
    "jamas",
    "tampoco",
    "ningun",
    "ninguna",
    "ninguno",
    # English
    "not",
    "never",
    "neither",
    "nor",
    "cannot",
    "can't",
    "doesn't",
    "isn't",
    "aren't",
    "without",
}


def _tokens(text: str) -> list[str]:
    normalized = normalize_semantic_text(text)
    return normalized.split()


def _contains_negation(text: str) -> bool:
    return any(token in NEGATION_MARKERS for token in _tokens(text))


def _negation_mismatch(
    response_text: str,
    reference_text: str,
) -> bool:
    """
    Detect a simple polarity mismatch.

    This is deliberately conservative and is not intended
    to replace full natural-language inference.
    """

    response_negative = _contains_negation(response_text)
    reference_negative = _contains_negation(reference_text)

    return response_negative != reference_negative


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


# ============================================================
# Hybrid conceptual adjudicator
# ============================================================

class HybridConceptualAdjudicator:
    """
    First operational conceptual adjudicator for Evalia Core.

    Architecture:
        semantic relevance
            +
        explicit polarity / negation check
            +
        conservative decision thresholds

    This adjudicator is useful as an explainable intermediate layer,
    but it is NOT equivalent to deep natural-language inference.

    Future versions may add:
        - NLI models
        - LLM structured reasoning
        - criterion-specific relations
        - causal reasoning
        - misconception detection
        - multi-evidence adjudication
    """

    def __init__(
        self,
        semantic_matcher: Optional[SemanticMatcher] = None,
        irrelevant_threshold: float = 0.35,
        partial_threshold: float = 0.50,
        correct_threshold: float = 0.72,
        contradiction_relevance_threshold: float = 0.45,
    ):
        self.semantic_matcher = (
            semantic_matcher or EmbeddingSemanticMatcher()
        )

        self.irrelevant_threshold = irrelevant_threshold
        self.partial_threshold = partial_threshold
        self.correct_threshold = correct_threshold
        self.contradiction_relevance_threshold = (
            contradiction_relevance_threshold
        )

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

        # ----------------------------------------------------
        # 1. Missing evidence
        # ----------------------------------------------------

        if not response or not reference:
            return AdjudicationResult(
                relation="insufficient_evidence",
                score=0.0,
                confidence=1.0,
                rationale="Response or reference is empty.",
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v1",
                    "task_type": task_type,
                    "language": language,
                },
            )

        # Very short responses are treated conservatively.
        if len(_tokens(response)) < 3:
            return AdjudicationResult(
                relation="insufficient_evidence",
                score=0.0,
                confidence=0.85,
                rationale=(
                    "Response is too brief for reliable conceptual "
                    "adjudication."
                ),
                evidence=response,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v1",
                    "task_type": task_type,
                    "language": language,
                },
            )

        # ----------------------------------------------------
        # 2. Semantic relevance
        # ----------------------------------------------------

        semantic_result = self.semantic_matcher.compare(
            response_text=response,
            reference_text=reference,
            threshold=self.partial_threshold,
        )

        similarity = semantic_result.similarity

        # ----------------------------------------------------
        # 3. Explicit polarity contradiction
        # ----------------------------------------------------

        polarity_conflict = _negation_mismatch(
            response_text=response,
            reference_text=reference,
        )

        if (
            polarity_conflict
            and similarity >= self.contradiction_relevance_threshold
        ):
            confidence = _bounded(
                0.70 + (0.30 * similarity)
            )

            return AdjudicationResult(
                relation="contradictory",
                score=0.0,
                confidence=confidence,
                rationale=(
                    "The response is semantically related to the "
                    "reference but presents an opposing explicit polarity."
                ),
                evidence=response,
                contradiction=True,
                partial=False,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v1",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": True,
                    "task_type": task_type,
                    "language": language,
                    "deep_nli": False,
                    "calibrated": False,
                },
            )

        # ----------------------------------------------------
        # 4. Irrelevant response
        # ----------------------------------------------------

        if similarity < self.irrelevant_threshold:
            confidence = _bounded(
                1.0 - similarity
            )

            return AdjudicationResult(
                relation="irrelevant",
                score=0.0,
                confidence=confidence,
                rationale=(
                    "Semantic similarity is too low to support "
                    "conceptual relevance."
                ),
                evidence=response,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v1",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "task_type": task_type,
                    "language": language,
                    "deep_nli": False,
                    "calibrated": False,
                },
            )

        # ----------------------------------------------------
        # 5. Weak / ambiguous conceptual evidence
        # ----------------------------------------------------

        if similarity < self.partial_threshold:
            return AdjudicationResult(
                relation="undetermined",
                score=0.0,
                confidence=0.50,
                rationale=(
                    "The response appears conceptually related, "
                    "but evidence is insufficient for a stronger judgment."
                ),
                evidence=response,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v1",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "task_type": task_type,
                    "language": language,
                    "deep_nli": False,
                    "calibrated": False,
                },
            )

        # ----------------------------------------------------
        # 6. Partial conceptual support
        # ----------------------------------------------------

        if similarity < self.correct_threshold:
            confidence = _bounded(
                0.50 + similarity / 3.0
            )

            return AdjudicationResult(
                relation="partially_correct",
                score=0.5,
                confidence=confidence,
                rationale=(
                    "The response provides meaningful semantic support "
                    "but does not reach the conservative threshold for "
                    "full conceptual correspondence."
                ),
                evidence=response,
                contradiction=False,
                partial=True,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v1",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "task_type": task_type,
                    "language": language,
                    "deep_nli": False,
                    "calibrated": False,
                },
            )

        # ----------------------------------------------------
        # 7. Strong conceptual support
        # ----------------------------------------------------

        confidence = _bounded(
            0.60 + (0.40 * similarity)
        )

        return AdjudicationResult(
            relation="correct",
            score=1.0,
            confidence=confidence,
            rationale=(
                "The response shows strong semantic correspondence "
                "with the reference and no explicit polarity conflict "
                "was detected."
            ),
            evidence=response,
            contradiction=False,
            partial=False,
            metadata={
                "semantic_reasoning": True,
                "adjudicator": "hybrid_v1",
                "semantic_similarity": similarity,
                "semantic_method": semantic_result.method,
                "polarity_conflict": False,
                "task_type": task_type,
                "language": language,
                "deep_nli": False,
                "calibrated": False,
            },
        )


# ============================================================
# Default engine
# ============================================================

DEFAULT_ADJUDICATOR: SemanticAdjudicator = (
    HybridConceptualAdjudicator()
)


# ============================================================
# Public entry point
# ============================================================

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
    Public conceptual adjudication interface.

    Current relation space:
        - correct
        - partially_correct
        - contradictory
        - irrelevant
        - insufficient_evidence
        - undetermined

    Future engines may additionally support:
        - misconception
        - entailment
        - causal_mismatch
        - relational_error
        - overgeneralization
        - conceptual_precision_error

    Evalia Core should depend on this public interface rather than
    directly on a specific model or provider.
    """

    engine = adjudicator or DEFAULT_ADJUDICATOR

    return engine.judge(
        response_text=response_text,
        reference_text=reference_text,
        task_type=task_type,
        language=language,
        context=context,
    )
