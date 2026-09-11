from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .semantic import (
    EmbeddingSemanticMatcher,
    SemanticMatcher,
    normalize_semantic_text,
)


# ============================================================
# Data model
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


class SemanticAdjudicator(Protocol):
    def judge(
        self,
        response_text: str,
        reference_text: str,
        task_type: Optional[str] = None,
        language: str = "auto",
        context: Optional[Dict[str, Any]] = None,
    ) -> AdjudicationResult:
        ...


# ============================================================
# Helpers
# ============================================================

NEGATION_MARKERS = {
    "no",
    "nunca",
    "jamas",
    "jamás",
    "tampoco",
    "ningun",
    "ningún",
    "ninguna",
    "ninguno",
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


GENERIC_CONCEPTUAL_CONFLICTS: Sequence[Tuple[Sequence[str], Sequence[str]]] = (
    (
        ("temporal", "temporalmente", "temporary", "temporarily"),
        ("permanente", "permanent", "largo plazo", "long term", "long-term"),
    ),
    (
        ("aumenta", "incrementa", "increase", "increases"),
        ("disminuye", "reduce", "decrease", "decreases", "reduces"),
    ),
    (
        ("antes", "before", "previo", "previous"),
        ("despues", "después", "after", "posterior"),
    ),
    (
        ("causa", "produce", "provoca", "causes", "produces"),
        ("previene", "impide", "prevents", "inhibits"),
    ),
)


def _tokens(text: str) -> List[str]:
    normalized = normalize_semantic_text(text)
    return [token for token in normalized.split() if token]


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _contains_negation(text: str) -> bool:
    tokens = set(_tokens(text))
    return any(marker in tokens for marker in NEGATION_MARKERS)


def _negation_mismatch(response: str, reference: str) -> bool:
    response_negated = _contains_negation(response)
    reference_negated = _contains_negation(reference)
    return response_negated != reference_negated


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_semantic_text(text)
    normalized_phrase = normalize_semantic_text(phrase)

    if not normalized_phrase:
        return False

    return normalized_phrase in normalized_text


def _declared_incompatible_concepts(
    response_text: str,
    context: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Detect misconceptions explicitly declared by the assessment specification.

    Expected context example:

    context = {
        "incompatible_concepts": [
            "memoria de largo plazo",
            "almacenamiento permanente"
        ]
    }

    This is intentionally specification-driven rather than hard-coded
    to one academic domain.
    """

    if not context:
        return []

    candidates: List[str] = []

    for key in (
        "incompatible_concepts",
        "forbidden_concepts",
        "misconception_patterns",
    ):
        values = context.get(key, [])

        if isinstance(values, str):
            values = [values]

        if isinstance(values, (list, tuple, set)):
            for value in values:
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())

    detected = [
        concept
        for concept in candidates
        if _contains_phrase(response_text, concept)
    ]

    return detected


def _generic_conceptual_conflicts(
    response_text: str,
    reference_text: str,
) -> List[Dict[str, Any]]:
    """
    Conservative generic conflict detector.

    It only reports a conflict when one side of an opposition appears
    in the reference and the opposite side appears in the response.
    """

    conflicts: List[Dict[str, Any]] = []

    for side_a, side_b in GENERIC_CONCEPTUAL_CONFLICTS:
        reference_has_a = any(
            _contains_phrase(reference_text, phrase)
            for phrase in side_a
        )
        reference_has_b = any(
            _contains_phrase(reference_text, phrase)
            for phrase in side_b
        )

        response_has_a = any(
            _contains_phrase(response_text, phrase)
            for phrase in side_a
        )
        response_has_b = any(
            _contains_phrase(response_text, phrase)
            for phrase in side_b
        )

        if reference_has_a and response_has_b:
            conflicts.append(
                {
                    "reference_side": list(side_a),
                    "response_side": list(side_b),
                }
            )

        elif reference_has_b and response_has_a:
            conflicts.append(
                {
                    "reference_side": list(side_b),
                    "response_side": list(side_a),
                }
            )

    return conflicts


# ============================================================
# Baseline adjudicator
# ============================================================

class BaselineAdjudicator:
    """
    Conservative placeholder adjudicator.

    Useful when a semantic model is unavailable.
    """

    def judge(
        self,
        response_text: str,
        reference_text: str,
        task_type: Optional[str] = None,
        language: str = "auto",
        context: Optional[Dict[str, Any]] = None,
    ) -> AdjudicationResult:

        response = normalize_semantic_text(response_text)
        reference = normalize_semantic_text(reference_text)

        if not response:
            return AdjudicationResult(
                relation="insufficient_evidence",
                score=0.0,
                confidence=1.0,
                rationale="The response is empty.",
                metadata={
                    "adjudicator": "baseline",
                    "calibrated": False,
                },
            )

        if response == reference:
            return AdjudicationResult(
                relation="correct",
                score=1.0,
                confidence=1.0,
                rationale="The response exactly matches the reference.",
                evidence=response_text,
                metadata={
                    "adjudicator": "baseline",
                    "calibrated": False,
                },
            )

        return AdjudicationResult(
            relation="undetermined",
            score=0.0,
            confidence=0.5,
            rationale=(
                "The baseline adjudicator cannot establish conceptual "
                "correctness reliably."
            ),
            evidence=response_text,
            metadata={
                "adjudicator": "baseline",
                "calibrated": False,
            },
        )


# ============================================================
# Hybrid conceptual adjudicator
# ============================================================

class HybridConceptualAdjudicator:
    """
    Hybrid conceptual adjudicator v2.

    Combines:

    1. multilingual semantic similarity;
    2. explicit polarity / negation conflict;
    3. assessment-specification driven misconception detection;
    4. conservative generic conceptual conflict detection.

    This is NOT deep natural-language inference.

    Thresholds and confidence values remain heuristic and
    are not empirically calibrated probabilities.
    """

    def __init__(
        self,
        semantic_matcher: Optional[SemanticMatcher] = None,
        irrelevant_threshold: float = 0.35,
        partial_threshold: float = 0.50,
        correct_threshold: float = 0.72,
        contradiction_relevance_threshold: float = 0.45,
        misconception_relevance_threshold: float = 0.45,
    ) -> None:

        self.semantic_matcher = (
            semantic_matcher or EmbeddingSemanticMatcher()
        )

        self.irrelevant_threshold = irrelevant_threshold
        self.partial_threshold = partial_threshold
        self.correct_threshold = correct_threshold

        self.contradiction_relevance_threshold = (
            contradiction_relevance_threshold
        )

        self.misconception_relevance_threshold = (
            misconception_relevance_threshold
        )

    def judge(
        self,
        response_text: str,
        reference_text: str,
        task_type: Optional[str] = None,
        language: str = "auto",
        context: Optional[Dict[str, Any]] = None,
    ) -> AdjudicationResult:

        response = response_text.strip()
        reference = reference_text.strip()

        # ----------------------------------------------------
        # 1. Evidence sufficiency
        # ----------------------------------------------------

        if not response:
            return AdjudicationResult(
                relation="insufficient_evidence",
                score=0.0,
                confidence=1.0,
                rationale="The response contains no evaluable evidence.",
                evidence=response_text,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
                },
            )

        if len(_tokens(response)) < 3:
            return AdjudicationResult(
                relation="insufficient_evidence",
                score=0.0,
                confidence=0.85,
                rationale=(
                    "The response is too brief to support a reliable "
                    "conceptual judgment."
                ),
                evidence=response_text,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "deep_nli": False,
                    "calibrated": False,
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

        similarity = float(semantic_result.similarity)

        # ----------------------------------------------------
        # 3. Polarity conflict
        # ----------------------------------------------------

        polarity_conflict = _negation_mismatch(
            response=response,
            reference=reference,
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
                    "The response is semantically related to the reference "
                    "but presents an opposing explicit polarity."
                ),
                evidence=response_text,
                contradiction=True,
                partial=False,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": True,
                    "misconception_detected": False,
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
                },
            )

        # ----------------------------------------------------
        # 4. Declared misconceptions
        # ----------------------------------------------------

        declared_misconceptions = _declared_incompatible_concepts(
            response_text=response,
            context=context,
        )

        if (
            declared_misconceptions
            and similarity >= self.misconception_relevance_threshold
        ):
            confidence = _bounded(
                0.68 + (0.28 * similarity)
            )

            return AdjudicationResult(
                relation="misconception",
                score=0.0,
                confidence=confidence,
                rationale=(
                    "The response is semantically related to the target "
                    "concept but contains a concept explicitly declared "
                    "as incompatible with the expected answer."
                ),
                evidence=", ".join(declared_misconceptions),
                contradiction=False,
                partial=False,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "misconception_detected": True,
                    "misconception_source": "assessment_specification",
                    "detected_incompatible_concepts": (
                        declared_misconceptions
                    ),
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
                },
            )

        # ----------------------------------------------------
        # 5. Generic conceptual conflicts
        # ----------------------------------------------------

        generic_conflicts = _generic_conceptual_conflicts(
            response_text=response,
            reference_text=reference,
        )

        if (
            generic_conflicts
            and similarity >= self.misconception_relevance_threshold
        ):
            confidence = _bounded(
                0.62 + (0.25 * similarity)
            )

            return AdjudicationResult(
                relation="misconception",
                score=0.0,
                confidence=confidence,
                rationale=(
                    "The response is semantically related to the target "
                    "but contains a conceptual relation that conflicts "
                    "with the reference."
                ),
                evidence=response_text,
                contradiction=False,
                partial=False,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "misconception_detected": True,
                    "misconception_source": "generic_conceptual_conflict",
                    "conceptual_conflicts": generic_conflicts,
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
                },
            )

        # ----------------------------------------------------
        # 6. Irrelevant response
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
                    "The response shows insufficient semantic relation "
                    "to the expected content."
                ),
                evidence=response_text,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "misconception_detected": False,
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
                },
            )

        # ----------------------------------------------------
        # 7. Undetermined zone
        # ----------------------------------------------------

        if similarity < self.partial_threshold:
            return AdjudicationResult(
                relation="undetermined",
                score=0.0,
                confidence=0.50,
                rationale=(
                    "The response is related to the target but does not "
                    "provide enough evidence for a stable conceptual "
                    "classification."
                ),
                evidence=response_text,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "misconception_detected": False,
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
                },
            )

        # ----------------------------------------------------
        # 8. Partial correspondence
        # ----------------------------------------------------

        if similarity < self.correct_threshold:
            confidence = _bounded(
                0.50 + (similarity / 3.0)
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
                evidence=response_text,
                contradiction=False,
                partial=True,
                metadata={
                    "semantic_reasoning": True,
                    "adjudicator": "hybrid_v2",
                    "semantic_similarity": similarity,
                    "semantic_method": semantic_result.method,
                    "polarity_conflict": False,
                    "misconception_detected": False,
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
                },
            )

        # ----------------------------------------------------
        # 9. Correct
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
                "with the reference and no explicit conceptual conflict "
                "was detected."
            ),
            evidence=response_text,
            contradiction=False,
            partial=False,
            metadata={
                "semantic_reasoning": True,
                "adjudicator": "hybrid_v2",
                "semantic_similarity": similarity,
                "semantic_method": semantic_result.method,
                "polarity_conflict": False,
                "misconception_detected": False,
                "deep_nli": False,
                "calibrated": False,
                "task_type": task_type,
                "language": language,
            },
        )


# ============================================================
# Public API
# ============================================================

DEFAULT_ADJUDICATOR = HybridConceptualAdjudicator()


def adjudicate(
    response_text: str,
    reference_text: str,
    task_type: Optional[str] = None,
    language: str = "auto",
    context: Optional[Dict[str, Any]] = None,
    engine: Optional[SemanticAdjudicator] = None,
) -> AdjudicationResult:

    adjudicator = engine or DEFAULT_ADJUDICATOR

    return adjudicator.judge(
        response_text=response_text,
        reference_text=reference_text,
        task_type=task_type,
        language=language,
        context=context,
    )
