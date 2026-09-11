from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .semantic import (
    EmbeddingSemanticMatcher,
    SemanticMatcher,
    normalize_semantic_text,
)


# ============================================================
# RESULT CONTRACT
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
# CONSTANTS
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


GENERIC_CONCEPTUAL_CONFLICTS: Sequence[
    Tuple[Sequence[str], Sequence[str]]
] = (
    (
        ("temporal", "temporalmente", "temporary", "temporarily"),
        (
            "permanente",
            "permanently",
            "permanent",
            "largo plazo",
            "long term",
            "long-term",
        ),
    ),
    (
        ("aumenta", "incrementa", "aumentar", "increase", "increases"),
        (
            "disminuye",
            "reduce",
            "disminuir",
            "decrease",
            "decreases",
            "reduces",
        ),
    ),
    (
        ("antes", "previo", "previamente", "before", "previous"),
        ("despues", "después", "posterior", "after"),
    ),
    (
        ("causa", "produce", "provoca", "causes", "produces"),
        ("previene", "impide", "inhibe", "prevents", "inhibits"),
    ),
)


# ============================================================
# HELPERS
# ============================================================

def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> List[str]:
    normalized = normalize_semantic_text(text)
    return [token for token in normalized.split() if token]


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_semantic_text(text)
    normalized_phrase = normalize_semantic_text(phrase)

    if not normalized_text or not normalized_phrase:
        return False

    return normalized_phrase in normalized_text


def _contains_negation(text: str) -> bool:
    tokens = set(_tokens(text))
    return any(marker in tokens for marker in NEGATION_MARKERS)


def _negation_mismatch(
    response_text: str,
    reference_text: str,
) -> bool:
    response_negation = _contains_negation(response_text)
    reference_negation = _contains_negation(reference_text)

    return response_negation != reference_negation


def _declared_incompatible_concepts(
    response_text: str,
    context: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Detect concepts explicitly declared as incompatible with
    the expected answer.

    Supported context keys:

        incompatible_concepts
        forbidden_concepts
        misconception_patterns

    Example:

        context={
            "incompatible_concepts": [
                "memoria de largo plazo",
                "recuerdos de largo plazo",
                "almacenamiento permanente",
            ]
        }

    The evaluator does not invent domain-specific misconceptions.
    They are supplied by the assessment specification.
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

    detected: List[str] = []

    for concept in candidates:
        if _contains_phrase(response_text, concept):
            detected.append(concept)

    return detected


def _generic_conceptual_conflicts(
    response_text: str,
    reference_text: str,
) -> List[Dict[str, Any]]:
    """
    Detect a small set of conservative conceptual oppositions.

    A conflict is reported only when:
    - one pole appears in the reference; and
    - the opposite pole appears in the response.

    This layer is intentionally small and transparent.
    """

    conflicts: List[Dict[str, Any]] = []

    for side_a, side_b in GENERIC_CONCEPTUAL_CONFLICTS:

        reference_a = [
            phrase
            for phrase in side_a
            if _contains_phrase(reference_text, phrase)
        ]

        reference_b = [
            phrase
            for phrase in side_b
            if _contains_phrase(reference_text, phrase)
        ]

        response_a = [
            phrase
            for phrase in side_a
            if _contains_phrase(response_text, phrase)
        ]

        response_b = [
            phrase
            for phrase in side_b
            if _contains_phrase(response_text, phrase)
        ]

        if reference_a and response_b:
            conflicts.append(
                {
                    "reference_terms": reference_a,
                    "response_terms": response_b,
                    "type": "conceptual_opposition",
                }
            )

        elif reference_b and response_a:
            conflicts.append(
                {
                    "reference_terms": reference_b,
                    "response_terms": response_a,
                    "type": "conceptual_opposition",
                }
            )

    return conflicts


def _base_metadata(
    *,
    adjudicator: str,
    similarity: Optional[float] = None,
    semantic_method: Optional[str] = None,
    polarity_conflict: bool = False,
    misconception_detected: bool = False,
    task_type: Optional[str] = None,
    language: str = "auto",
) -> Dict[str, Any]:

    metadata: Dict[str, Any] = {
        "semantic_reasoning": True,
        "adjudicator": adjudicator,
        "polarity_conflict": polarity_conflict,
        "misconception_detected": misconception_detected,
        "task_type": task_type,
        "language": language,
        "deep_nli": False,
        "calibrated": False,
    }

    if similarity is not None:
        metadata["semantic_similarity"] = float(similarity)

    if semantic_method is not None:
        metadata["semantic_method"] = semantic_method

    return metadata


# ============================================================
# BASELINE ADJUDICATOR
# ============================================================

class BaselineAdjudicator:
    """
    Conservative adjudicator for environments where the
    semantic embedding model is unavailable.
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
                rationale="The response contains no evaluable evidence.",
                evidence=response_text,
                metadata={
                    "adjudicator": "baseline",
                    "semantic_reasoning": False,
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
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
                    "semantic_reasoning": False,
                    "deep_nli": False,
                    "calibrated": False,
                    "task_type": task_type,
                    "language": language,
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
                "semantic_reasoning": False,
                "deep_nli": False,
                "calibrated": False,
                "task_type": task_type,
                "language": language,
            },
        )


# ============================================================
# HYBRID CONCEPTUAL ADJUDICATOR
# ============================================================

class HybridConceptualAdjudicator:
    """
    Evalia hybrid conceptual adjudicator v2.

    Combines:

    1. multilingual semantic similarity;
    2. explicit polarity / negation conflict;
    3. specification-driven misconception detection;
    4. conservative generic conceptual conflict detection;
    5. structured adjudication relations.

    Current relation space:

        correct
        partially_correct
        misconception
        contradictory
        irrelevant
        insufficient_evidence
        undetermined

    Important:
    This is not deep natural-language inference.
    Confidence values and thresholds are heuristic and
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
        # 1. EMPTY RESPONSE
        # ----------------------------------------------------

        if not response:
            return AdjudicationResult(
                relation="insufficient_evidence",
                score=0.0,
                confidence=1.0,
                rationale=(
                    "The response contains no evaluable evidence."
                ),
                evidence=response_text,
                metadata=_base_metadata(
                    adjudicator="hybrid_v2",
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 2. VERY SHORT RESPONSE
        # ----------------------------------------------------

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
                metadata=_base_metadata(
                    adjudicator="hybrid_v2",
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 3. SEMANTIC SIMILARITY
        # ----------------------------------------------------

        semantic_result = self.semantic_matcher.compare(
            response_text=response,
            reference_text=reference,
            threshold=self.partial_threshold,
        )

        similarity = float(semantic_result.similarity)
        semantic_method = semantic_result.method

        # ----------------------------------------------------
        # 4. EXPLICIT POLARITY CONTRADICTION
        # ----------------------------------------------------

        polarity_conflict = _negation_mismatch(
            response_text=response,
            reference_text=reference,
        )

        if (
            polarity_conflict
            and similarity
            >= self.contradiction_relevance_threshold
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
                    "reference but presents an opposing explicit "
                    "polarity."
                ),
                evidence=response_text,
                contradiction=True,
                partial=False,
                metadata=_base_metadata(
                    adjudicator="hybrid_v2",
                    similarity=similarity,
                    semantic_method=semantic_method,
                    polarity_conflict=True,
                    misconception_detected=False,
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 5. DECLARED MISCONCEPTIONS
        # ----------------------------------------------------

        declared_misconceptions = (
            _declared_incompatible_concepts(
                response_text=response,
                context=context,
            )
        )

        if (
            declared_misconceptions
            and similarity
            >= self.misconception_relevance_threshold
        ):
            confidence = _bounded(
                0.68 + (0.28 * similarity)
            )

            metadata = _base_metadata(
                adjudicator="hybrid_v2",
                similarity=similarity,
                semantic_method=semantic_method,
                polarity_conflict=False,
                misconception_detected=True,
                task_type=task_type,
                language=language,
            )

            metadata.update(
                {
                    "misconception_source": (
                        "assessment_specification"
                    ),
                    "detected_incompatible_concepts": (
                        declared_misconceptions
                    ),
                }
            )

            return AdjudicationResult(
                relation="misconception",
                score=0.0,
                confidence=confidence,
                rationale=(
                    "The response is semantically related to the "
                    "target concept but contains a concept explicitly "
                    "declared as incompatible with the expected answer."
                ),
                evidence=", ".join(declared_misconceptions),
                contradiction=False,
                partial=False,
                metadata=metadata,
            )

        # ----------------------------------------------------
        # 6. GENERIC CONCEPTUAL CONFLICT
        # ----------------------------------------------------

        generic_conflicts = _generic_conceptual_conflicts(
            response_text=response,
            reference_text=reference,
        )

        if (
            generic_conflicts
            and similarity
            >= self.misconception_relevance_threshold
        ):
            confidence = _bounded(
                0.62 + (0.25 * similarity)
            )

            metadata = _base_metadata(
                adjudicator="hybrid_v2",
                similarity=similarity,
                semantic_method=semantic_method,
                polarity_conflict=False,
                misconception_detected=True,
                task_type=task_type,
                language=language,
            )

            metadata.update(
                {
                    "misconception_source": (
                        "generic_conceptual_conflict"
                    ),
                    "conceptual_conflicts": generic_conflicts,
                }
            )

            return AdjudicationResult(
                relation="misconception",
                score=0.0,
                confidence=confidence,
                rationale=(
                    "The response is semantically related to the "
                    "target but contains a conceptual relation that "
                    "conflicts with the reference."
                ),
                evidence=response_text,
                contradiction=False,
                partial=False,
                metadata=metadata,
            )

        # ----------------------------------------------------
        # 7. IRRELEVANT
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
                    "The response shows insufficient semantic "
                    "relation to the expected content."
                ),
                evidence=response_text,
                contradiction=False,
                partial=False,
                metadata=_base_metadata(
                    adjudicator="hybrid_v2",
                    similarity=similarity,
                    semantic_method=semantic_method,
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 8. UNDETERMINED
        # ----------------------------------------------------

        if similarity < self.partial_threshold:
            return AdjudicationResult(
                relation="undetermined",
                score=0.0,
                confidence=0.50,
                rationale=(
                    "The response is related to the target but "
                    "does not provide enough evidence for a stable "
                    "conceptual classification."
                ),
                evidence=response_text,
                contradiction=False,
                partial=False,
                metadata=_base_metadata(
                    adjudicator="hybrid_v2",
                    similarity=similarity,
                    semantic_method=semantic_method,
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 9. PARTIALLY CORRECT
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
                    "The response provides meaningful semantic "
                    "support but does not reach the conservative "
                    "threshold for full conceptual correspondence."
                ),
                evidence=response_text,
                contradiction=False,
                partial=True,
                metadata=_base_metadata(
                    adjudicator="hybrid_v2",
                    similarity=similarity,
                    semantic_method=semantic_method,
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 10. CORRECT
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
                "with the reference and no explicit conceptual "
                "conflict was detected."
            ),
            evidence=response_text,
            contradiction=False,
            partial=False,
            metadata=_base_metadata(
                adjudicator="hybrid_v2",
                similarity=similarity,
                semantic_method=semantic_method,
                task_type=task_type,
                language=language,
            ),
        )


# ============================================================
# DEFAULT ENGINE
# ============================================================

DEFAULT_ADJUDICATOR = HybridConceptualAdjudicator()


# ============================================================
# PUBLIC API
# ============================================================

def adjudicate(
    response_text: str,
    reference_text: str,
    task_type: Optional[str] = None,
    language: str = "auto",
    context: Optional[Dict[str, Any]] = None,
    engine: Optional[SemanticAdjudicator] = None,
) -> AdjudicationResult:
    """
    Public conceptual adjudication API.

    This function intentionally hides the adjudicator
    implementation from the rest of Evalia Core.
    """

    adjudicator = engine or DEFAULT_ADJUDICATOR

    return adjudicator.judge(
        response_text=response_text,
        reference_text=reference_text,
        task_type=task_type,
        language=language,
        context=context,
    )
