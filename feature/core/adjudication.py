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


def _flexional_root(token: str) -> str:
    """
    Conservative lexical-flexional root.

    This is not a general-purpose stemmer or lemmatizer.
    It only provides a weak lexical signal for obvious
    inflectional variants. Semantic equivalence remains
    the responsibility of the semantic matcher.
    """
    token = normalize_semantic_text(token).strip()

    if len(token) < 5:
        return token

    # Spanish infinitives and common verbal inflections.
    endings = (
        "ando", "iendo",
        "ados", "adas", "idos", "idas",
        "ado", "ada", "ido", "ida",
        "amos", "emos", "imos",
        "an", "en",
        "ar", "er", "ir",
        "a", "e",
    )

    for ending in endings:
        if token.endswith(ending):
            root = token[:-len(ending)]
            if len(root) >= 4:
                return root

    # Conservative English inflections.
    # Preserve lexical bases ending in -ss (e.g., process),
    # while allowing processes -> process through the -es rule.
    if token.endswith("ss"):
        return token

    for ending in ("ing", "ed", "es", "s"):
        if token.endswith(ending):
            root = token[:-len(ending)]
            if len(root) >= 4:
                return root

    return token


def _flexional_overlap(
    response_text: str,
    reference_text: str,
) -> float:
    """
    Weak lexical-flexional evidence between two texts.

    Returns the proportion of reference lexical roots
    represented in the response. This signal complements,
    but never replaces, semantic similarity.
    """
    response_roots = {
        _flexional_root(token)
        for token in _tokens(response_text)
        if len(token) >= 4
    }

    reference_roots = [
        _flexional_root(token)
        for token in _tokens(reference_text)
        if len(token) >= 4
    ]

    if not response_roots or not reference_roots:
        return 0.0

    matched = sum(
        1 for root in reference_roots
        if root in response_roots
    )

    return _bounded(matched / len(reference_roots))


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



def _conceptual_units_from_context(
    context: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normalize conceptual units supplied by AssessmentSpec/context.

    Accepted forms:

        "maintain information temporarily"

    or:

        {
            "id": "temporary_maintenance",
            "description": "maintain information temporarily",
            "variants": [
                "keep information active",
                "hold information temporarily"
            ],
            "required": True,
            "weight": 1.0
        }

    Conceptual units are deliberately specification-driven.
    Evalia does not infer required concepts from the reference text.
    """

    if not context:
        return []

    raw_units = context.get("conceptual_units") or []
    units: List[Dict[str, Any]] = []

    for index, raw in enumerate(raw_units):

        if isinstance(raw, str):
            description = raw.strip()

            if description:
                units.append(
                    {
                        "id": f"concept_{index + 1}",
                        "description": description,
                        "variants": [],
                        "required": True,
                        "weight": 1.0,
                    }
                )

            continue

        if not isinstance(raw, dict):
            continue

        description = str(
            raw.get("description")
            or raw.get("concept")
            or raw.get("text")
            or ""
        ).strip()

        variants = raw.get("variants") or raw.get(
            "semantic_variants"
        ) or []

        if isinstance(variants, str):
            variants = [variants]

        variants = [
            str(value).strip()
            for value in variants
            if str(value).strip()
        ]

        if not description and not variants:
            continue

        units.append(
            {
                "id": str(
                    raw.get("id") or f"concept_{index + 1}"
                ),
                "description": description,
                "variants": variants,
                "required": bool(raw.get("required", True)),
                "weight": float(raw.get("weight", 1.0)),
            }
        )

    return units


def _evaluate_conceptual_coverage(
    response_text: str,
    conceptual_units: List[Dict[str, Any]],
    semantic_matcher: SemanticMatcher,
    unit_threshold: float,
) -> Dict[str, Any]:
    """
    Evaluate each required conceptual unit independently.

    This prevents high whole-answer similarity from being treated as
    evidence of complete conceptual coverage.
    """

    if not conceptual_units:
        return {
            "available": False,
            "coverage": None,
            "required_coverage": None,
            "matched_units": [],
            "missing_units": [],
            "unit_results": [],
        }

    unit_results: List[Dict[str, Any]] = []
    matched_units: List[str] = []
    missing_units: List[str] = []

    total_weight = 0.0
    matched_weight = 0.0

    required_weight = 0.0
    required_matched_weight = 0.0

    for unit in conceptual_units:

        unit_id = unit["id"]
        description = unit["description"]
        variants = unit["variants"]
        required = unit["required"]
        weight = max(float(unit["weight"]), 0.0)

        references = []

        if description:
            references.append(description)

        references.extend(variants)

        best_similarity = 0.0
        best_reference = ""
        best_method = ""

        best_flexional_overlap = 0.0
        best_flexional_reference = ""

        for reference in references:
            result = semantic_matcher.compare(
                response_text=response_text,
                reference_text=reference,
                threshold=unit_threshold,
            )

            similarity = float(result.similarity)

            if similarity > best_similarity:
                best_similarity = similarity
                best_reference = reference
                best_method = result.method

            flexional_overlap = _flexional_overlap(
                response_text=response_text,
                reference_text=reference,
            )

            if flexional_overlap > best_flexional_overlap:
                best_flexional_overlap = flexional_overlap
                best_flexional_reference = reference

        semantic_match = best_similarity >= unit_threshold
        flexional_match = best_flexional_overlap >= 0.80

        matched = semantic_match or flexional_match

        total_weight += weight

        if matched:
            matched_weight += weight
            matched_units.append(unit_id)
        else:
            missing_units.append(unit_id)

        if required:
            required_weight += weight

            if matched:
                required_matched_weight += weight

        unit_results.append(
            {
                "id": unit_id,
                "description": description,
                "required": required,
                "weight": weight,
                "matched": matched,
                "similarity": best_similarity,
                "best_reference": best_reference,
                "semantic_method": best_method,
                "semantic_match": semantic_match,
                "flexional_overlap": best_flexional_overlap,
                "flexional_match": flexional_match,
                "best_flexional_reference": best_flexional_reference,
                "match_source": (
                    "semantic_and_flexional"
                    if semantic_match and flexional_match
                    else "semantic"
                    if semantic_match
                    else "flexional"
                    if flexional_match
                    else "none"
                ),
            }
        )

    coverage = (
        matched_weight / total_weight
        if total_weight > 0
        else 0.0
    )

    required_coverage = (
        required_matched_weight / required_weight
        if required_weight > 0
        else coverage
    )

    return {
        "available": True,
        "coverage": _bounded(coverage),
        "required_coverage": _bounded(required_coverage),
        "matched_units": matched_units,
        "missing_units": missing_units,
        "unit_results": unit_results,
    }


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
    Evalia hybrid conceptual adjudicator v3.

    Combines:

    1. multilingual semantic similarity;
    2. explicit polarity / negation conflict;
    3. specification-driven misconception detection;
    4. conservative generic conceptual conflict detection;
    5. specification-driven conceptual coverage;
    6. structured adjudication relations.

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
        conceptual_unit_threshold: float = 0.60,
        single_unit_evidence_threshold: float = 0.70,
        full_coverage_threshold: float = 0.85,
        partial_coverage_threshold: float = 0.34,
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

        self.conceptual_unit_threshold = conceptual_unit_threshold
        self.single_unit_evidence_threshold = single_unit_evidence_threshold
        self.full_coverage_threshold = full_coverage_threshold
        self.partial_coverage_threshold = partial_coverage_threshold

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
                    adjudicator="hybrid_v3",
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
                    adjudicator="hybrid_v3",
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
                    adjudicator="hybrid_v3",
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
                adjudicator="hybrid_v3",
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
                adjudicator="hybrid_v3",
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
        # 7. CONCEPTUAL COVERAGE
        # ----------------------------------------------------

        conceptual_units = _conceptual_units_from_context(context)

        coverage_result = _evaluate_conceptual_coverage(
            response_text=response,
            conceptual_units=conceptual_units,
            semantic_matcher=self.semantic_matcher,
            unit_threshold=self.conceptual_unit_threshold,
        )

        if coverage_result["available"]:

            coverage = float(coverage_result["coverage"])
            required_coverage = float(
                coverage_result["required_coverage"]
            )

            coverage_metadata = _base_metadata(
                adjudicator="hybrid_v3",
                similarity=similarity,
                semantic_method=semantic_method,
                task_type=task_type,
                language=language,
            )

            coverage_metadata.update(
                {
                    "conceptual_coverage": coverage,
                    "required_conceptual_coverage": (
                        required_coverage
                    ),
                    "matched_conceptual_units": (
                        coverage_result["matched_units"]
                    ),
                    "missing_conceptual_units": (
                        coverage_result["missing_units"]
                    ),
                    "conceptual_unit_results": (
                        coverage_result["unit_results"]
                    ),
                    "conceptual_unit_threshold": (
                        self.conceptual_unit_threshold
                    ),
                }
            )

            if (
                required_coverage
                >= self.full_coverage_threshold
            ):
                confidence = _bounded(
                    0.55
                    + (0.25 * required_coverage)
                    + (0.20 * similarity)
                )

                return AdjudicationResult(
                    relation="correct",
                    score=1.0,
                    confidence=confidence,
                    rationale=(
                        "The response provides sufficient coverage "
                        "of the required conceptual units and no "
                        "explicit conceptual conflict was detected."
                    ),
                    evidence=response_text,
                    contradiction=False,
                    partial=False,
                    metadata=coverage_metadata,
                )

            required_units_matched = sum(
                1
                for unit in coverage_result["unit_results"]
                if unit["required"] and unit["matched"]
            )

            matched_required_units = [
                unit
                for unit in coverage_result["unit_results"]
                if unit["required"] and unit["matched"]
            ]

            strongest_required_similarity = max(
                (
                    unit["similarity"]
                    for unit in matched_required_units
                ),
                default=0.0,
            )

            sufficient_partial_evidence = (
                required_units_matched >= 2
                or (
                    required_units_matched == 1
                    and strongest_required_similarity
                    >= self.single_unit_evidence_threshold
                )
            )

            if sufficient_partial_evidence:
                confidence = _bounded(
                    0.50
                    + (0.25 * required_coverage)
                    + (0.15 * similarity)
                )

                return AdjudicationResult(
                    relation="partially_correct",
                    score=0.5,
                    confidence=confidence,
                    rationale=(
                        "The response provides valid conceptual "
                        "evidence but does not cover all required "
                        "conceptual units."
                    ),
                    evidence=response_text,
                    contradiction=False,
                    partial=True,
                    metadata=coverage_metadata,
                )

            # Conceptual units exist but coverage is too weak.
            # Once conceptual requirements are explicitly supplied,
            # global similarity must not override insufficient coverage.

            confidence = _bounded(
                0.55 + (0.20 * (1.0 - required_coverage))
            )

            return AdjudicationResult(
                relation=(
                    "irrelevant"
                    if similarity < self.irrelevant_threshold
                    else "undetermined"
                ),
                score=0.0,
                confidence=confidence,
                rationale=(
                    "The response does not provide sufficient coverage "
                    "of the explicitly required conceptual units."
                ),
                evidence=response_text,
                contradiction=False,
                partial=False,
                metadata=coverage_metadata,
            )

        # ----------------------------------------------------
        # 8. IRRELEVANT
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
                    adjudicator="hybrid_v3",
                    similarity=similarity,
                    semantic_method=semantic_method,
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 9. UNDETERMINED
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
                    adjudicator="hybrid_v3",
                    similarity=similarity,
                    semantic_method=semantic_method,
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 10. PARTIALLY CORRECT
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
                    adjudicator="hybrid_v3",
                    similarity=similarity,
                    semantic_method=semantic_method,
                    task_type=task_type,
                    language=language,
                ),
            )

        # ----------------------------------------------------
        # 11. CORRECT
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
                adjudicator="hybrid_v3",
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
