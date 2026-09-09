from __future__ import annotations

from typing import Dict, List, Tuple

from .models import (
    AssessmentResult,
    CriterionJudgment,
    DiagnosisResult,
    EvaluationInput,
    RepresentationResult,
)


def _evidence_by_criterion(
    representation: RepresentationResult,
) -> Dict[str, Dict[str, object]]:
    """
    Organiza la mejor evidencia semántica detectada por criterion_id.
    """
    evidence_map: Dict[str, Dict[str, object]] = {}

    for item in representation.evidence_spans:
        criterion_id = str(item.get("criterion_id", "")).strip()

        if not criterion_id:
            continue

        evidence_map[criterion_id] = {
            "evidence": item.get("evidence", ""),
            "best_variant": item.get("best_variant", ""),
            "best_similarity": float(
                item.get("best_similarity", 0.0) or 0.0
            ),
            "match_method": item.get("match_method", ""),
            "matched_variants": item.get("matched_variants", []),
        }

    return evidence_map


def _criterion_support_strength(
    similarity: float,
    threshold: float,
) -> str:
    """
    Clasifica la fuerza de la evidencia semántica.
    """
    similarity = max(0.0, min(1.0, float(similarity)))
    threshold = max(0.0, min(1.0, float(threshold)))

    if similarity >= 0.90:
        return "strong"

    if similarity >= max(0.80, threshold):
        return "moderate"

    if similarity >= threshold:
        return "weak"

    return "insufficient"


def assess(
    data: EvaluationInput,
    representation: RepresentationResult,
) -> Tuple[AssessmentResult, DiagnosisResult]:
    """
    Convierte evidencia semántica estructurada en juicio evaluativo.

    Evalia considera ahora:
    - existencia de evidencia;
    - fuerza de similitud semántica;
    - peso del criterio;
    - obligatoriedad del criterio.

    El scoring sigue siendo conservador:
    un criterio solo puntúa cuando supera el umbral semántico.
    """

    spec = data.task.assessment_spec

    if spec is None or not spec.criteria:
        diagnosis = DiagnosisResult(
            strengths=[],
            gaps=[],
            misconceptions=[],
            error_type=(
                "empty_response"
                if representation.response_profile == "empty"
                else "insufficient_specification"
            ),
            error_severity="high",
            metadata={
                "mode": "semantic_strength_no_assessment_spec",
            },
        )

        return (
            AssessmentResult(
                score=0.0,
                max_score=float(data.task.max_score or 0.0),
                criterion_judgments=[],
                metadata={
                    "mode": "semantic_strength_no_assessment_spec",
                },
            ),
            diagnosis,
        )

    evidence_map = _evidence_by_criterion(
        representation
    )

    threshold = float(
        representation.metadata.get(
            "semantic_threshold",
            0.75,
        )
    )

    total_weight = sum(
        max(float(criterion.weight or 0.0), 0.0)
        for criterion in spec.criteria
    ) or 1.0

    criterion_judgments: List[CriterionJudgment] = []
    strengths: List[str] = []
    gaps: List[str] = []

    total_score = 0.0
    required_missing = 0

    similarities: List[float] = []

    for criterion in spec.criteria:
        semantic_evidence = evidence_map.get(
            criterion.id,
            {},
        )

        similarity = float(
            semantic_evidence.get(
                "best_similarity",
                0.0,
            ) or 0.0
        )

        satisfied = (
            bool(semantic_evidence)
            and similarity >= threshold
        )

        support_strength = _criterion_support_strength(
            similarity,
            threshold,
        )

        criterion_max = (
            float(data.task.max_score)
            * max(
                float(criterion.weight or 0.0),
                0.0,
            )
            / total_weight
        )

        criterion_score = (
            criterion_max
            if satisfied
            else 0.0
        )

        if satisfied:
            strengths.append(
                criterion.description
            )
            similarities.append(similarity)

        else:
            gaps.append(
                criterion.description
            )

            if criterion.required:
                required_missing += 1

        evidence_text = str(
            semantic_evidence.get(
                "evidence",
                "",
            )
        ).strip()

        criterion_judgments.append(
            CriterionJudgment(
                criterion_id=criterion.id,
                satisfied=satisfied,
                score=round(
                    criterion_score,
                    3,
                ),
                max_score=round(
                    criterion_max,
                    3,
                ),
                confidence=round(
                    similarity,
                    3,
                ),
                evidence=(
                    [evidence_text]
                    if evidence_text
                    else []
                ),
                reason=(
                    f"semantic_support_{support_strength}"
                    if satisfied
                    else "semantic_support_insufficient"
                ),
                metadata={
                    "required": criterion.required,
                    "weight": criterion.weight,
                    "similarity": round(
                        similarity,
                        3,
                    ),
                    "support_strength": support_strength,
                    "best_variant": semantic_evidence.get(
                        "best_variant",
                        "",
                    ),
                    "match_method": semantic_evidence.get(
                        "match_method",
                        "",
                    ),
                    "semantic_threshold": threshold,
                },
            )
        )

        total_score += criterion_score

    criteria_total = len(
        spec.criteria
    )
    criteria_satisfied = len(
        strengths
    )

    if representation.response_profile == "empty":
        error_type = "empty_response"
        severity = "high"

    elif criteria_satisfied == criteria_total:
        error_type = "no_evident_error"
        severity = "low"

    elif required_missing > 0:
        error_type = "required_concept_missing"
        severity = "high"

    elif strengths and gaps:
        error_type = "incomplete_response"
        severity = "medium"

    else:
        error_type = "insufficient_evidence"
        severity = "high"

    average_similarity = (
        sum(similarities) / len(similarities)
        if similarities
        else 0.0
    )

    diagnosis = DiagnosisResult(
        strengths=strengths,
        gaps=gaps,
        misconceptions=list(
            representation.contradictions
        ),
        error_type=error_type,
        error_severity=severity,
        metadata={
            "mode": "semantic_strength_assessment",
            "criteria_total": criteria_total,
            "criteria_satisfied": criteria_satisfied,
            "required_criteria_missing": required_missing,
            "conceptual_coverage": representation.conceptual_coverage,
            "average_supported_similarity": round(
                average_similarity,
                3,
            ),
        },
    )

    return (
        AssessmentResult(
            score=round(
                min(
                    total_score,
                    float(
                        data.task.max_score or 0.0
                    ),
                ),
                3,
            ),
            max_score=float(
                data.task.max_score or 0.0
            ),
            criterion_judgments=criterion_judgments,
            metadata={
                "mode": "semantic_strength_assessment",
                "criterion_weight_total": round(
                    total_weight,
                    3,
                ),
                "criteria_satisfied": criteria_satisfied,
                "criteria_total": criteria_total,
                "average_supported_similarity": round(
                    average_similarity,
                    3,
                ),
                "semantic_threshold": threshold,
            },
        ),
        diagnosis,
    )
