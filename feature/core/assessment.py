from __future__ import annotations

from typing import Dict, List, Tuple

from .models import (
    AssessmentResult,
    Criterion,
    CriterionJudgment,
    DiagnosisResult,
    EvaluationInput,
    RepresentationResult,
)


def _criterion_evidence_map(
    representation: RepresentationResult,
) -> Dict[str, List[str]]:
    """
    Organiza la evidencia detectada por criterion_id.

    representation.evidence_spans actúa como puente entre
    la representación semántica y el juicio evaluativo.
    """
    evidence_map: Dict[str, List[str]] = {}

    for item in representation.evidence_spans:
        criterion_id = str(item.get("criterion_id", "")).strip()

        if not criterion_id:
            continue

        matched_variants = item.get("matched_variants", [])

        if not isinstance(matched_variants, list):
            matched_variants = []

        evidence_map[criterion_id] = [
            str(value)
            for value in matched_variants
            if str(value).strip()
        ]

    return evidence_map


def _criterion_max_score(
    criterion: Criterion,
    total_weight: float,
    task_max_score: float,
) -> float:
    """
    Calcula el puntaje máximo asignable a cada criterio
    a partir de su peso relativo.
    """
    weight = max(float(criterion.weight or 0.0), 0.0)

    if total_weight <= 0.0:
        return 0.0

    return (
        float(task_max_score or 0.0)
        * weight
        / total_weight
    )


def assess(
    data: EvaluationInput,
    representation: RepresentationResult,
) -> Tuple[AssessmentResult, DiagnosisResult]:
    """
    Convierte la evidencia semántica estructurada
    en un juicio evaluativo.

    El scoring trabaja sobre criterios conceptuales,
    no sobre coincidencias textuales independientes.
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
                "mode": "baseline_no_assessment_spec",
            },
        )

        return (
            AssessmentResult(
                score=0.0,
                max_score=float(data.task.max_score or 0.0),
                criterion_judgments=[],
                metadata={
                    "mode": "baseline_no_assessment_spec",
                },
            ),
            diagnosis,
        )

    total_weight = sum(
        max(float(criterion.weight or 0.0), 0.0)
        for criterion in spec.criteria
    )

    if total_weight <= 0.0:
        total_weight = float(len(spec.criteria)) or 1.0

    evidence_map = _criterion_evidence_map(representation)

    criterion_judgments: List[CriterionJudgment] = []
    strengths: List[str] = []
    gaps: List[str] = []

    total_score = 0.0

    for criterion in spec.criteria:
        evidence = evidence_map.get(criterion.id, [])
        satisfied = bool(evidence)

        criterion_max = _criterion_max_score(
            criterion=criterion,
            total_weight=total_weight,
            task_max_score=float(data.task.max_score or 0.0),
        )

        criterion_score = criterion_max if satisfied else 0.0

        if satisfied:
            strengths.append(criterion.description)
        else:
            gaps.append(criterion.description)

        criterion_confidence = (
            0.90
            if satisfied
            else max(
                0.0,
                1.0 - representation.conceptual_coverage,
            )
        )

        criterion_judgments.append(
            CriterionJudgment(
                criterion_id=criterion.id,
                satisfied=satisfied,
                score=round(criterion_score, 3),
                max_score=round(criterion_max, 3),
                confidence=round(criterion_confidence, 3),
                evidence=evidence,
                reason=(
                    "criterion_supported_by_evidence"
                    if satisfied
                    else "criterion_without_detected_evidence"
                ),
                metadata={
                    "required": criterion.required,
                    "weight": criterion.weight,
                    "evidence_count": len(evidence),
                },
            )
        )

        total_score += criterion_score

    required_gaps = [
        criterion.description
        for criterion in spec.criteria
        if criterion.required
        and criterion.id not in evidence_map
    ]

    if representation.response_profile == "empty":
        error_type = "empty_response"
        severity = "high"

    elif required_gaps:
        error_type = "missing_required_concepts"
        severity = "high"

    elif gaps and strengths:
        error_type = "incomplete_response"
        severity = "medium"

    elif gaps and not strengths:
        error_type = "insufficient_evidence"
        severity = "high"

    else:
        error_type = "no_evident_error"
        severity = "low"

    diagnosis = DiagnosisResult(
        strengths=strengths,
        gaps=gaps,
        misconceptions=list(representation.contradictions),
        error_type=error_type,
        error_severity=severity,
        metadata={
            "mode": "criterion_centered_assessment",
            "criteria_total": len(spec.criteria),
            "criteria_satisfied": len(strengths),
            "required_gaps": required_gaps,
        },
    )

    return (
        AssessmentResult(
            score=round(
                min(
                    total_score,
                    float(data.task.max_score or 0.0),
                ),
                3,
            ),
            max_score=float(data.task.max_score or 0.0),
            criterion_judgments=criterion_judgments,
            metadata={
                "mode": "criterion_centered_assessment",
                "criterion_weight_total": round(total_weight, 3),
            },
        ),
        diagnosis,
    )
