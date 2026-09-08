from __future__ import annotations

from typing import List, Tuple

from .models import (
    AssessmentResult,
    CriterionJudgment,
    DiagnosisResult,
    EvaluationInput,
    RepresentationResult,
)


def _criterion_max_score(data: EvaluationInput) -> float:
    """
    Distribuye el puntaje máximo de la tarea según los pesos
    definidos en el AssessmentSpec.
    """
    spec = data.task.assessment_spec

    if spec is None or not spec.criteria:
        return float(data.task.max_score or 0.0)

    total_weight = sum(
        max(float(criterion.weight or 0.0), 0.0)
        for criterion in spec.criteria
    )

    return total_weight or float(data.task.max_score or 0.0)


def _criterion_detected(
    criterion_description: str,
    representation: RepresentationResult,
) -> bool:
    """
    Baseline simple:
    considera satisfecho un criterio si su descripción o alguna evidencia
    equivalente aparece entre los conceptos detectados.
    """
    target = str(criterion_description or "").strip().lower()

    if not target:
        return False

    detected = {
        str(value or "").strip().lower()
        for value in representation.concepts_detected
    }

    return target in detected


def assess(
    data: EvaluationInput,
    representation: RepresentationResult,
) -> Tuple[AssessmentResult, DiagnosisResult]:
    """
    Segunda capa de Evalia Core 2.0.

    Convierte la evidencia representada en un juicio evaluativo básico.
    Este baseline será reemplazable por motores semánticos y de scoring
    más sofisticados sin cambiar el contrato externo.
    """
    spec = data.task.assessment_spec

    if spec is None or not spec.criteria:
        score = (
            float(data.task.max_score)
            if representation.conceptual_coverage >= 0.8
            else 0.0
        )

        diagnosis = DiagnosisResult(
            strengths=[],
            gaps=[],
            misconceptions=[],
            error_type=(
                "insufficient_specification"
                if representation.response_profile != "empty"
                else "empty_response"
            ),
            error_severity="high",
            metadata={
                "mode": "baseline_no_assessment_spec",
            },
        )

        return (
            AssessmentResult(
                score=round(score, 3),
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
    ) or 1.0

    criterion_judgments: List[CriterionJudgment] = []
    strengths: List[str] = []
    gaps: List[str] = []

    total_score = 0.0

    for criterion in spec.criteria:
        satisfied = _criterion_detected(
            criterion.description,
            representation,
        )

        criterion_max = (
            float(data.task.max_score)
            * max(float(criterion.weight or 0.0), 0.0)
            / total_weight
        )

        criterion_score = criterion_max if satisfied else 0.0

        if satisfied:
            strengths.append(criterion.description)
        else:
            gaps.append(criterion.description)

        criterion_judgments.append(
            CriterionJudgment(
                criterion_id=criterion.id,
                satisfied=satisfied,
                score=round(criterion_score, 3),
                max_score=round(criterion_max, 3),
                confidence=(
                    representation.conceptual_coverage
                    if satisfied
                    else 1.0 - representation.conceptual_coverage
                ),
                evidence=[
                    value
                    for value in representation.concepts_detected
                    if str(value).strip().lower()
                    == str(criterion.description).strip().lower()
                ],
                reason=(
                    "criterion_detected"
                    if satisfied
                    else "criterion_not_detected"
                ),
                metadata={
                    "required": criterion.required,
                    "weight": criterion.weight,
                },
            )
        )

        total_score += criterion_score

    if representation.response_profile == "empty":
        error_type = "empty_response"
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
            "criteria_total": len(spec.criteria),
            "criteria_satisfied": len(strengths),
            "mode": "baseline_rule_based",
        },
    )

    return (
        AssessmentResult(
            score=round(
                min(total_score, float(data.task.max_score or 0.0)),
                3,
            ),
            max_score=float(data.task.max_score or 0.0),
            criterion_judgments=criterion_judgments,
            metadata={
                "mode": "baseline_rule_based",
                "criterion_weight_total": round(total_weight, 3),
            },
        ),
        diagnosis,
    )
