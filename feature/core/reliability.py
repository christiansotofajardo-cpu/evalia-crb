from __future__ import annotations

from typing import List

from .models import (
    AssessmentResult,
    DiagnosisResult,
    EvaluationInput,
    ReliabilityResult,
    RepresentationResult,
)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _source_risk(data: EvaluationInput) -> float:
    """
    Riesgo asociado a la fuente de la respuesta.

    OCR o captura con baja confianza aumenta el riesgo.
    Texto directo parte con riesgo bajo.
    """
    source = str(data.response.source or "text").strip().lower()

    if source in {"text", "manual", "typed"}:
        return 0.05

    confidence = data.response.source_confidence

    if confidence is None:
        return 0.25

    return _clamp(1.0 - float(confidence))


def _task_type_risk(task_type: str) -> float:
    """
    Baseline inicial de riesgo por tipo de tarea.

    Estos valores NO son calibración definitiva.
    Más adelante podrán aprenderse empíricamente.
    """
    normalized = str(task_type or "").strip().lower()

    if normalized in {
        "true_false",
        "short_exact_answer",
        "completion",
    }:
        return 0.30

    if normalized in {
        "enumeration",
        "classification_matching",
    }:
        return 0.20

    if normalized in {
        "causal_explanation",
        "justification",
        "differentiation",
        "experimental_interpretation",
        "constructed_response",
    }:
        return 0.15

    return 0.20


def _diagnostic_risk(diagnosis: DiagnosisResult) -> float:
    severity = str(diagnosis.error_severity or "").strip().lower()

    if severity == "high":
        return 0.40
    if severity == "medium":
        return 0.20
    if severity == "low":
        return 0.05

    return 0.15


def estimate_reliability(
    data: EvaluationInput,
    representation: RepresentationResult,
    judgment: AssessmentResult,
    diagnosis: DiagnosisResult,
) -> ReliabilityResult:
    """
    Estima de forma separada:

    - confianza interna del juicio;
    - riesgo estimado de desacuerdo con revisión humana.

    Esta separación es central en Evalia Core 2.0.
    """

    risk_factors: List[str] = []

    coverage = _clamp(representation.conceptual_coverage)

    if judgment.criterion_judgments:
        criterion_confidences = [
            _clamp(item.confidence)
            for item in judgment.criterion_judgments
        ]

        confidence = (
            sum(criterion_confidences)
            / len(criterion_confidences)
        )
    else:
        confidence = coverage

    source_risk = _source_risk(data)
    task_risk = _task_type_risk(data.task.task_type)
    diagnostic_risk = _diagnostic_risk(diagnosis)

    coverage_risk = 1.0 - coverage

    if coverage < 0.50:
        risk_factors.append("low_conceptual_coverage")

    if source_risk >= 0.30:
        risk_factors.append("source_uncertainty")

    if task_risk >= 0.30:
        risk_factors.append("task_type_risk")

    if diagnosis.error_severity == "high":
        risk_factors.append("high_diagnostic_severity")

    if representation.response_profile == "empty":
        risk_factors.append("empty_response")

    disagreement_risk = (
        0.35 * coverage_risk
        + 0.25 * source_risk
        + 0.20 * task_risk
        + 0.20 * diagnostic_risk
    )

    disagreement_risk = _clamp(disagreement_risk)

    if disagreement_risk < 0.20:
        reliability_class = "high"
    elif disagreement_risk < 0.45:
        reliability_class = "medium"
    else:
        reliability_class = "low"

    return ReliabilityResult(
        confidence=round(_clamp(confidence), 3),
        disagreement_risk=round(disagreement_risk, 3),
        reliability_class=reliability_class,
        risk_factors=risk_factors,
        metadata={
            "mode": "baseline_task_aware_risk",
            "coverage_risk": round(coverage_risk, 3),
            "source_risk": round(source_risk, 3),
            "task_type_risk": round(task_risk, 3),
            "diagnostic_risk": round(diagnostic_risk, 3),
        },
    )
