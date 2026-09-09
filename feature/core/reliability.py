from __future__ import annotations

from statistics import pstdev
from typing import List

from .models import (
    AssessmentResult,
    DiagnosisResult,
    EvaluationInput,
    ReliabilityResult,
    RepresentationResult,
)


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, float(value)),
    )


def _source_risk(
    data: EvaluationInput,
) -> float:
    """
    Riesgo asociado a la fuente de la respuesta.

    Texto directo parte con riesgo bajo.
    OCR u otras fuentes pueden incorporar incertidumbre adicional.
    """

    source = str(
        data.response.source or "text"
    ).strip().lower()

    if source in {
        "text",
        "manual",
        "typed",
    }:
        return 0.05

    confidence = data.response.source_confidence

    if confidence is None:
        return 0.25

    return _clamp(
        1.0 - float(confidence)
    )


def _task_type_risk(
    data: EvaluationInput,
) -> float:
    """
    Riesgo basal por tipo de tarea.

    Puede ser sobrescrito mediante reliability_policy
    del AssessmentSpec.
    """

    task_type = str(
        data.task.task_type or ""
    ).strip().lower()

    default_risks = {
        "true_false": 0.10,
        "short_exact_answer": 0.10,
        "completion": 0.15,
        "enumeration": 0.20,
        "classification_matching": 0.20,
        "differentiation": 0.25,
        "causal_explanation": 0.30,
        "justification": 0.30,
        "experimental_interpretation": 0.35,
        "constructed_response": 0.30,
    }

    risk = default_risks.get(
        task_type,
        0.25,
    )

    spec = data.task.assessment_spec

    if spec is not None:
        policy = spec.reliability_policy or {}

        custom_risks = policy.get(
            "task_type_risk",
            {},
        )

        if isinstance(custom_risks, dict):
            try:
                risk = float(
                    custom_risks.get(
                        task_type,
                        risk,
                    )
                )
            except (TypeError, ValueError):
                pass

    return _clamp(risk)


def _diagnostic_risk(
    diagnosis: DiagnosisResult,
) -> float:
    """
    Traduce severidad diagnóstica en riesgo basal.
    """

    severity = str(
        diagnosis.error_severity or ""
    ).strip().lower()

    if severity == "high":
        return 0.40

    if severity == "medium":
        return 0.20

    if severity == "low":
        return 0.05

    return 0.15


def _semantic_strengths(
    judgment: AssessmentResult,
) -> List[float]:
    """
    Extrae similitudes semánticas por criterio.
    """

    strengths: List[float] = []

    for item in judgment.criterion_judgments:
        try:
            value = float(
                item.metadata.get(
                    "similarity",
                    item.confidence,
                )
            )
        except (TypeError, ValueError):
            continue

        strengths.append(
            _clamp(value)
        )

    return strengths


def _semantic_evidence_risk(
    judgment: AssessmentResult,
) -> tuple[float, float, float]:
    """
    Estima riesgo a partir de la fuerza y consistencia
    de la evidencia semántica.

    Devuelve:
    - riesgo semántico;
    - fuerza media;
    - dispersión.
    """

    strengths = _semantic_strengths(
        judgment
    )

    if not strengths:
        return 0.70, 0.0, 0.0

    average_strength = (
        sum(strengths)
        / len(strengths)
    )

    dispersion = (
        pstdev(strengths)
        if len(strengths) > 1
        else 0.0
    )

    weakness_risk = (
        1.0 - average_strength
    )

    dispersion_risk = _clamp(
        dispersion * 2.0
    )

    semantic_risk = (
        0.75 * weakness_risk
        + 0.25 * dispersion_risk
    )

    return (
        _clamp(semantic_risk),
        _clamp(average_strength),
        _clamp(dispersion),
    )


def estimate_reliability(
    data: EvaluationInput,
    representation: RepresentationResult,
    judgment: AssessmentResult,
    diagnosis: DiagnosisResult,
) -> ReliabilityResult:
    """
    Estima de forma separada:

    - confianza interna del juicio;
    - riesgo de desacuerdo con evaluación humana.

    El riesgo combina:
    - cobertura conceptual;
    - fuerza semántica;
    - dispersión entre criterios;
    - calidad de fuente;
    - tipo de tarea;
    - severidad diagnóstica.

    Este modelo sigue siendo heurístico y NO está calibrado
    empíricamente todavía.
    """

    risk_factors: List[str] = []

    coverage = _clamp(
        representation.conceptual_coverage
    )

    (
        semantic_risk,
        average_semantic_strength,
        semantic_dispersion,
    ) = _semantic_evidence_risk(
        judgment
    )

    source_risk = _source_risk(data)
    task_risk = _task_type_risk(data)
    diagnostic_risk = _diagnostic_risk(
        diagnosis
    )

    coverage_risk = (
        1.0 - coverage
    )

    confidence = _clamp(
        0.70 * average_semantic_strength
        + 0.20 * coverage
        + 0.10 * (1.0 - source_risk)
    )

    disagreement_risk = (
        0.30 * coverage_risk
        + 0.30 * semantic_risk
        + 0.15 * source_risk
        + 0.15 * task_risk
        + 0.10 * diagnostic_risk
    )

    if representation.response_profile == "empty":
        disagreement_risk = max(
            disagreement_risk,
            0.95,
        )
        risk_factors.append(
            "empty_response"
        )

    elif representation.response_profile == "very_brief":
        disagreement_risk += 0.05
        risk_factors.append(
            "very_brief_response"
        )

    if coverage < 0.50:
        risk_factors.append(
            "low_conceptual_coverage"
        )

    if average_semantic_strength < 0.60:
        risk_factors.append(
            "weak_semantic_evidence"
        )

    if semantic_dispersion >= 0.20:
        risk_factors.append(
            "inconsistent_semantic_evidence"
        )

    if source_risk >= 0.30:
        risk_factors.append(
            "source_uncertainty"
        )

    if task_risk >= 0.30:
        risk_factors.append(
            "task_type_risk"
        )

    if diagnosis.error_severity == "high":
        risk_factors.append(
            "high_diagnostic_severity"
        )

    if representation.contradictions:
        disagreement_risk += 0.10
        risk_factors.append(
            "conceptual_contradiction"
        )

    disagreement_risk = _clamp(
        disagreement_risk
    )

    if disagreement_risk < 0.20:
        reliability_class = "high"

    elif disagreement_risk < 0.45:
        reliability_class = "medium"

    else:
        reliability_class = "low"

    return ReliabilityResult(
        confidence=round(
            confidence,
            3,
        ),
        disagreement_risk=round(
            disagreement_risk,
            3,
        ),
        reliability_class=reliability_class,
        risk_factors=list(
            dict.fromkeys(risk_factors)
        ),
        metadata={
            "mode": "semantic_task_aware_risk",
            "calibrated": False,
            "conceptual_coverage": round(
                coverage,
                3,
            ),
            "coverage_risk": round(
                coverage_risk,
                3,
            ),
            "average_semantic_strength": round(
                average_semantic_strength,
                3,
            ),
            "semantic_dispersion": round(
                semantic_dispersion,
                3,
            ),
            "semantic_evidence_risk": round(
                semantic_risk,
                3,
            ),
            "source_risk": round(
                source_risk,
                3,
            ),
            "task_type_risk": round(
                task_risk,
                3,
            ),
            "diagnostic_risk": round(
                diagnostic_risk,
                3,
            ),
        },
    )
