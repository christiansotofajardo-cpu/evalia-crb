
from __future__ import annotations

"""Evalia 3.0: orquestador de rúbrica universal y núcleo semántico.

Conserva el circuito histórico para rúbricas anteriores. Para la rúbrica v3,
separa rutas cerradas, semánticas y especializadas, y adjudica cada criterio
por separado antes de puntuar. En esta fase todas las notas v3 son provisionales.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict

from .adjudication import AdjudicationResult, adjudicate
from .assessment import assess as legacy_assess
from .delegation import decide_delegation
from .feedback import generate_feedback
from .models import (
    DelegationResult, EvaliaResult, EvaluationInput, RepresentationResult,
)
from .reliability import estimate_reliability
from .representation import represent
from .rubric_scoring_v3 import assess_v3

RUBRIC_VERSION = "evalia-rubric-3.0-1"
CORE_VERSION = "3.0-rubric-criterion-pilot"

NONSEMANTIC_ROUTES = {
    "exact",
    "controlled",
    "numeric_specialist_required",
    "math_specialist_required",
}

ALLOWED_ROUTES = NONSEMANTIC_ROUTES | {"semantic"}


def _plain_representation(
    data: EvaluationInput,
) -> RepresentationResult:
    """Impide usar embeddings para letras, claves y resultados numericos."""
    response = str(data.response.text or "").strip()

    language = (
        data.task.language
        if data.task.language != "auto"
        else data.language
    )

    return RepresentationResult(
        language=language if language != "auto" else "unknown",
        response_profile="very_brief" if response else "empty",
        metadata={
            "mode": "nonsemantic_route",
            "criteria_total": 0,
        },
    )


def _criterion_adjudications(
    data: EvaluationInput,
    representation: RepresentationResult,
) -> tuple[Dict[str, AdjudicationResult], list[str]]:
    """Adjudica criterios individuales sin concatenar toda la pauta.

    Si falla el motor semantico, no inventa un resultado:
    marca el criterio como indeterminado para revision humana.
    """
    spec = data.task.assessment_spec
    decisions: Dict[str, AdjudicationResult] = {}
    failures: list[str] = []

    if spec is None or not str(data.response.text or "").strip():
        return decisions, failures

    for criterion in spec.criteria:
        context = {
            "conceptual_units": [
                {
                    "id": criterion.id,
                    "description": criterion.description,
                    "variants": (
                        list(criterion.semantic_variants)
                        + list(criterion.accepted_values)
                    ),
                    "required": True,
                    "weight": 1.0,
                }
            ],
            "incompatible_concepts": list(
                criterion.contradictory_values
            ),
        }

        try:
            decisions[criterion.id] = adjudicate(
                response_text=data.response.text,
                reference_text=criterion.description,
                task_type=data.task.task_type,
                language=representation.language,
                context=context,
            )

        except RuntimeError as exc:
            # No usar una alternativa lexica oculta para simular
            # una evaluacion semantica confiable.
            failures.append(criterion.id)

            decisions[criterion.id] = AdjudicationResult(
                relation="undetermined",
                score=0.0,
                confidence=0.0,
                rationale=(
                    "Motor semantico no disponible; "
                    "requiere revision humana."
                ),
                metadata={
                    "adjudicator": "unavailable",
                    "error_type": type(exc).__name__,
                    "calibrated": False,
                },
            )

    return decisions, failures


def _legacy_adjudication(
    data: EvaluationInput,
    representation: RepresentationResult,
):
    """Mantiene el analisis global usado antes de la rubrica v3."""
    spec = data.task.assessment_spec
    units = []
    incompatible = []

    if spec is not None and spec.criteria:
        for criterion in spec.criteria:
            units.append(
                {
                    "id": criterion.id,
                    "description": criterion.description,
                    "variants": (
                        criterion.semantic_variants
                        + criterion.accepted_values
                    ),
                    "required": criterion.required,
                    "weight": criterion.weight,
                }
            )
            incompatible.extend(criterion.contradictory_values)

    reference = (
        " ".join(c.description for c in spec.criteria)
        if spec is not None and spec.criteria
        else data.task.prompt
    )

    return adjudicate(
        response_text=data.response.text,
        reference_text=reference,
        task_type=data.task.task_type,
        language=representation.language,
        context={
            "conceptual_units": units,
            "incompatible_concepts": incompatible,
        },
    )


def _trace(
    data: EvaluationInput,
    representation: RepresentationResult,
    judgment: Any,
    diagnosis: Any,
    reliability: Any,
    delegation: DelegationResult,
    *,
    version: str,
    route: str,
) -> Dict[str, Any]:
    is_v3 = version == CORE_VERSION

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "assessment_id": data.assessment_id,
        "task_id": data.task.id,
        "learner_id": data.learner.id,
        "language": representation.language,
        "response_source": data.response.source,
        "source_confidence": data.response.source_confidence,
        "score": judgment.score,
        "max_score": judgment.max_score,
        "error_type": diagnosis.error_type,
        "error_severity": diagnosis.error_severity,
        "confidence": reliability.confidence,
        "disagreement_risk": reliability.disagreement_risk,
        "reliability_class": reliability.reliability_class,
        "delegation_decision": delegation.decision,
        "delegation_reason": delegation.reason,
        "core_version": version,
        "rubric_version": (
            RUBRIC_VERSION if is_v3 else "legacy"
        ),
        "scoring_route": route,
        "score_provisional": is_v3,
        "unscored_criteria": list(
            judgment.metadata.get("unscored_criteria", [])
        ),
        "final_grade_authorized": (
            False if is_v3 else None
        ),
        "calibration_status": (
            "not_validated"
            if is_v3
            else "not_assessed_here"
        ),
    }


def evaluate(data: EvaluationInput) -> EvaliaResult:
    """Ejecuta representacion, adjudicacion, puntuacion, riesgo y feedback."""
    spec = data.task.assessment_spec

    policy = (
        spec.scoring_policy
        if spec is not None
        else {}
    )

    policy = (
        policy
        if isinstance(policy, dict)
        else {}
    )

    is_v3 = (
        policy.get("rubric_version")
        == RUBRIC_VERSION
    )

    route = str(
        policy.get("scoring_route", "semantic")
    )

    if is_v3 and policy.get("teacher_approved") is not True:
        raise ValueError(
            "La rubrica v3 no tiene aprobacion docente."
        )

    if is_v3 and route not in ALLOWED_ROUTES:
        raise ValueError(
            "Ruta de puntuacion v3 no reconocida: " + route
        )

    representation_failure = ""

    if is_v3 and route in NONSEMANTIC_ROUTES:
        representation = _plain_representation(data)

    elif is_v3:
        try:
            representation = represent(data)

        except RuntimeError as exc:
            # Un modelo ausente no implica respuesta incorrecta.
            # No adjudicar hasta poder representar la respuesta.
            representation_failure = type(exc).__name__
            representation = _plain_representation(data)

            representation.metadata["mode"] = (
                "semantic_representation_unavailable"
            )
            representation.metadata["requires_human_review"] = True

    else:
        representation = represent(data)

    if is_v3:
        if route == "semantic" and not representation_failure:
            decisions, failures = _criterion_adjudications(
                data,
                representation,
            )
        else:
            decisions, failures = {}, []

        judgment, diagnosis = assess_v3(
            data,
            representation,
            decisions,
        )

        adjudication_data: Dict[str, Any] = {
            "mode": "per_criterion",
            "scoring_route": route,
            "provisional": True,
            "failed_criteria": failures,
            "representation_failure": representation_failure,
            "criteria": {
                key: asdict(value)
                for key, value in decisions.items()
            },
        }

        version = CORE_VERSION

    else:
        whole = _legacy_adjudication(
            data,
            representation,
        )

        judgment, diagnosis = legacy_assess(
            data,
            representation,
        )

        adjudication_data = asdict(whole)
        version = "2.0-baseline"

    reliability = estimate_reliability(
        data,
        representation,
        judgment,
        diagnosis,
    )

    delegation = decide_delegation(
        data,
        representation,
        judgment,
        diagnosis,
        reliability,
    )

    # Durante el piloto, una prediccion v3 nunca se convierte
    # automaticamente en una calificacion definitiva.
    if (
        is_v3
        and delegation.decision in {
            "AUTO_ACCEPT",
            "ACCEPT_WITH_CAUTION",
        }
    ):
        delegation = DelegationResult(
            decision="HUMAN_REVIEW",
            reason="v3_provisional_scoring_requires_review",
            metadata={
                **delegation.metadata,
                "final_grade_authorized": False,
            },
        )

    feedback = generate_feedback(
        data,
        representation,
        judgment,
        diagnosis,
        reliability,
        delegation,
    )

    trace = _trace(
        data,
        representation,
        judgment,
        diagnosis,
        reliability,
        delegation,
        version=version,
        route=route if is_v3 else "legacy",
    )

    return EvaliaResult(
        assessment_id=data.assessment_id,
        task_id=data.task.id,
        learner_id=data.learner.id,
        language=representation.language,
        representation=representation,
        judgment=judgment,
        diagnosis=diagnosis,
        reliability=reliability,
        delegation=delegation,
        feedback=feedback,
        adjudication=adjudication_data,
        traceability=trace,
    )
