
from __future__ import annotations

"""Evalia 3.0: decisión conservadora y auditable de revisión humana.

Este módulo NO asigna puntos ni infiere comprensión. Recibe la evaluación
calculada por las otras capas y decide si requiere revisión docente.

Política segura por defecto: las decisiones automáticas quedan deshabilitadas
hasta contar con validación independiente y autorización explícita.
"""

import math
from typing import Any, Dict, Optional

from .models import (
    AssessmentResult,
    DelegationResult,
    DiagnosisResult,
    EvaluationInput,
    ReliabilityResult,
    RepresentationResult,
)

POLICY_VERSION = "evalia-3.0-validation-guard-v1"
DEFAULT_CAUTION_THRESHOLD = 0.20
DEFAULT_HUMAN_REVIEW_THRESHOLD = 0.45
DEFAULT_SOURCE_REVIEW_THRESHOLD = 0.50


def _probability(value: Any) -> Optional[float]:
    """Acepta solo números finitos en [0, 1]; nunca interpreta booleanos."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _policy(data: EvaluationInput) -> Dict[str, Any]:
    spec = data.task.assessment_spec
    value = spec.reliability_policy if spec is not None else None
    return value if isinstance(value, dict) else {}


def _result(
    decision: str,
    reason: str,
    *,
    reliability: ReliabilityResult,
    risk: Optional[float],
    source_confidence: Optional[float],
    proposed_decision: str = "HUMAN_REVIEW",
    extra: Optional[Dict[str, Any]] = None,
) -> DelegationResult:
    """Estandariza metadatos sin afirmar calibración estadística."""
    metadata: Dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "mode": "validation_first",
        "risk": risk,
        "confidence": _probability(reliability.confidence),
        "reliability_class": reliability.reliability_class,
        "risk_factors": list(reliability.risk_factors or []),
        "source_confidence": source_confidence,
        "risk_based_proposal": proposed_decision,
        "final_grade_authorized": False,
    }
    if extra:
        metadata.update(extra)
    return DelegationResult(decision=decision, reason=reason, metadata=metadata)


def decide_delegation(
    data: EvaluationInput,
    representation: RepresentationResult,
    judgment: AssessmentResult,
    diagnosis: DiagnosisResult,
    reliability: ReliabilityResult,
) -> DelegationResult:
    """Decide delegación sin modificar las notas ni la rúbrica.

    Estados conservados: INSUFFICIENT_EVIDENCE, HUMAN_REVIEW,
    ACCEPT_WITH_CAUTION y AUTO_ACCEPT.

    Los dos indicadores internos `validation_approved` y
    `auto_accept_enabled` deben ser exactamente True para permitir
    AUTO_ACCEPT. No se habilitan automáticamente ni se exponen al profesor.

    `ACCEPT_WITH_CAUTION` nunca equivale aquí a una nota definitiva.
    """
    del judgment  # Firma compatible; la nota no determina por sí sola la fiabilidad.

    risk = _probability(reliability.disagreement_risk)
    raw_source = data.response.source_confidence
    source_confidence = _probability(raw_source)
    policy = _policy(data)

    def result(
        decision: str,
        reason: str,
        *,
        proposed_decision: str = "HUMAN_REVIEW",
        extra: Optional[Dict[str, Any]] = None,
    ) -> DelegationResult:
        return _result(
            decision,
            reason,
            reliability=reliability,
            risk=risk,
            source_confidence=source_confidence,
            proposed_decision=proposed_decision,
            extra=extra,
        )

    if representation.response_profile == "empty":
        return result("INSUFFICIENT_EVIDENCE", "empty_response")

    spec = data.task.assessment_spec
    if spec is None or not spec.criteria:
        return result("HUMAN_REVIEW", "missing_assessment_criteria")

    if raw_source is not None and source_confidence is None:
        return result("HUMAN_REVIEW", "invalid_source_confidence")

    source_threshold = _probability(
        policy.get("source_review_threshold", DEFAULT_SOURCE_REVIEW_THRESHOLD)
    )
    caution_threshold = _probability(
        policy.get("caution_threshold", DEFAULT_CAUTION_THRESHOLD)
    )
    human_threshold = _probability(
        policy.get("human_review_threshold", DEFAULT_HUMAN_REVIEW_THRESHOLD)
    )

    if (
        source_threshold is None
        or caution_threshold is None
        or human_threshold is None
        or caution_threshold > human_threshold
    ):
        return result("HUMAN_REVIEW", "invalid_reliability_policy")

    if source_confidence is not None and source_confidence < source_threshold:
        return result("HUMAN_REVIEW", "low_source_confidence")

    # Las alertas conceptuales se revisan; no se convierten en descuentos
    # automáticos porque su detección también puede ser imperfecta.
    if representation.contradictions or diagnosis.misconceptions:
        return result("HUMAN_REVIEW", "conceptual_conflict_detected")

    # La severidad describe el posible error del estudiante, no la
    # certeza del algoritmo. En esta fase inicial se revisan los casos
    # graves como precaución, sin alterar su puntaje.
    if diagnosis.error_severity == "high":
        return result("HUMAN_REVIEW", "high_diagnostic_severity")

    if risk is None:
        return result("HUMAN_REVIEW", "invalid_disagreement_risk")

    if risk >= human_threshold:
        proposal = "HUMAN_REVIEW"
    elif risk >= caution_threshold:
        proposal = "ACCEPT_WITH_CAUTION"
    else:
        proposal = "AUTO_ACCEPT"

    if proposal == "HUMAN_REVIEW":
        return result("HUMAN_REVIEW", "high_disagreement_risk")

    # Un permiso aislado no basta para declarar validado el sistema.
    approved = policy.get("validation_approved") is True
    enabled = policy.get("auto_accept_enabled") is True
    if not (approved and enabled):
        return result(
            "HUMAN_REVIEW",
            "automatic_decisions_disabled_until_validation",
            proposed_decision=proposal,
            extra={"validation_approved": approved, "auto_accept_enabled": enabled},
        )

    if proposal == "ACCEPT_WITH_CAUTION":
        return result(
            "ACCEPT_WITH_CAUTION",
            "moderate_disagreement_risk",
            proposed_decision=proposal,
        )

    # Aceptación de la decisión, no certificación de la nota académica.
    # El cierre administrativo de calificaciones corresponde a otra capa.
    return result(
        "AUTO_ACCEPT",
        "low_disagreement_risk",
        proposed_decision=proposal,
    )
    
