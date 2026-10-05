from __future__ import annotations

from .models import (
    AssessmentResult,
    DelegationResult,
    DiagnosisResult,
    EvaluationInput,
    ReliabilityResult,
    RepresentationResult,
)


def decide_delegation(
    data: EvaluationInput,
    representation: RepresentationResult,
    judgment: AssessmentResult,
    diagnosis: DiagnosisResult,
    reliability: ReliabilityResult,
) -> DelegationResult:
    """
    Decide si Evalia puede aceptar el juicio automáticamente,
    si debe aceptarlo con cautela o si conviene revisión humana.

    Esta lógica es modular y podrá reemplazarse más adelante
    por políticas calibradas o modelos aprendidos.
    """

    if representation.response_profile == "empty":
        return DelegationResult(
            decision="INSUFFICIENT_EVIDENCE",
            reason="empty_response",
            metadata={
                "risk": reliability.disagreement_risk,
                "confidence": reliability.confidence,
            },
        )

    if data.response.source_confidence is not None:
        if float(data.response.source_confidence) < 0.50:
            return DelegationResult(
                decision="HUMAN_REVIEW",
                reason="low_source_confidence",
                metadata={
                    "risk": reliability.disagreement_risk,
                    "confidence": reliability.confidence,
                    "source_confidence": data.response.source_confidence,
                },
            )

    spec = data.task.assessment_spec
    policy = spec.reliability_policy if spec is not None else {}
    caution_threshold = float(policy.get("caution_threshold", 0.20))
    human_review_threshold = float(policy.get("human_review_threshold", 0.45))

    risk = float(reliability.disagreement_risk)

    if risk >= human_review_threshold:
        decision = "HUMAN_REVIEW"
        reason = "high_disagreement_risk"

    elif risk >= caution_threshold:
        decision = "ACCEPT_WITH_CAUTION"
        reason = "moderate_disagreement_risk"

    else:
        decision = "AUTO_ACCEPT"
        reason = "low_disagreement_risk"

    if diagnosis.error_severity == "high" and decision == "AUTO_ACCEPT":
        decision = "ACCEPT_WITH_CAUTION"
        reason = "diagnostic_severity_override"

    return DelegationResult(
        decision=decision,
        reason=reason,
        metadata={
            "risk": reliability.disagreement_risk,
            "confidence": reliability.confidence,
            "reliability_class": reliability.reliability_class,
            "risk_factors": reliability.risk_factors,
            "mode": "baseline_selective_delegation",
        },
    )
