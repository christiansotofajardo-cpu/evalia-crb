from __future__ import annotations

from .models import (
    AssessmentResult,
    DelegationResult,
    DiagnosisResult,
    EvaluationInput,
    FeedbackResult,
    ReliabilityResult,
    RepresentationResult,
)


def _feedback_language(data: EvaluationInput, representation: RepresentationResult) -> str:
    """
    Determina el idioma de salida del feedback.
    Por ahora prioriza el idioma resuelto por la representación.
    """
    language = str(representation.language or "").strip().lower()

    if language in {"es", "en"}:
        return language

    candidate = str(data.language or "").strip().lower()

    if candidate in {"es", "en"}:
        return candidate

    return "es"


def _summary_es(
    diagnosis: DiagnosisResult,
    delegation: DelegationResult,
) -> str:
    if diagnosis.error_type == "empty_response":
        return "No hay evidencia suficiente para evaluar la respuesta."

    if delegation.decision == "HUMAN_REVIEW":
        return (
            "La respuesta contiene evidencia evaluable, pero el juicio requiere "
            "revisión humana antes de considerarse definitivo."
        )

    if diagnosis.error_type == "no_evident_error":
        return "La respuesta satisface los criterios evaluativos detectados."

    if diagnosis.error_type == "incomplete_response":
        return "La respuesta contiene elementos correctos, pero presenta aspectos incompletos."

    if diagnosis.error_type == "insufficient_evidence":
        return "La evidencia detectada no es suficiente para satisfacer los criterios actuales."

    return "Evalia generó un juicio evaluativo a partir de la evidencia disponible."


def _summary_en(
    diagnosis: DiagnosisResult,
    delegation: DelegationResult,
) -> str:
    if diagnosis.error_type == "empty_response":
        return "There is not enough evidence to evaluate the response."

    if delegation.decision == "HUMAN_REVIEW":
        return (
            "The response contains assessable evidence, but the judgment requires "
            "human review before it should be considered final."
        )

    if diagnosis.error_type == "no_evident_error":
        return "The response satisfies the detected assessment criteria."

    if diagnosis.error_type == "incomplete_response":
        return "The response contains correct elements but remains incomplete."

    if diagnosis.error_type == "insufficient_evidence":
        return "The detected evidence is insufficient to satisfy the current criteria."

    return "Evalia generated an assessment judgment from the available evidence."


def generate_feedback(
    data: EvaluationInput,
    representation: RepresentationResult,
    judgment: AssessmentResult,
    diagnosis: DiagnosisResult,
    reliability: ReliabilityResult,
    delegation: DelegationResult,
) -> FeedbackResult:
    """
    Genera feedback estructurado a partir del diagnóstico.

    El feedback queda separado del motor de scoring para permitir
    distintos estilos, audiencias y modelos generativos en el futuro.
    """

    language = _feedback_language(data, representation)

    strengths = list(diagnosis.strengths)
    gaps = list(diagnosis.gaps)

    if language == "en":
        summary = _summary_en(diagnosis, delegation)

        next_step = (
            f"Review or develop the following aspect: {gaps[0]}."
            if gaps
            else "Continue applying the demonstrated understanding to new tasks."
        )

    else:
        summary = _summary_es(diagnosis, delegation)

        next_step = (
            f"Revisa o desarrolla el siguiente aspecto: {gaps[0]}."
            if gaps
            else "Continúa aplicando la comprensión demostrada en nuevas tareas."
        )

    if delegation.decision == "HUMAN_REVIEW":
        if language == "en":
            next_step = (
                "A human evaluator should review this response before the final "
                "assessment decision."
            )
        else:
            next_step = (
                "Un evaluador humano debería revisar esta respuesta antes de "
                "establecer la decisión final."
            )

    return FeedbackResult(
        summary=summary,
        strengths=strengths,
        needs_improvement=gaps,
        next_step=next_step,
        audience="learner",
        language=language,
        metadata={
            "mode": "baseline_structured_feedback",
            "delegation_decision": delegation.decision,
            "disagreement_risk": reliability.disagreement_risk,
            "score": judgment.score,
            "max_score": judgment.max_score,
        },
    )
