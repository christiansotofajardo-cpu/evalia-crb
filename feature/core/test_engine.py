from __future__ import annotations

from .engine import evaluate
from .models import (
    AssessmentSpec,
    Criterion,
    EvaluationInput,
    Learner,
    ResponseInput,
    Task,
)


def test_evalia_core_pipeline():
    """
    Prueba mínima de integración de Evalia Core 2.0.

    Verifica que una respuesta pueda recorrer el pipeline completo:
    representación -> evaluación -> diagnóstico -> confiabilidad ->
    delegación -> feedback -> trazabilidad.
    """

    spec = AssessmentSpec(
        id="demo_spec",
        name="Demo assessment specification",
        language="es",
        domain="education",
        criteria=[
            Criterion(
                id="criterion_1",
                description="memoria de trabajo",
                weight=1.0,
                required=True,
                semantic_variants=[
                    "memoria operativa",
                    "working memory",
                ],
            ),
            Criterion(
                id="criterion_2",
                description="comprensión",
                weight=1.0,
                required=True,
                semantic_variants=[
                    "comprension",
                    "understanding",
                ],
            ),
        ],
    )

    task = Task(
        id="task_1",
        prompt=(
            "Explica brevemente la relación entre memoria de trabajo "
            "y comprensión."
        ),
        task_type="causal_explanation",
        max_score=2.0,
        language="es",
        domain="education",
        assessment_spec=spec,
    )

    response = ResponseInput(
        text=(
            "La memoria de trabajo permite mantener y procesar información "
            "mientras se construye la comprensión."
        ),
        source="text",
        source_confidence=1.0,
    )

    data = EvaluationInput(
        assessment_id="demo_assessment",
        task=task,
        response=response,
        learner=Learner(
            id="learner_1",
            name="Demo Learner",
        ),
        language="es",
        domain="education",
    )

    result = evaluate(data)

    assert result.assessment_id == "demo_assessment"
    assert result.task_id == "task_1"
    assert result.learner_id == "learner_1"

    assert result.language == "es"

    assert result.representation.conceptual_coverage > 0.0

    assert result.judgment.max_score == 2.0
    assert result.judgment.score >= 0.0

    assert 0.0 <= result.reliability.confidence <= 1.0
    assert 0.0 <= result.reliability.disagreement_risk <= 1.0

    assert result.delegation.decision in {
        "AUTO_ACCEPT",
        "ACCEPT_WITH_CAUTION",
        "HUMAN_REVIEW",
        "INSUFFICIENT_EVIDENCE",
    }

    assert result.feedback.summary
    assert result.feedback.language == "es"

    assert result.traceability["assessment_id"] == "demo_assessment"
    assert result.traceability["task_id"] == "task_1"
    assert result.traceability["core_version"] == "2.0-baseline"
