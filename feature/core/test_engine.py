from __future__ import annotations

from .delegation import decide_delegation
from .reliability import _threshold_margin_risk
from .engine import evaluate
from .feedback import generate_feedback
from .nli import infer_conceptual_relation
from .refinement import is_borderline, refine_borderline_evidence
from .models import (
    AssessmentSpec,
    AssessmentResult,
    DiagnosisResult,
    DelegationResult,
    ReliabilityResult,
    Criterion,
    EvaluationInput,
    Learner,
    ResponseInput,
    RepresentationResult,
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

    assert result.adjudication
    assert "relation" in result.adjudication
    assert "confidence" in result.adjudication
    assert "criterion_evidence" in result.representation.metadata
    assert result.representation.metadata["criterion_evidence"]
    assert {"matched", "best_similarity", "threshold_margin", "matches", "nli"} <= result.representation.metadata["criterion_evidence"][0].keys()

    assert result.traceability["assessment_id"] == "demo_assessment"
    assert result.traceability["task_id"] == "task_1"
    assert result.traceability["core_version"] == "2.0-baseline"


def test_unmatched_criterion_preserves_evidence():
    spec = AssessmentSpec(id="unmatched_spec", language="es", criteria=[Criterion(id="missing", description="fotosíntesis", required=True)])
    task = Task(id="unmatched_task", prompt="Explica la fotosíntesis.", assessment_spec=spec)
    response = ResponseInput(text="La memoria de trabajo mantiene información temporalmente.", source="text", source_confidence=1.0)
    data = EvaluationInput(assessment_id="unmatched_assessment", task=task, response=response, language="es", domain="education")
    result = evaluate(data)
    assert result.representation.metadata["criterion_evidence"][0]["matched"] is False
    assert result.representation.metadata["criterion_evidence"][0]["matches"]
def test_conceptual_nli_relations():
    f = infer_conceptual_relation
    assert f("La fotosíntesis transforma energía luminosa en energía química.", "La fotosíntesis transforma energía luminosa en energía química.", 1.0, 0.75).relation == "entailment"
    assert f("La memoria de trabajo mantiene información temporalmente.", "La fotosíntesis transforma energía luminosa en energía química.", 0.20, 0.75).relation == "neutral"
    assert f("La fotosíntesis consume oxígeno.", "La fotosíntesis produce oxígeno.", 0.70, 0.75, ["consume oxígeno"]).relation == "contradiction"


def test_threshold_margin_uncertainty():
    near = RepresentationResult(metadata={"criterion_evidence": [{"threshold_margin": 0.01}]})
    far = RepresentationResult(metadata={"criterion_evidence": [{"threshold_margin": 0.50}]})
    assert _threshold_margin_risk(near) > _threshold_margin_risk(far)


def test_configurable_delegation_policy():
    spec = AssessmentSpec(reliability_policy={"caution_threshold": 0.10, "human_review_threshold": 0.25})
    data = EvaluationInput(assessment_id="test_assessment", task=Task(id="test_task", prompt="Test", assessment_spec=spec), response=ResponseInput(text="Respuesta suficiente."))
    result = decide_delegation(data, RepresentationResult(response_profile="substantive"), AssessmentResult(), DiagnosisResult(), ReliabilityResult(confidence=0.70, disagreement_risk=0.30, reliability_class="medium"))
    assert result.decision == "HUMAN_REVIEW"


def test_pedagogical_scaffolding():
    data = EvaluationInput(assessment_id="test_feedback", task=Task(id="task_feedback", prompt="Explica el proceso."), response=ResponseInput(text="Respuesta parcial."), language="es")
    result = generate_feedback(data, RepresentationResult(language="es"), AssessmentResult(), DiagnosisResult(gaps=["la relación entre causa y efecto"]), ReliabilityResult(), DelegationResult(decision="AUTO_ACCEPT"))
    assert result.hint
    assert result.scaffold
    assert result.explanation
    assert "causa y efecto" in result.hint
    review = generate_feedback(data, RepresentationResult(language="es"), AssessmentResult(), DiagnosisResult(gaps=["la relación entre causa y efecto"]), ReliabilityResult(), DelegationResult(decision="HUMAN_REVIEW"))
    assert review.hint == review.scaffold == review.explanation == ""


def test_refinement_selection():
    assert is_borderline(0.05)
    assert is_borderline(-0.10)
    assert not is_borderline(0.30)
    result = refine_borderline_evidence("respuesta", "criterio", 0.30, 0.75)
    assert result.activated is False
    assert result.refined_similarity == result.original_similarity
    borderline = refine_borderline_evidence("La fotosíntesis transforma energía luminosa en energía química.", "La fotosíntesis convierte energía luminosa en energía química.", 0.70, 0.75)
    assert borderline.activated is True
    assert borderline.reason == "borderline_semantic_evidence"
    assert 0.0 <= borderline.refined_similarity <= 1.0
    assert borderline.metadata["method"]


if __name__ == "__main__":
    test_evalia_core_pipeline()
    test_unmatched_criterion_preserves_evidence()
    test_conceptual_nli_relations()
    test_threshold_margin_uncertainty()
    test_configurable_delegation_policy()
    test_pedagogical_scaffolding()
    test_refinement_selection()
    print("✅ Evalia Core 2.0 pipeline test passed.")
