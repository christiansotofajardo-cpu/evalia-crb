from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from .assessment import assess
from .delegation import decide_delegation
from .feedback import generate_feedback
from .models import EvaliaResult, EvaluationInput
from .reliability import estimate_reliability
from .representation import represent


def _build_traceability(
    data: EvaluationInput,
    representation: Any,
    judgment: Any,
    diagnosis: Any,
    reliability: Any,
    delegation: Any,
) -> Dict[str, Any]:
    """
    Construye un registro mínimo y auditable del proceso evaluativo.

    Más adelante esta capa podrá ampliarse con:
    - versiones de modelos;
    - proveedores OCR;
    - reglas aplicadas;
    - evidencia semántica;
    - tiempos de ejecución;
    - intervención humana.
    """

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
        "core_version": "2.0-baseline",
    }


def evaluate(data: EvaluationInput) -> EvaliaResult:
    """
    Punto de entrada principal de Evalia Core 2.0.

    Pipeline:
        INPUT
          ↓
        REPRESENT
          ↓
        ASSESS
          ↓
        DIAGNOSE
          ↓
        ESTIMATE RELIABILITY
          ↓
        DECIDE DELEGATION
          ↓
        GENERATE FEEDBACK
          ↓
        TRACE
    """

    representation = represent(data)

    judgment, diagnosis = assess(
        data,
        representation,
    )

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

    feedback = generate_feedback(
        data,
        representation,
        judgment,
        diagnosis,
        reliability,
        delegation,
    )

    traceability = _build_traceability(
        data,
        representation,
        judgment,
        diagnosis,
        reliability,
        delegation,
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
        traceability=traceability,
    )
