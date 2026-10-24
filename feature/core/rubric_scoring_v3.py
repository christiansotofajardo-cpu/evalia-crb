
from __future__ import annotations

"""Evalia 3.0: puntuación orientada por rúbrica (NO calificación certificada).

No utiliza similitud semántica como puntos ni equipara similitud con confianza.
Una decisión semántica no verificada implica revisión; no genera crédito
parcial automático salvo política explícita aprobada por el profesor.
"""

import math
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .models import (
    AssessmentResult, CriterionJudgment, DiagnosisResult,
    EvaluationInput, RepresentationResult,
)


def _normalize(text: Any) -> str:
    t = unicodedata.normalize('NFD', str(text or '').strip().casefold())
    return ' '.join(''.join(c for c in t if unicodedata.category(c) != 'Mn').split())


def _closed_answer(value: Any, task_type: str) -> str:
    t = _normalize(value)
    if task_type == 'true_false':
        if t in {'v', 'verdadero', 'true'}:
            return 'V'
        if t in {'f', 'falso', 'false'}:
            return 'F'
        return ''
    if task_type == 'multiple_choice':
        return t.upper() if len(t) == 1 and t.upper() in 'ABCDE' else ''
    return t


def _finite_nonnegative(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (ValueError, TypeError, OverflowError):
        return None
    return v if math.isfinite(v) and v >= 0.0 else None


def _adjudication_property(a: Any, field: str, default: Any = None) -> Any:
    if isinstance(a, Mapping):
        return a.get(field, default)
    return getattr(a, field, default)


def assess_v3(
    data: EvaluationInput,
    representation: RepresentationResult,
    adjudications: Optional[Mapping[str, Any]] = None,
) -> Tuple[AssessmentResult, DiagnosisResult]:
    """Entrega puntuación PROVISIONAL y alerta explícita si falta evidencia.

    `adjudications` debe contener dictámenes individuales por criterio,
    suministrados por engine.py; no un dictamen global concatenado.
    """
    spec = data.task.assessment_spec
    maximum = _finite_nonnegative(data.task.max_score)
    maximum = maximum if maximum is not None else 0.0
    route = (spec.scoring_policy or {}).get('scoring_route', 'semantic') if spec else 'semantic'
    is_v3 = bool(spec and (spec.scoring_policy or {}).get('rubric_version') == 'evalia-rubric-3.0-1')

    if not spec or not spec.criteria:
        result = AssessmentResult(score=0.0, max_score=maximum, metadata={
            'mode': 'rubric_v3', 'provisional': True,
            'requires_human_review': True, 'unscored': True,
        })
        diagnostic = DiagnosisResult(error_type='insufficient_specification', error_severity='high')
        return result, diagnostic

    weights = [_finite_nonnegative(c.weight) for c in spec.criteria]
    weight_total = sum(x or 0.0 for x in weights)
    if any(x is None for x in weights) or weight_total <= 0 or maximum <= 0:
        return (
            AssessmentResult(score=0.0, max_score=maximum, metadata={
                'mode': 'rubric_v3', 'provisional': True,
                'requires_human_review': True, 'unscored': True,
                'reason': 'invalid_point_configuration',
            }),
            DiagnosisResult(error_type='invalid_point_configuration', error_severity='high'),
        )

    # Para rúbricas v3, las ponderaciones son puntos acordados, NO pesos
    # que se normalizan sin consentimiento. Cualquier diferencia bloquea.
    if is_v3 and abs(weight_total - maximum) > 1e-6:
        return (
            AssessmentResult(score=0.0, max_score=maximum, metadata={
                'mode': 'rubric_v3', 'provisional': True,
                'requires_human_review': True, 'unscored': True,
                'reason': 'rubric_points_mismatch',
            }),
            DiagnosisResult(error_type='rubric_points_mismatch', error_severity='high'),
        )

    response = str(data.response.text or '').strip()
    judgments: List[CriterionJudgment] = []
    strengths: List[str] = []
    gaps: List[str] = []
    problems: List[str] = []
    total_score = 0.0
    adjudications = adjudications or {}
    evidence_map = {str(item.get('criterion_id')): item for item in representation.evidence_spans}

    # Para rúbricas antiguas, mantener proporción histórica, pero nunca
    # considerar un juicio heurístico como nota académica certificada.
    for criterion, weight in zip(spec.criteria, weights):
        point_max = float(weight if is_v3 else maximum * weight / weight_total)
        score = 0.0
        evidence: List[str] = []
        similarity = 0.0
        reason = 'human_review_required'
        status = 'unresolved'
        a = adjudications.get(criterion.id)

        if not response:
            status, reason = 'absent', 'empty_response'
        elif route == 'exact' and data.task.task_type in {'true_false', 'multiple_choice'}:
            allowed = criterion.accepted_values or [criterion.description]
            answer = _closed_answer(response, data.task.task_type)
            keys = {_closed_answer(x, data.task.task_type) for x in allowed}
            if answer and answer in keys:
                score, status, reason = point_max, 'correct', 'exact_key_match'
                evidence = [response]
            else:
                status, reason = 'incorrect', 'exact_key_mismatch'
        elif route == 'controlled' and data.task.task_type == 'fill_in':
            # Solo equivalencias expresamente aprobadas para completar.
            allowed = [_normalize(x) for x in (criterion.accepted_values or [criterion.description])]
            if _normalize(response) in allowed:
                score, status, reason = point_max, 'correct', 'controlled_match'
                evidence = [response]
            else:
                status, reason = 'unresolved', 'completion_needs_review'
        elif route in {'numeric_specialist_required', 'math_specialist_required'}:
            status, reason = 'unresolved', 'specialist_not_available'
        elif a is not None:
            relation = str(_adjudication_property(a, 'relation', 'undetermined'))
            raw_similarity = _adjudication_property(a, 'metadata', {})
            if isinstance(raw_similarity, Mapping):
                similarity = _finite_nonnegative(raw_similarity.get('semantic_similarity')) or 0.0
            contradiction = bool(_adjudication_property(a, 'contradiction', False))
            nli = ((representation.metadata.get('criterion_evidence') or [])
                   if isinstance(representation.metadata, dict) else [])
            nli_conflict = any(
                e.get('criterion_id') == criterion.id
                and (e.get('nli') or {}).get('relation') == 'contradiction'
                for e in nli if isinstance(e, dict)
            )
            if contradiction or nli_conflict or relation in {'contradictory', 'misconception'}:
                status, reason = 'unresolved', 'conceptual_conflict_needs_review'
            elif relation == 'correct':
                score, status, reason = point_max, 'correct', 'conceptual_judgment_correct'
                evidence = [str(_adjudication_property(a, 'evidence', response))]
            elif relation == 'partially_correct':
                # La categoría "parcial" no autoriza regalar el 50%.
                # Debe existir una regla aprobada para otorgar puntos parciales.
                fraction = criterion.metadata.get('partial_credit_fraction')
                fraction = _finite_nonnegative(fraction)
                if fraction is not None and 0.0 < fraction < 1.0 and is_v3:
                    score, status, reason = point_max * fraction, 'partial', 'teacher_authorized_partial_credit'
                    evidence = [str(_adjudication_property(a, 'evidence', response))]
                else:
                    status, reason = 'unresolved', 'partial_credit_not_specified'
            elif relation == 'irrelevant':
                status, reason = 'incorrect', 'evidence_irrelevant'
            else:
                status, reason = 'unresolved', 'insufficient_conceptual_evidence'
        else:
            # Cuando engine no entregue dictamen por criterio, no inferir
            # puntos solo por similitud.
            status, reason = 'unresolved', 'criterion_adjudication_missing'

        if status in {'correct', 'partial'}:
            strengths.append(criterion.description)
        else:
            gaps.append(criterion.description)
        if status == 'unresolved':
            problems.append(criterion.id)
        total_score += score
        judgments.append(CriterionJudgment(
            criterion_id=criterion.id, satisfied=(status == 'correct'),
            score=round(score, 4), max_score=round(point_max, 4),
            confidence=round(similarity, 4), evidence=evidence,
            reason=reason,
            metadata={
                'rubric_status': status, 'criterion_points': point_max,
                'similarity_not_calibrated': True,
                'required': criterion.required,
            },
        ))

    if not response:
        error, severity = 'empty_response', 'high'
    elif problems:
        error, severity = 'unresolved_criteria', 'high'
    elif any(j.reason == 'exact_key_mismatch' for j in judgments):
        error, severity = 'incorrect_closed_answer', 'medium'
    elif not gaps:
        error, severity = 'no_evident_error', 'low'
    elif strengths:
        error, severity = 'incomplete_response', 'medium'
    else:
        error, severity = 'insufficient_evidence', 'high'

    result = AssessmentResult(
        score=round(min(total_score, maximum), 4), max_score=maximum,
        criterion_judgments=judgments,
        metadata={
            'mode': 'rubric_v3_criterion_scoring', 'provisional': True,
            'requires_human_review': bool(problems) or not is_v3,
            'unscored_criteria': problems,
            'scoring_route': route, 'teacher_approved': bool(is_v3),
        },
    )
    diagnosis = DiagnosisResult(
        strengths=strengths, gaps=gaps,
        misconceptions=list(representation.contradictions),
        error_type=error, error_severity=severity,
        metadata={'unresolved_criteria': problems, 'mode': 'rubric_v3_criterion_scoring'},
    )
    return result, diagnosis
