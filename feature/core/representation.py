from __future__ import annotations

import re
import unicodedata
from typing import List

from .models import EvaluationInput, RepresentationResult


def _normalize_text(text: str) -> str:
    """
    Normalización liviana para el baseline explicable.
    No reemplaza el futuro motor semántico multilingüe.
    """
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)
    return text


def _response_profile(text: str) -> str:
    """
    Perfil muy simple de longitud.
    Servirá como baseline y luego podrá enriquecerse.
    """
    words = re.findall(r"\b\w+\b", str(text or ""))

    if not words:
        return "empty"
    if len(words) <= 5:
        return "very_brief"
    if len(words) <= 15:
        return "brief"
    if len(words) <= 40:
        return "developed"
    return "extended"


def _criterion_terms(data: EvaluationInput) -> List[str]:
    """
    Obtiene conceptos esperados desde el AssessmentSpec.
    """
    spec = data.task.assessment_spec

    if spec is None:
        return []

    terms: List[str] = []

    for criterion in spec.criteria:
        if criterion.description:
            terms.append(criterion.description)

        terms.extend(criterion.semantic_variants)
        terms.extend(criterion.accepted_values)

    return [
        term.strip()
        for term in terms
        if str(term).strip()
    ]


def represent(data: EvaluationInput) -> RepresentationResult:
    """
    Primera capa de representación de Evalia Core 2.0.

    Convierte una respuesta en evidencia estructurada básica.
    Este baseline será reemplazable por motores más avanzados.
    """
    response_text = str(data.response.text or "").strip()
    normalized_answer = _normalize_text(response_text)

    expected_terms = _criterion_terms(data)

    detected: List[str] = []
    missing: List[str] = []

    for term in expected_terms:
        normalized_term = _normalize_text(term)

        if normalized_term and normalized_term in normalized_answer:
            detected.append(term)
        else:
            missing.append(term)

    unique_expected = list(dict.fromkeys(expected_terms))
    unique_detected = list(dict.fromkeys(detected))
    unique_missing = list(dict.fromkeys(missing))

    coverage = (
        len(unique_detected) / len(unique_expected)
        if unique_expected
        else 0.0
    )

    language = (
        data.task.language
        if data.task.language != "auto"
        else data.language
    )

    if language == "auto":
        language = "unknown"

    return RepresentationResult(
        language=language,
        concepts_detected=unique_detected,
        concepts_missing=unique_missing,
        conceptual_relations=[],
        contradictions=[],
        conceptual_coverage=round(coverage, 3),
        response_profile=_response_profile(response_text),
        evidence_spans=[],
        metadata={
            "mode": "baseline_rule_based",
            "expected_terms_count": len(unique_expected),
            "detected_terms_count": len(unique_detected),
        },
    )
