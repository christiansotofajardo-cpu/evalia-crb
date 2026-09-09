from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

from .models import Criterion, EvaluationInput, RepresentationResult


def _normalize_text(text: str) -> str:
    """
    Normalización liviana para el baseline semántico.
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
    Perfil básico de extensión de respuesta.
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


def _criterion_variants(criterion: Criterion) -> List[str]:
    """
    Agrupa todas las formas válidas de evidenciar un mismo criterio.

    Importante:
    una descripción y sus variantes semánticas representan
    el mismo concepto evaluativo, no conceptos independientes.
    """
    variants: List[str] = []

    if criterion.description:
        variants.append(criterion.description)

    variants.extend(criterion.semantic_variants)
    variants.extend(criterion.accepted_values)

    return list(
        dict.fromkeys(
            value.strip()
            for value in variants
            if str(value).strip()
        )
    )


def _match_criterion(
    criterion: Criterion,
    normalized_answer: str,
) -> Tuple[bool, List[str]]:
    """
    Busca evidencia para un criterio usando cualquiera
    de sus realizaciones semánticas aceptadas.
    """
    matched_variants: List[str] = []

    for variant in _criterion_variants(criterion):
        normalized_variant = _normalize_text(variant)

        if normalized_variant and normalized_variant in normalized_answer:
            matched_variants.append(variant)

    return bool(matched_variants), matched_variants


def represent(data: EvaluationInput) -> RepresentationResult:
    """
    Capa de representación semántica de Evalia Core 2.0.

    La unidad de análisis es ahora el criterio conceptual,
    no cada variante lingüística por separado.
    """

    response_text = str(data.response.text or "").strip()
    normalized_answer = _normalize_text(response_text)

    spec = data.task.assessment_spec

    detected_concepts: List[str] = []
    missing_concepts: List[str] = []
    evidence_spans: List[Dict[str, object]] = []

    total_criteria = 0
    detected_criteria = 0

    if spec is not None:
        total_criteria = len(spec.criteria)

        for criterion in spec.criteria:
            matched, matched_variants = _match_criterion(
                criterion,
                normalized_answer,
            )

            if matched:
                detected_criteria += 1
                detected_concepts.append(criterion.description)

                evidence_spans.append(
                    {
                        "criterion_id": criterion.id,
                        "concept": criterion.description,
                        "matched_variants": matched_variants,
                        "match_type": "semantic_variant_match",
                    }
                )

            else:
                missing_concepts.append(criterion.description)

    coverage = (
        detected_criteria / total_criteria
        if total_criteria
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
        concepts_detected=list(dict.fromkeys(detected_concepts)),
        concepts_missing=list(dict.fromkeys(missing_concepts)),
        conceptual_relations=[],
        contradictions=[],
        conceptual_coverage=round(coverage, 3),
        response_profile=_response_profile(response_text),
        evidence_spans=evidence_spans,
        metadata={
            "mode": "criterion_centered_semantic_baseline",
            "criteria_total": total_criteria,
            "criteria_detected": detected_criteria,
            "coverage_unit": "criterion",
        },
    )
