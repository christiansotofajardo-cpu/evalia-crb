from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

from .models import Criterion, EvaluationInput, RepresentationResult
from .semantic import semantic_compare


def _normalize_text(text: str) -> str:
    """
    Normalización liviana para perfiles y compatibilidad interna.
    La comparación semántica principal se delega a semantic.py.
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
    Agrupa todas las realizaciones aceptables de un mismo criterio.
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


def _semantic_threshold(
    data: EvaluationInput,
) -> float:
    """
    Obtiene el umbral semántico desde AssessmentSpec cuando existe.

    Permite adaptar Evalia a distintos dominios y tipos de tarea.
    """
    spec = data.task.assessment_spec

    if spec is None:
        return 0.75

    policy = spec.scoring_policy or {}

    try:
        threshold = float(
            policy.get("semantic_threshold", 0.75)
        )
    except (TypeError, ValueError):
        threshold = 0.75

    return max(0.0, min(1.0, threshold))


def _match_criterion(
    criterion: Criterion,
    response_text: str,
    threshold: float,
) -> Tuple[bool, List[Dict[str, object]]]:
    """
    Evalúa todas las realizaciones de un criterio usando
    el matcher semántico desacoplado.

    Devuelve evidencia estructurada y conserva el mejor match.
    """
    matches: List[Dict[str, object]] = []

    for variant in _criterion_variants(criterion):
        semantic_match = semantic_compare(
            response_text=response_text,
            reference_text=variant,
            threshold=threshold,
        )

        matches.append(
            {
                "variant": variant,
                "matched": semantic_match.matched,
                "similarity": semantic_match.similarity,
                "evidence": semantic_match.evidence,
                "method": semantic_match.method,
                "metadata": semantic_match.metadata,
            }
        )

    positive_matches = [
        item
        for item in matches
        if bool(item["matched"])
    ]

    positive_matches.sort(
        key=lambda item: float(item["similarity"]),
        reverse=True,
    )

    return bool(positive_matches), positive_matches


def represent(data: EvaluationInput) -> RepresentationResult:
    """
    Representación semántica de Evalia Core 2.0.

    Cada criterio constituye una unidad conceptual.
    Sus descripciones, variantes y valores aceptados funcionan
    como realizaciones alternativas del mismo significado.

    La detección concreta queda delegada al motor semántico,
    permitiendo sustituir el baseline por embeddings, LLM
    o motores híbridos sin modificar esta capa.
    """

    response_text = str(data.response.text or "").strip()

    spec = data.task.assessment_spec

    detected_concepts: List[str] = []
    missing_concepts: List[str] = []
    evidence_spans: List[Dict[str, object]] = []

    total_criteria = 0
    detected_criteria = 0

    threshold = _semantic_threshold(data)

    if spec is not None:
        total_criteria = len(spec.criteria)

        for criterion in spec.criteria:
            matched, matches = _match_criterion(
                criterion=criterion,
                response_text=response_text,
                threshold=threshold,
            )

            if matched:
                detected_criteria += 1
                detected_concepts.append(
                    criterion.description
                )

                best_match = matches[0]

                evidence_spans.append(
                    {
                        "criterion_id": criterion.id,
                        "concept": criterion.description,
                        "matched_variants": [
                            item["variant"]
                            for item in matches
                        ],
                        "best_variant": best_match["variant"],
                        "best_similarity": best_match["similarity"],
                        "evidence": best_match["evidence"],
                        "match_method": best_match["method"],
                        "semantic_threshold": threshold,
                        "matches": matches,
                    }
                )

            else:
                missing_concepts.append(
                    criterion.description
                )

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
        concepts_detected=list(
            dict.fromkeys(detected_concepts)
        ),
        concepts_missing=list(
            dict.fromkeys(missing_concepts)
        ),
        conceptual_relations=[],
        contradictions=[],
        conceptual_coverage=round(coverage, 3),
        response_profile=_response_profile(
            response_text
        ),
        evidence_spans=evidence_spans,
        metadata={
            "mode": "pluggable_semantic_representation",
            "criteria_total": total_criteria,
            "criteria_detected": detected_criteria,
            "coverage_unit": "criterion",
            "semantic_threshold": threshold,
        },
    )
