from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Protocol


def normalize_semantic_text(text: str) -> str:
    """
    Normalización lingüística básica y agnóstica al dominio.
    """
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> List[str]:
    """
    Tokenización mínima para el baseline.
    """
    return [
        token
        for token in normalize_semantic_text(text).split()
        if token
    ]


def _token_overlap(text_a: str, text_b: str) -> float:
    """
    Similaridad por cobertura léxica.
    No pretende representar significado profundo.
    """
    tokens_a = set(_tokens(text_a))
    tokens_b = set(_tokens(text_b))

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)

    return len(intersection) / min(
        len(tokens_a),
        len(tokens_b),
    )


def _sequence_similarity(text_a: str, text_b: str) -> float:
    """
    Similaridad superficial entre secuencias normalizadas.
    """
    normalized_a = normalize_semantic_text(text_a)
    normalized_b = normalize_semantic_text(text_b)

    if not normalized_a or not normalized_b:
        return 0.0

    return SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    ).ratio()


@dataclass
class SemanticMatch:
    """
    Contrato común para cualquier motor semántico de Evalia.
    """

    matched: bool
    similarity: float
    evidence: str = ""
    reference: str = ""
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticMatcher(Protocol):
    """
    Interfaz que deberá respetar cualquier motor semántico.

    Ejemplos futuros:
    - embeddings;
    - modelos multilingües;
    - LLM;
    - motores híbridos;
    - modelos locales.
    """

    def compare(
        self,
        response_text: str,
        reference_text: str,
        threshold: float = 0.75,
    ) -> SemanticMatch:
        ...


class BaselineSemanticMatcher:
    """
    Motor inicial, transparente y sin dependencias externas.

    Importante:
    este matcher todavía NO constituye comprensión semántica profunda.
    Su función es establecer el contrato arquitectónico que luego
    podrá ser implementado por motores mucho más potentes.
    """

    name = "baseline_lexical_similarity"

    def compare(
        self,
        response_text: str,
        reference_text: str,
        threshold: float = 0.75,
    ) -> SemanticMatch:

        normalized_response = normalize_semantic_text(response_text)
        normalized_reference = normalize_semantic_text(reference_text)

        if not normalized_response or not normalized_reference:
            return SemanticMatch(
                matched=False,
                similarity=0.0,
                evidence="",
                reference=reference_text,
                method=self.name,
                metadata={
                    "threshold": threshold,
                    "empty_input": True,
                },
            )

        # Coincidencia explícita: evidencia fuerte y totalmente explicable.
        if normalized_reference in normalized_response:
            return SemanticMatch(
                matched=True,
                similarity=1.0,
                evidence=reference_text,
                reference=reference_text,
                method="explicit_reference_match",
                metadata={
                    "threshold": threshold,
                },
            )

        token_score = _token_overlap(
            normalized_response,
            normalized_reference,
        )

        sequence_score = _sequence_similarity(
            normalized_response,
            normalized_reference,
        )

        # Baseline híbrido superficial.
        similarity = (
            0.65 * token_score
            + 0.35 * sequence_score
        )

        similarity = max(
            0.0,
            min(1.0, similarity),
        )

        return SemanticMatch(
            matched=similarity >= threshold,
            similarity=round(similarity, 3),
            evidence=response_text if similarity >= threshold else "",
            reference=reference_text,
            method=self.name,
            metadata={
                "threshold": threshold,
                "token_overlap": round(token_score, 3),
                "sequence_similarity": round(sequence_score, 3),
                "semantic_model": False,
            },
        )


DEFAULT_SEMANTIC_MATCHER = BaselineSemanticMatcher()


def semantic_compare(
    response_text: str,
    reference_text: str,
    threshold: float = 0.75,
    matcher: SemanticMatcher = DEFAULT_SEMANTIC_MATCHER,
) -> SemanticMatch:
    """
    Punto de entrada común para comparación semántica.

    El resto de Evalia podrá utilizar esta función sin saber
    qué modelo concreto existe por debajo.
    """
    return matcher.compare(
        response_text=response_text,
        reference_text=reference_text,
        threshold=threshold,
    )
