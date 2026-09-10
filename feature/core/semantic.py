from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Protocol
import re
import unicodedata


# ============================================================
# Normalization utilities
# ============================================================

def normalize_semantic_text(text: str) -> str:
    """Normalize text for transparent lexical comparison."""
    text = text or ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokens(text: str) -> set[str]:
    normalized = normalize_semantic_text(text)
    return set(re.findall(r"\b\w+\b", normalized))


def _token_overlap(a: str, b: str) -> float:
    tokens_a = _tokens(a)
    tokens_b = _tokens(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union if union else 0.0


def _sequence_similarity(a: str, b: str) -> float:
    a_norm = normalize_semantic_text(a)
    b_norm = normalize_semantic_text(b)

    if not a_norm or not b_norm:
        return 0.0

    return SequenceMatcher(None, a_norm, b_norm).ratio()


# ============================================================
# Semantic result contract
# ============================================================

@dataclass
class SemanticMatch:
    similarity: float
    matched: bool
    method: str
    evidence: str
    metadata: Dict[str, Any]


# ============================================================
# Semantic matcher interface
# ============================================================

class SemanticMatcher(Protocol):
    def compare(
        self,
        response_text: str,
        reference_text: str,
        threshold: float = 0.75,
    ) -> SemanticMatch:
        ...


# ============================================================
# Transparent lexical baseline
# ============================================================

class BaselineSemanticMatcher:
    """
    Transparent lexical baseline.

    This matcher is intentionally NOT a deep semantic model.
    It combines explicit substring matching, token overlap,
    and character-sequence similarity.
    """

    def compare(
        self,
        response_text: str,
        reference_text: str,
        threshold: float = 0.75,
    ) -> SemanticMatch:

        response_norm = normalize_semantic_text(response_text)
        reference_norm = normalize_semantic_text(reference_text)

        if not response_norm or not reference_norm:
            return SemanticMatch(
                similarity=0.0,
                matched=False,
                method="empty_input",
                evidence="",
                metadata={
                    "semantic_model": False,
                    "threshold": threshold,
                },
            )

        # High-precision explicit evidence
        if reference_norm in response_norm:
            return SemanticMatch(
                similarity=1.0,
                matched=True,
                method="explicit_reference_match",
                evidence=reference_text,
                metadata={
                    "semantic_model": False,
                    "threshold": threshold,
                },
            )

        token_score = _token_overlap(response_text, reference_text)
        sequence_score = _sequence_similarity(response_text, reference_text)

        similarity = (0.65 * token_score) + (0.35 * sequence_score)
        similarity = max(0.0, min(1.0, similarity))

        return SemanticMatch(
            similarity=similarity,
            matched=similarity >= threshold,
            method="baseline_lexical_similarity",
            evidence=reference_text if similarity >= threshold else "",
            metadata={
                "semantic_model": False,
                "threshold": threshold,
                "token_overlap": token_score,
                "sequence_similarity": sequence_score,
            },
        )


# ============================================================
# Multilingual embedding matcher
# ============================================================

class EmbeddingSemanticMatcher:
    """
    Real semantic matcher based on multilingual sentence embeddings.

    The model is loaded lazily so Evalia can still operate with
    the lexical baseline if sentence-transformers is unavailable.

    Default model:
        sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    Supports multilingual semantic comparison, including Spanish
    and English.
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "EmbeddingSemanticMatcher requires the "
                "'sentence-transformers' package."
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def compare(
        self,
        response_text: str,
        reference_text: str,
        threshold: float = 0.60,
    ) -> SemanticMatch:

        response_text = (response_text or "").strip()
        reference_text = (reference_text or "").strip()

        if not response_text or not reference_text:
            return SemanticMatch(
                similarity=0.0,
                matched=False,
                method="empty_input",
                evidence="",
                metadata={
                    "semantic_model": True,
                    "model": self.model_name,
                    "threshold": threshold,
                },
            )

        # Preserve exact evidence when available.
        response_norm = normalize_semantic_text(response_text)
        reference_norm = normalize_semantic_text(reference_text)

        if reference_norm in response_norm:
            return SemanticMatch(
                similarity=1.0,
                matched=True,
                method="explicit_reference_match",
                evidence=reference_text,
                metadata={
                    "semantic_model": True,
                    "model": self.model_name,
                    "threshold": threshold,
                },
            )

        model = self._load_model()

        embeddings = model.encode(
            [response_text, reference_text],
            normalize_embeddings=True,
        )

        # With normalized embeddings, dot product = cosine similarity.
        similarity = float(embeddings[0] @ embeddings[1])

        # Cosine similarity can theoretically be negative.
        similarity = max(0.0, min(1.0, similarity))

        return SemanticMatch(
            similarity=similarity,
            matched=similarity >= threshold,
            method="multilingual_embedding_similarity",
            evidence=reference_text if similarity >= threshold else "",
            metadata={
                "semantic_model": True,
                "model": self.model_name,
                "threshold": threshold,
                "cosine_similarity": similarity,
            },
        )


# ============================================================
# Default engine and public entry point
# ============================================================

DEFAULT_SEMANTIC_MATCHER: SemanticMatcher = BaselineSemanticMatcher()


def semantic_compare(
    response_text: str,
    reference_text: str,
    threshold: float = 0.75,
    matcher: Optional[SemanticMatcher] = None,
) -> SemanticMatch:
    """
    Public semantic comparison interface.

    The rest of Evalia should call this function rather than
    depending directly on a specific semantic engine.
    """

    engine = matcher or DEFAULT_SEMANTIC_MATCHER

    return engine.compare(
        response_text=response_text,
        reference_text=reference_text,
        threshold=threshold,
    )
