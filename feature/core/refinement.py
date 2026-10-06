from __future__ import annotations


from dataclasses import dataclass, field
from .semantic import EmbeddingSemanticMatcher
from typing import Any, Dict

@dataclass
class RefinementResult:
    activated: bool = False
    original_similarity: float = 0.0
    refined_similarity: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

def is_borderline(threshold_margin: float, refinement_band: float = 0.15) -> bool:
    """Return True when semantic evidence is close enough to the decision threshold to merit refinement."""
    return abs(float(threshold_margin)) <= float(refinement_band)
def refine_borderline_evidence(
    response_text: str,
    reference_text: str,
    original_similarity: float,
    semantic_threshold: float,
    refinement_band: float = 0.15,
) -> RefinementResult:
    margin = float(original_similarity) - float(semantic_threshold)

    if not is_borderline(margin, refinement_band):
        return RefinementResult(
            activated=False,
            original_similarity=float(original_similarity),
            refined_similarity=float(original_similarity),
            reason="outside_refinement_band",
        )

    match = EmbeddingSemanticMatcher().compare(
        response_text,
        reference_text,
        threshold=semantic_threshold,
    )

    return RefinementResult(
        activated=True,
        original_similarity=float(original_similarity),
        refined_similarity=float(match.similarity),
        reason="borderline_semantic_evidence",
        metadata={
            "method": match.method,
            "matched": match.matched,
            "refinement_band": refinement_band,
        },
    )
