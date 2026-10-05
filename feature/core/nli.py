from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from .adjudication import _declared_incompatible_concepts, _generic_conceptual_conflicts, _negation_mismatch

@dataclass
class ConceptualNLIResult:
    relation: str
    confidence: float
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

ENTAILMENT = "entailment"
NEUTRAL = "neutral"
CONTRADICTION = "contradiction"

def infer_conceptual_relation(
    response_text: str,
    reference_text: str,
    best_similarity: float,
    semantic_threshold: float,
    contradictory_values: List[str] | None = None,
) -> ConceptualNLIResult:
    """Infer an NLI relation for one conceptual criterion."""
    incompatible = _declared_incompatible_concepts(response_text, {"incompatible_concepts": contradictory_values or []})
    generic_conflicts = _generic_conceptual_conflicts(response_text, reference_text)
    if incompatible:
        return ConceptualNLIResult(relation=CONTRADICTION, confidence=0.90, evidence=", ".join(incompatible), metadata={"source": "declared_incompatible_concept", "detected": incompatible})
    if generic_conflicts and best_similarity >= 0.45:
        return ConceptualNLIResult(relation=CONTRADICTION, confidence=min(1.0, 0.62 + 0.25 * best_similarity), evidence=response_text, metadata={"source": "generic_conceptual_conflict", "conflicts": generic_conflicts})
    if _negation_mismatch(response_text, reference_text) and best_similarity >= 0.45:
        return ConceptualNLIResult(relation=CONTRADICTION, confidence=min(1.0, 0.70 + 0.30 * best_similarity), evidence=response_text, metadata={"source": "polarity_conflict"})
    if best_similarity >= semantic_threshold:
        return ConceptualNLIResult(
            relation=ENTAILMENT,
            confidence=min(1.0, best_similarity),
            evidence=response_text,
            metadata={"source": "semantic_evidence"},
        )
    return ConceptualNLIResult(
        relation=NEUTRAL,
        confidence=min(1.0, 1.0 - abs(best_similarity - semantic_threshold)),
        evidence=response_text,
        metadata={"source": "insufficient_semantic_evidence"},
    )
