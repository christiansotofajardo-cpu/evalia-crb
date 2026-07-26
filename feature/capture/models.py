from dataclasses import dataclass
from typing import Any

@dataclass
class DetectedPage:
    page_id: int
    image: Any
    corners: list
    confidence: float
