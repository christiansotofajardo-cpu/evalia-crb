"""
Modelos de datos compartidos por Evalia Smart Capture.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectedPage:
    """
    Hoja identificada dentro de una fotografía general.
    """

    page_id: int
    image: Any
    corners: list[tuple[float, float]]
    confidence: float

    x_center: float
    y_center: float

    width: float
    height: float

    sharpness_score: float = 0.0
    brightness_score: float = 0.0
    resolution_score: float = 0.0

    warnings: list[str] = field(default_factory=list)


@dataclass
class StudentPageGroup:
    """
    Conjunto de páginas que corresponden a un estudiante.
    """

    student_index: int
    pages: list[DetectedPage] = field(default_factory=list)

    complete: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """
    Informe de calidad de la fotografía o de una página recortada.
    """

    acceptable: bool
    score: float

    sharpness_score: float
    brightness_score: float
    resolution_score: float

    warnings: list[str] = field(default_factory=list)


@dataclass
class CaptureResult:
    """
    Resultado completo del procesamiento inicial.
    """

    capture_quality: QualityReport
    detected_pages: list[DetectedPage] = field(default_factory=list)
    student_groups: list[StudentPageGroup] = field(default_factory=list)

    detected_page_count: int = 0
    inferred_student_count: int = 0

    warnings: list[str] = field(default_factory=list)
