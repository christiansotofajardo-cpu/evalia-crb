"""
Detector de páginas para Evalia Smart Capture.

Este módulo detecta automáticamente las hojas presentes
en una fotografía y devuelve sus coordenadas.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectedPage:
    page_id: int
    image: Any
    corners: list
    confidence: float


def detect_pages(image: Any) -> list[DetectedPage]:
    """
    Detecta todas las páginas presentes en una fotografía.

    Parameters
    ----------
    image
        Imagen capturada desde el teléfono.

    Returns
    -------
    list[DetectedPage]
        Lista de páginas detectadas.
    """

    raise NotImplementedError
