"""
Evalia Smart Capture.
"""

from .capture_assistant import CaptureAssistant
from .models import (
    CaptureResult,
    DetectedPage,
    QualityReport,
    StudentPageGroup,
)
from .page_detector import PageDetector
from .page_organizer import PageOrganizer
from .preview import PreviewGenerator
from .quality_analyzer import QualityAnalyzer

__all__ = [
    "CaptureAssistant",
    "CaptureResult",
    "DetectedPage",
    "QualityReport",
    "StudentPageGroup",
    "PageDetector",
    "PageOrganizer",
    "PreviewGenerator",
    "QualityAnalyzer",
]
