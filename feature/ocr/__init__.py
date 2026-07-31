"""
Servicios OCR reutilizables de Evalia.

Este paquete centraliza el reconocimiento óptico de caracteres para que pueda
ser utilizado tanto por el OCR tradicional como por Smart Capture.
"""

from .service import (
    OCR_STATUS,
    get_ocr_status,
    preprocess_image_for_ocr,
    run_easyocr_on_image,
    run_mistral_ocr_on_file,
    run_ocr_on_image,
    run_tesseract_ocr_on_image,
)

__all__ = [
    "OCR_STATUS",
    "get_ocr_status",
    "preprocess_image_for_ocr",
    "run_easyocr_on_image",
    "run_mistral_ocr_on_file",
    "run_ocr_on_image",
    "run_tesseract_ocr_on_image",
]
