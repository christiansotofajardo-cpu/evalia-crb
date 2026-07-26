"""
capture_assistant.py
====================

Orquestador principal del flujo Smart Capture.

Responsabilidades:

1. Analizar calidad de la fotografía.
2. Detectar páginas.
3. Organizar páginas por estudiante.
4. Generar vista previa.
5. Entregar un resultado único para el OCR.

No contiene algoritmos de visión.
Toda la lógica específica vive en los módulos correspondientes.
"""

from typing import List

from .models import CaptureResult
from .quality_analyzer import QualityAnalyzer
from .page_detector import PageDetector
from .page_organizer import PageOrganizer
from .preview import PreviewGenerator


class CaptureAssistant:

    def __init__(self):

        self.quality = QualityAnalyzer()
        self.detector = PageDetector()
        self.organizer = PageOrganizer()
        self.preview = PreviewGenerator()

    def process(self, image):

        """
        Ejecuta el flujo completo Smart Capture.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        CaptureResult
        """

        ############################################
        # 1. Calidad
        ############################################

        quality_report = self.quality.analyze(image)

        ############################################
        # 2. Detectar hojas
        ############################################

        detected_pages = self.detector.detect(image)

        ############################################
        # 3. Organizar por estudiante
        ############################################

        student_groups = self.organizer.group(detected_pages)

        ############################################
        # 4. Vista previa
        ############################################

        preview_image = self.preview.render(
            image=image,
            groups=student_groups
        )

        ############################################
        # 5. Resultado
        ############################################

        return CaptureResult(

            success=True,

            quality=quality_report,

            pages_detected=len(detected_pages),

            students_detected=len(student_groups),

            pages=detected_pages,

            students=student_groups,

            preview=preview_image
        )
