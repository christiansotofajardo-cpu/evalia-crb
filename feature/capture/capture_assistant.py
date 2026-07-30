"""
capture_assistant.py
====================

Orquestador principal de Evalia Smart Capture.

Este módulo conecta las distintas etapas del procesamiento:

    fotografía
        ↓
    análisis de calidad
        ↓
    detección de páginas
        ↓
    organización por estudiante
        ↓
    generación de vista previa
        ↓
    confirmación del profesor
        ↓
    OCR posterior

CaptureAssistant no ejecuta directamente el OCR ni modifica el
motor semántico existente de Evalia.

Su responsabilidad es preparar y validar las hojas antes de que
sean enviadas al sistema de reconocimiento y evaluación.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .models import (
    CaptureResult,
    DetectedPage,
    QualityReport,
    StudentPageGroup,
    create_failed_capture_result,
)
from .page_detector import PageDetector
from .page_organizer import PageOrganizer
from .preview import PreviewGenerator
from .quality_analyzer import QualityAnalyzer


logger = logging.getLogger(__name__)


class CaptureAssistant:
    """
    Coordina el flujo completo de Smart Capture.

    Parameters
    ----------
    quality_analyzer:
        Instancia personalizada de QualityAnalyzer.

    page_detector:
        Instancia personalizada de PageDetector.

    page_organizer:
        Instancia personalizada de PageOrganizer.

    preview_generator:
        Instancia personalizada de PreviewGenerator.

    pages_per_student:
        Cantidad esperada de páginas por estudiante.

    stop_on_bad_quality:
        Si es True, el procesamiento se detiene cuando la fotografía
        no cumple los criterios mínimos de calidad.

    generate_preview:
        Si es True, genera la imagen anotada para confirmación.

    raise_exceptions:
        Si es True, vuelve a lanzar las excepciones. Se recomienda
        usar False en producción para evitar que un error de captura
        interrumpa el funcionamiento general de Evalia.
    """

    def __init__(
        self,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        page_detector: Optional[PageDetector] = None,
        page_organizer: Optional[PageOrganizer] = None,
        preview_generator: Optional[PreviewGenerator] = None,
        pages_per_student: int = 1,
        stop_on_bad_quality: bool = False,
        generate_preview: bool = True,
        raise_exceptions: bool = False,
    ) -> None:

        if pages_per_student < 1:
            raise ValueError(
                "pages_per_student debe ser igual o mayor que 1."
            )

        self.pages_per_student = pages_per_student
        self.stop_on_bad_quality = stop_on_bad_quality
        self.generate_preview = generate_preview
        self.raise_exceptions = raise_exceptions

        self.quality_analyzer = (
            quality_analyzer
            if quality_analyzer is not None
            else QualityAnalyzer()
        )

        self.page_detector = (
            page_detector
            if page_detector is not None
            else PageDetector()
        )

        self.page_organizer = (
            page_organizer
            if page_organizer is not None
            else self._create_page_organizer()
        )

        self.preview_generator = (
            preview_generator
            if preview_generator is not None
            else PreviewGenerator()
        )

    # ========================================================
    # API pública principal
    # ========================================================

    def inspect(
        self,
        image: np.ndarray,
    ) -> CaptureResult:
        """
        Ejecuta el flujo completo de inspección Smart Capture.

        Parameters
        ----------
        image:
            Imagen en formato NumPy/OpenCV.

        Returns
        -------
        CaptureResult
            Resultado completo del procesamiento.
        """

        try:
            normalized_image = self._validate_and_normalize_image(
                image
            )

            quality_report = self._analyze_quality(
                normalized_image
            )

            if (
                self.stop_on_bad_quality
                and not quality_report.acceptable
            ):
                return CaptureResult(
                    success=False,
                    quality=quality_report,
                    pages_detected=0,
                    students_detected=0,
                    pages=[],
                    students=[],
                    preview=None,
                    message=(
                        "La fotografía no cumple los criterios "
                        "mínimos de calidad."
                    ),
                    warnings=list(
                        quality_report.warnings
                    ),
                    errors=[],
                    metadata={
                        "stage": "quality",
                        "processing_stopped": True,
                        "pages_per_student": (
                            self.pages_per_student
                        ),
                    },
                )

            pages = self._detect_pages(
                normalized_image
            )

            if not pages:
                preview = None

                if self.generate_preview:
                    preview = (
                        self.preview_generator.render_pages(
                            image=normalized_image,
                            pages=[],
                        )
                    )

                warnings = list(
                    quality_report.warnings
                )

                warnings.append(
                    "No se detectaron hojas en la fotografía."
                )

                return CaptureResult(
                    success=False,
                    quality=quality_report,
                    pages_detected=0,
                    students_detected=0,
                    pages=[],
                    students=[],
                    preview=preview,
                    message=(
                        "Smart Capture no pudo detectar hojas "
                        "en la imagen."
                    ),
                    warnings=warnings,
                    errors=[],
                    metadata={
                        "stage": "page_detection",
                        "processing_stopped": True,
                        "pages_per_student": (
                            self.pages_per_student
                        ),
                    },
                )

            student_groups = self._organize_pages(
                pages
            )

            preview = None

            if self.generate_preview:
                preview = self.preview_generator.render(
                    image=normalized_image,
                    groups=student_groups,
                )

            warnings = self._collect_warnings(
                quality=quality_report,
                groups=student_groups,
            )

            incomplete_students = sum(
                1
                for group in student_groups
                if not group.complete
            )

            success = (
                len(pages) > 0
                and len(student_groups) > 0
            )

            if incomplete_students > 0:
                message = (
                    "Las hojas fueron detectadas, pero existen "
                    "grupos incompletos que deben ser revisados."
                )
            elif not quality_report.acceptable:
                message = (
                    "Las hojas fueron detectadas, aunque la calidad "
                    "de la fotografía requiere revisión."
                )
            else:
                message = (
                    "Las hojas fueron detectadas y organizadas "
                    "correctamente. Confirma la vista previa antes "
                    "de continuar al OCR."
                )

            return CaptureResult(
                success=success,
                quality=quality_report,
                pages_detected=len(pages),
                students_detected=len(
                    student_groups
                ),
                pages=pages,
                students=student_groups,
                preview=preview,
                message=message,
                warnings=warnings,
                errors=[],
                metadata={
                    "stage": "completed",
                    "pages_per_student": (
                        self.pages_per_student
                    ),
                    "quality_acceptable": (
                        quality_report.acceptable
                    ),
                    "incomplete_students": (
                        incomplete_students
                    ),
                    "preview_generated": (
                        preview is not None
                    ),
                },
            )

        except Exception as error:
            return self._handle_exception(
                error=error,
                stage="inspect",
            )

    def process(
        self,
        image: np.ndarray,
    ) -> CaptureResult:
        """
        Alias de inspect() para mantener compatibilidad.
        """

        return self.inspect(image)

    def inspect_bytes(
        self,
        image_bytes: bytes,
    ) -> CaptureResult:
        """
        Decodifica una imagen recibida como bytes y ejecuta la
        inspección completa.

        Este método es especialmente útil para FastAPI.
        """

        try:
            image = self.decode_image(
                image_bytes
            )

            return self.inspect(image)

        except Exception as error:
            return self._handle_exception(
                error=error,
                stage="image_decoding",
            )

    # ========================================================
    # Métodos públicos auxiliares
    # ========================================================

    def analyze_quality(
        self,
        image: np.ndarray,
    ) -> QualityReport:
        """
        Ejecuta solamente el análisis de calidad.
        """

        normalized_image = self._validate_and_normalize_image(
            image
        )

        return self._analyze_quality(
            normalized_image
        )

    def detect_pages(
        self,
        image: np.ndarray,
    ) -> List[DetectedPage]:
        """
        Ejecuta solamente la detección de páginas.
        """

        normalized_image = self._validate_and_normalize_image(
            image
        )

        return self._detect_pages(
            normalized_image
        )

    def organize_pages(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[StudentPageGroup]:
        """
        Ejecuta solamente la organización de páginas.
        """

        return self._organize_pages(
            list(pages)
        )

    def create_preview(
        self,
        image: np.ndarray,
        groups: Sequence[StudentPageGroup],
    ) -> np.ndarray:
        """
        Genera la vista previa de una organización existente.
        """

        normalized_image = self._validate_and_normalize_image(
            image
        )

        return self.preview_generator.render(
            image=normalized_image,
            groups=groups,
        )

    @staticmethod
    def decode_image(
        image_bytes: bytes,
    ) -> np.ndarray:
        """
        Convierte bytes de una imagen en un numpy.ndarray BGR.
        """

        if image_bytes is None:
            raise ValueError(
                "No se recibieron bytes de imagen."
            )

        if not isinstance(
            image_bytes,
            (bytes, bytearray),
        ):
            raise TypeError(
                "image_bytes debe ser bytes o bytearray."
            )

        if len(image_bytes) == 0:
            raise ValueError(
                "El archivo de imagen está vacío."
            )

        buffer = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "No fue posible decodificar la imagen. "
                "Comprueba que el archivo sea JPG, JPEG o PNG."
            )

        return image

    # ========================================================
    # Adaptadores internos
    # ========================================================

    def _analyze_quality(
        self,
        image: np.ndarray,
    ) -> QualityReport:
        """
        Ejecuta QualityAnalyzer admitiendo distintos nombres
        de método durante el desarrollo.
        """

        analyzer = self.quality_analyzer

        if hasattr(analyzer, "analyze"):
            report = analyzer.analyze(image)

        elif hasattr(analyzer, "inspect"):
            report = analyzer.inspect(image)

        elif hasattr(analyzer, "evaluate"):
            report = analyzer.evaluate(image)

        else:
            raise AttributeError(
                "QualityAnalyzer debe implementar uno de los "
                "métodos: analyze(), inspect() o evaluate()."
            )

        if not isinstance(
            report,
            QualityReport,
        ):
            raise TypeError(
                "QualityAnalyzer debe devolver un QualityReport."
            )

        return report

    def _detect_pages(
        self,
        image: np.ndarray,
    ) -> List[DetectedPage]:
        """
        Ejecuta PageDetector admitiendo distintos nombres de método.
        """

        detector = self.page_detector

        if hasattr(detector, "detect"):
            detected = detector.detect(image)

        elif hasattr(detector, "detect_pages"):
            detected = detector.detect_pages(image)

        elif hasattr(detector, "process"):
            detected = detector.process(image)

        else:
            raise AttributeError(
                "PageDetector debe implementar uno de los métodos: "
                "detect(), detect_pages() o process()."
            )

        pages = self._extract_pages_from_detector_result(
            detected
        )

        self._validate_detected_pages(
            pages
        )

        return pages

    def _organize_pages(
        self,
        pages: List[DetectedPage],
    ) -> List[StudentPageGroup]:
        """
        Ejecuta PageOrganizer admitiendo distintas firmas.
        """

        if not pages:
            return []

        organizer = self.page_organizer

        grouped: Any

        if hasattr(organizer, "group"):
            try:
                grouped = organizer.group(
                    pages
                )

            except TypeError:
                grouped = organizer.group(
                    pages=pages
                )

        elif hasattr(organizer, "organize"):
            try:
                grouped = organizer.organize(
                    pages
                )

            except TypeError:
                grouped = organizer.organize(
                    pages=pages
                )

        elif hasattr(organizer, "regroup"):
            try:
                grouped = organizer.regroup(
                    pages
                )

            except TypeError:
                grouped = organizer.regroup(
                    pages=pages
                )

        else:
            raise AttributeError(
                "PageOrganizer debe implementar uno de los métodos: "
                "group(), organize() o regroup()."
            )

        groups = self._extract_groups_from_organizer_result(
            grouped
        )

        self._normalize_group_numbers(
            groups
        )

        self._normalize_page_numbers(
            groups
        )

        return groups

    def _create_page_organizer(
        self,
    ) -> PageOrganizer:
        """
        Crea PageOrganizer intentando pasar pages_per_student.

        La compatibilidad permite trabajar tanto con un constructor
        configurable como con uno sin parámetros.
        """

        try:
            return PageOrganizer(
                pages_per_student=(
                    self.pages_per_student
                )
            )

        except TypeError:
            try:
                return PageOrganizer(
                    expected_pages_per_student=(
                        self.pages_per_student
                    )
                )

            except TypeError:
                return PageOrganizer()

    # ========================================================
    # Normalización de resultados
    # ========================================================

    @staticmethod
    def _extract_pages_from_detector_result(
        detected: Any,
    ) -> List[DetectedPage]:
        """
        Extrae páginas aunque PageDetector devuelva una lista,
        una tupla o un objeto contenedor.
        """

        if detected is None:
            return []

        if isinstance(detected, list):
            return detected

        if isinstance(detected, tuple):
            if (
                len(detected) > 0
                and isinstance(
                    detected[0],
                    list,
                )
            ):
                return detected[0]

            return list(detected)

        pages_attribute = getattr(
            detected,
            "pages",
            None,
        )

        if pages_attribute is not None:
            return list(
                pages_attribute
            )

        detected_pages_attribute = getattr(
            detected,
            "detected_pages",
            None,
        )

        if detected_pages_attribute is not None:
            return list(
                detected_pages_attribute
            )

        raise TypeError(
            "El resultado de PageDetector no contiene una "
            "lista reconocible de páginas."
        )

    @staticmethod
    def _extract_groups_from_organizer_result(
        grouped: Any,
    ) -> List[StudentPageGroup]:
        """
        Extrae los grupos aunque PageOrganizer devuelva una lista,
        tupla u objeto contenedor.
        """

        if grouped is None:
            return []

        if isinstance(grouped, list):
            return grouped

        if isinstance(grouped, tuple):
            if (
                len(grouped) > 0
                and isinstance(
                    grouped[0],
                    list,
                )
            ):
                return grouped[0]

            return list(grouped)

        groups_attribute = getattr(
            grouped,
            "groups",
            None,
        )

        if groups_attribute is not None:
            return list(
                groups_attribute
            )

        students_attribute = getattr(
            grouped,
            "students",
            None,
        )

        if students_attribute is not None:
            return list(
                students_attribute
            )

        raise TypeError(
            "El resultado de PageOrganizer no contiene una "
            "lista reconocible de grupos."
        )

    @staticmethod
    def _validate_detected_pages(
        pages: Sequence[DetectedPage],
    ) -> None:
        """
        Comprueba que el detector haya devuelto DetectedPage.
        """

        for index, page in enumerate(
            pages,
            start=1,
        ):
            if not isinstance(
                page,
                DetectedPage,
            ):
                raise TypeError(
                    "PageDetector devolvió un objeto incompatible "
                    f"en la posición {index}. Se esperaba "
                    "DetectedPage."
                )

    @staticmethod
    def _normalize_group_numbers(
        groups: List[StudentPageGroup],
    ) -> None:
        """
        Asigna números secuenciales cuando sean inexistentes o
        inválidos.
        """

        for index, group in enumerate(
            groups,
            start=1,
        ):
            if not isinstance(
                group,
                StudentPageGroup,
            ):
                raise TypeError(
                    "PageOrganizer debe devolver objetos "
                    "StudentPageGroup."
                )

            student_number = getattr(
                group,
                "student_number",
                None,
            )

            if student_number is None:
                group.student_number = index

                continue

            try:
                group.student_number = int(
                    student_number
                )

            except (
                TypeError,
                ValueError,
            ):
                group.student_number = index

    @staticmethod
    def _normalize_page_numbers(
        groups: List[StudentPageGroup],
    ) -> None:
        """
        Garantiza numeración secuencial dentro de cada estudiante.
        """

        for group in groups:
            for page_index, page in enumerate(
                group.pages,
                start=1,
            ):
                page.page_number = page_index

    # ========================================================
    # Validación de entrada
    # ========================================================

    @staticmethod
    def _validate_and_normalize_image(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Valida y normaliza una imagen a formato BGR uint8.
        """

        if image is None:
            raise ValueError(
                "No se recibió una imagen."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "La imagen debe ser un numpy.ndarray."
            )

        if image.size == 0:
            raise ValueError(
                "La imagen recibida está vacía."
            )

        if image.ndim not in (
            2,
            3,
        ):
            raise ValueError(
                "La imagen debe tener dos o tres dimensiones."
            )

        normalized = image

        if normalized.dtype != np.uint8:
            normalized = CaptureAssistant._convert_to_uint8(
                normalized
            )

        if normalized.ndim == 2:
            normalized = cv2.cvtColor(
                normalized,
                cv2.COLOR_GRAY2BGR,
            )

        elif normalized.shape[2] == 1:
            normalized = cv2.cvtColor(
                normalized[:, :, 0],
                cv2.COLOR_GRAY2BGR,
            )

        elif normalized.shape[2] == 4:
            normalized = cv2.cvtColor(
                normalized,
                cv2.COLOR_BGRA2BGR,
            )

        elif normalized.shape[2] != 3:
            raise ValueError(
                "La imagen debe estar en formato gris, BGR o BGRA."
            )

        return np.ascontiguousarray(
            normalized
        )

    @staticmethod
    def _convert_to_uint8(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convierte una imagen numérica a uint8.
        """

        if not np.issubdtype(
            image.dtype,
            np.number,
        ):
            raise TypeError(
                "La imagen contiene un tipo de datos no numérico."
            )

        finite_image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        minimum = float(
            np.min(finite_image)
        )

        maximum = float(
            np.max(finite_image)
        )

        if minimum >= 0.0 and maximum <= 1.0:
            finite_image = (
                finite_image * 255.0
            )

        finite_image = np.clip(
            finite_image,
            0.0,
            255.0,
        )

        return finite_image.astype(
            np.uint8
        )

    # ========================================================
    # Advertencias y errores
    # ========================================================

    @staticmethod
    def _collect_warnings(
        quality: QualityReport,
        groups: Sequence[StudentPageGroup],
    ) -> List[str]:
        """
        Consolida advertencias sin duplicarlas.
        """

        warnings: List[str] = []

        for warning in quality.warnings:
            if warning not in warnings:
                warnings.append(
                    warning
                )

        for group in groups:
            for warning in group.warnings:
                if warning not in warnings:
                    warnings.append(
                        warning
                    )

            if not group.complete:
                warning = (
                    f"El estudiante {group.student_number} "
                    "tiene un grupo incompleto."
                )

                if warning not in warnings:
                    warnings.append(
                        warning
                    )

        return warnings

    def _handle_exception(
        self,
        error: Exception,
        stage: str,
    ) -> CaptureResult:
        """
        Registra una excepción y devuelve un resultado controlado.
        """

        error_message = (
            f"{type(error).__name__}: {error}"
        )

        logger.exception(
            "Error en Smart Capture durante la etapa %s",
            stage,
        )

        if self.raise_exceptions:
            raise error

        result = create_failed_capture_result(
            message=(
                "Smart Capture no pudo completar el "
                "procesamiento de la imagen."
            ),
            error=error_message,
        )

        result.metadata.update(
            {
                "stage": stage,
                "exception_type": (
                    type(error).__name__
                ),
                "pages_per_student": (
                    self.pages_per_student
                ),
            }
        )

        # El traceback es útil durante desarrollo, pero puede
        # eliminarse en producción si se desea.
        result.metadata["traceback"] = (
            traceback.format_exc()
        )

        return result

    # ========================================================
    # Serialización para API
    # ========================================================

    def result_to_api_dict(
        self,
        result: CaptureResult,
        include_preview_base64: bool = False,
        preview_format: str = "jpeg",
    ) -> Dict[str, Any]:
        """
        Convierte CaptureResult en un diccionario apto para FastAPI.

        Parameters
        ----------
        result:
            Resultado generado por inspect().

        include_preview_base64:
            Si es True, incorpora la vista previa en Base64.

        preview_format:
            Formato de la vista previa: jpeg o png.
        """

        response = result.to_dict(
            include_images=False,
            include_preview=False,
        )

        if (
            include_preview_base64
            and result.has_preview
        ):
            import base64

            normalized_format = (
                preview_format
                .strip()
                .lower()
            )

            if normalized_format in (
                "jpg",
                "jpeg",
            ):
                encoded = (
                    self.preview_generator.encode_jpeg(
                        result.preview
                    )
                )

                mime_type = "image/jpeg"

            elif normalized_format == "png":
                encoded = (
                    self.preview_generator.encode_png(
                        result.preview
                    )
                )

                mime_type = "image/png"

            else:
                raise ValueError(
                    "preview_format debe ser jpeg, jpg o png."
                )

            encoded_base64 = base64.b64encode(
                encoded
            ).decode(
                "utf-8"
            )

            response["preview"] = {
                "mime_type": mime_type,
                "base64": encoded_base64,
                "data_url": (
                    f"data:{mime_type};base64,"
                    f"{encoded_base64}"
                ),
            }

        return response
