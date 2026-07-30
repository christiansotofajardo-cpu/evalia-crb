"""
capture_assistant.py
====================

Orquestador principal de Evalia Smart Capture.

Flujo de esta versión:

    fotografía
        ↓
    análisis de calidad
        ↓
    detección y separación multipágina
        ↓
    orden espacial
        ↓
    lectura de identidad en el encabezado
        ↓
    agrupación por nombre/código
        ↓
    vista previa y confirmación
        ↓
    OCR académico posterior

La identidad se obtiene mediante ``identity_reader``. Puede ser una función
o un objeto adaptador conectado al OCR existente de Evalia.
"""

from __future__ import annotations

import base64
import logging
import re
import traceback
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from .multi_page_splitter import MultiPageSplitter
from .page_organizer import PageOrganizer
from .preview import PreviewGenerator
from .quality_analyzer import QualityAnalyzer


logger = logging.getLogger(__name__)

IdentityReader = Callable[..., Any]


class CaptureAssistant:
    """
    Coordina el flujo completo de Smart Capture.

    Parameters
    ----------
    pages_per_student:
        Número esperado de páginas por estudiante. Para la prueba actual
        debe ser 1: cuatro hojas corresponden a cuatro estudiantes.

    identity_reader:
        Función u objeto que recibe el recorte superior de cada hoja y
        devuelve la identidad reconocida. Formatos admitidos:

        - "Nombre Apellido"
        - {"student_name": "...", "student_id": "...", "confidence": 0.9}
        - ("Nombre Apellido", 0.9)

        Un objeto puede implementar read_identity(), read_name(), read(),
        extract() o recognize().

    identity_header_ratio:
        Proporción superior del recorte enviada al lector de identidad.

    require_identity:
        Si es True, una hoja sin identidad reconocida genera advertencia.
    """

    def __init__(
        self,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        page_detector: Optional[PageDetector] = None,
        multi_page_splitter: Optional[MultiPageSplitter] = None,
        page_organizer: Optional[PageOrganizer] = None,
        preview_generator: Optional[PreviewGenerator] = None,
        identity_reader: Optional[IdentityReader] = None,
        pages_per_student: int = 1,
        identity_header_ratio: float = 0.24,
        require_identity: bool = True,
        stop_on_bad_quality: bool = False,
        generate_preview: bool = True,
        raise_exceptions: bool = False,
    ) -> None:

        if pages_per_student < 1:
            raise ValueError(
                "pages_per_student debe ser igual o mayor que 1."
            )

        if not 0.10 <= identity_header_ratio <= 0.50:
            raise ValueError(
                "identity_header_ratio debe estar entre 0.10 y 0.50."
            )

        self.pages_per_student = pages_per_student
        self.identity_header_ratio = identity_header_ratio
        self.require_identity = require_identity
        self.stop_on_bad_quality = stop_on_bad_quality
        self.generate_preview = generate_preview
        self.raise_exceptions = raise_exceptions
        self.identity_reader = identity_reader

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
        self.multi_page_splitter = (
            multi_page_splitter
            if multi_page_splitter is not None
            else MultiPageSplitter()
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

    def inspect(
        self,
        image: np.ndarray,
    ) -> CaptureResult:
        """Ejecuta la inspección completa."""

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
                    warnings=list(quality_report.warnings),
                    errors=[],
                    metadata={
                        "stage": "quality",
                        "processing_stopped": True,
                        "pages_per_student": self.pages_per_student,
                    },
                )

            pages = self._detect_pages(normalized_image)

            if not pages:
                preview = None

                if self.generate_preview:
                    preview = self.preview_generator.render_pages(
                        image=normalized_image,
                        pages=[],
                    )

                warnings = list(quality_report.warnings)
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
                        "pages_per_student": self.pages_per_student,
                    },
                )

            pages = self._order_pages(pages)
            self._read_page_identities(pages)
            student_groups = self._group_pages_by_identity(pages)

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

            identified_students = sum(
                1
                for group in student_groups
                if group.identity_resolved
            )
            unresolved_pages = sum(
                1
                for page in pages
                if not page.has_identity
            )
            incomplete_students = sum(
                1
                for group in student_groups
                if not group.complete
            )

            success = bool(pages and student_groups)

            if unresolved_pages > 0:
                message = (
                    "Las hojas fueron detectadas y separadas, pero "
                    "hay identidades que deben confirmarse."
                )
            elif incomplete_students > 0:
                message = (
                    "Los estudiantes fueron identificados, pero existen "
                    "grupos de páginas incompletos."
                )
            elif not quality_report.acceptable:
                message = (
                    "Los estudiantes fueron identificados, aunque la "
                    "calidad de la fotografía requiere revisión."
                )
            else:
                message = (
                    "Las hojas fueron detectadas, ordenadas e "
                    "identificadas correctamente."
                )

            return CaptureResult(
                success=success,
                quality=quality_report,
                pages_detected=len(pages),
                students_detected=len(student_groups),
                pages=pages,
                students=student_groups,
                preview=preview,
                message=message,
                warnings=warnings,
                errors=[],
                metadata={
                    "stage": "completed",
                    "pages_per_student": self.pages_per_student,
                    "quality_acceptable": quality_report.acceptable,
                    "identified_students": identified_students,
                    "unresolved_identity_pages": unresolved_pages,
                    "incomplete_students": incomplete_students,
                    "identity_reader_configured": (
                        self.identity_reader is not None
                    ),
                    "identity_header_ratio": (
                        self.identity_header_ratio
                    ),
                    "preview_generated": preview is not None,
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
        return self.inspect(image)

    def inspect_bytes(
        self,
        image_bytes: bytes,
    ) -> CaptureResult:
        try:
            image = self.decode_image(image_bytes)
            return self.inspect(image)
        except Exception as error:
            return self._handle_exception(
                error=error,
                stage="image_decoding",
            )

    def analyze_quality(
        self,
        image: np.ndarray,
    ) -> QualityReport:
        normalized = self._validate_and_normalize_image(image)
        return self._analyze_quality(normalized)

    def detect_pages(
        self,
        image: np.ndarray,
    ) -> List[DetectedPage]:
        normalized = self._validate_and_normalize_image(image)
        return self._detect_pages(normalized)

    def organize_pages(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[StudentPageGroup]:
        ordered = self._order_pages(list(pages))
        self._read_page_identities(ordered)
        return self._group_pages_by_identity(ordered)

    def create_preview(
        self,
        image: np.ndarray,
        groups: Sequence[StudentPageGroup],
    ) -> np.ndarray:
        normalized = self._validate_and_normalize_image(image)
        return self.preview_generator.render(
            image=normalized,
            groups=groups,
        )

    @staticmethod
    def decode_image(
        image_bytes: bytes,
    ) -> np.ndarray:
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

    def _analyze_quality(
        self,
        image: np.ndarray,
    ) -> QualityReport:
        analyzer = self.quality_analyzer

        if hasattr(analyzer, "analyze"):
            report = analyzer.analyze(image)
        elif hasattr(analyzer, "inspect"):
            report = analyzer.inspect(image)
        elif hasattr(analyzer, "evaluate"):
            report = analyzer.evaluate(image)
        else:
            raise AttributeError(
                "QualityAnalyzer debe implementar analyze(), "
                "inspect() o evaluate()."
            )

        if not isinstance(report, QualityReport):
            raise TypeError(
                "QualityAnalyzer debe devolver un QualityReport."
            )

        return report

    def _detect_pages(
        self,
        image: np.ndarray,
    ) -> List[DetectedPage]:
        detector = self.page_detector

        if hasattr(detector, "detect"):
            detected = detector.detect(image)
        elif hasattr(detector, "detect_pages"):
            detected = detector.detect_pages(image)
        elif hasattr(detector, "process"):
            detected = detector.process(image)
        else:
            raise AttributeError(
                "PageDetector debe implementar detect(), "
                "detect_pages() o process()."
            )

        pages = self._extract_pages_from_detector_result(detected)
        self._validate_detected_pages(pages)

        pages = self.multi_page_splitter.split_pages(
            image=image,
            pages=pages,
        )
        self._validate_detected_pages(pages)

        return pages

    def _order_pages(
        self,
        pages: List[DetectedPage],
    ) -> List[DetectedPage]:
        if not pages:
            return []

        organizer = self.page_organizer

        if hasattr(organizer, "order_pages"):
            ordered = organizer.order_pages(pages)
            return list(ordered)

        # Compatibilidad con organizadores antiguos.
        return sorted(
            pages,
            key=lambda page: (
                float(page.center_y),
                float(page.center_x),
            ),
        )

    def _read_page_identities(
        self,
        pages: Sequence[DetectedPage],
    ) -> None:
        """
        Lee la zona superior de cada recorte.

        Si no hay lector configurado, conserva una identidad pendiente.
        Cada hoja sigue siendo un estudiante independiente para evitar
        agrupaciones falsas.
        """

        for page in pages:
            page.student_name = None
            page.student_id = None
            page.identity_text = None
            page.identity_confidence = None

            if not page.has_crop:
                page.identity_status = "missing_crop"
                page.metadata["identity_error"] = (
                    "La hoja no tiene cropped_image."
                )
                continue

            if self.identity_reader is None:
                page.identity_status = "reader_not_configured"
                continue

            header = self._extract_identity_header(
                page.cropped_image
            )

            try:
                raw_result = self._invoke_identity_reader(
                    header=header,
                    page=page,
                )
                parsed = self._parse_identity_result(
                    raw_result
                )

                page.student_name = parsed["student_name"]
                page.student_id = parsed["student_id"]
                page.identity_text = parsed["identity_text"]
                page.identity_confidence = parsed["confidence"]
                page.identity_status = (
                    "recognized"
                    if page.has_identity
                    else "not_recognized"
                )
                page.metadata["identity_header_shape"] = [
                    int(value)
                    for value in header.shape
                ]

            except Exception as error:
                logger.warning(
                    "No fue posible leer la identidad de page_id=%s: %s",
                    page.page_id,
                    error,
                )
                page.identity_status = "reader_error"
                page.metadata["identity_error"] = (
                    f"{type(error).__name__}: {error}"
                )

    def _extract_identity_header(
        self,
        cropped_image: np.ndarray,
    ) -> np.ndarray:
        height = int(cropped_image.shape[0])
        header_height = max(
            1,
            int(round(
                height * self.identity_header_ratio
            )),
        )

        header = cropped_image[:header_height, :]

        # Aumenta el tamaño para ayudar al OCR con letras pequeñas.
        if header.shape[1] < 1600:
            scale = 1600.0 / max(
                float(header.shape[1]),
                1.0,
            )
            header = cv2.resize(
                header,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        return np.ascontiguousarray(header)

    def _invoke_identity_reader(
        self,
        header: np.ndarray,
        page: DetectedPage,
    ) -> Any:
        reader = self.identity_reader

        if reader is None:
            return None

        if callable(reader):
            try:
                return reader(
                    image=header,
                    page=page,
                )
            except TypeError:
                try:
                    return reader(header, page)
                except TypeError:
                    return reader(header)

        for method_name in (
            "read_identity",
            "read_name",
            "read",
            "extract",
            "recognize",
        ):
            method = getattr(
                reader,
                method_name,
                None,
            )
            if method is None:
                continue

            try:
                return method(
                    image=header,
                    page=page,
                )
            except TypeError:
                try:
                    return method(header, page)
                except TypeError:
                    return method(header)

        raise AttributeError(
            "identity_reader debe ser callable o implementar "
            "read_identity(), read_name(), read(), extract() "
            "o recognize()."
        )

    def _parse_identity_result(
        self,
        result: Any,
    ) -> Dict[str, Any]:
        student_name: Optional[str] = None
        student_id: Optional[str] = None
        identity_text: Optional[str] = None
        confidence: Optional[float] = None

        if result is None:
            pass

        elif isinstance(result, str):
            identity_text = result
            student_name = self._clean_identity_text(result)

        elif isinstance(result, dict):
            student_name = self._first_nonempty(
                result,
                (
                    "student_name",
                    "name",
                    "nombre",
                    "full_name",
                ),
            )
            student_id = self._first_nonempty(
                result,
                (
                    "student_id",
                    "id",
                    "code",
                    "codigo",
                    "matricula",
                ),
            )
            identity_text = self._first_nonempty(
                result,
                (
                    "identity_text",
                    "raw_text",
                    "text",
                    "ocr_text",
                ),
            )
            confidence = self._safe_float(
                result.get(
                    "confidence",
                    result.get("score"),
                )
            )

            if student_name:
                student_name = self._clean_identity_text(
                    student_name
                )

        elif isinstance(result, (tuple, list)):
            if len(result) >= 1:
                student_name = self._clean_identity_text(
                    str(result[0])
                )
                identity_text = str(result[0])
            if len(result) >= 2:
                confidence = self._safe_float(
                    result[1]
                )
            if len(result) >= 3 and result[2]:
                student_id = str(result[2]).strip()

        else:
            identity_text = str(result)
            student_name = self._clean_identity_text(
                identity_text
            )

        if student_id:
            student_id = str(student_id).strip() or None

        if identity_text:
            identity_text = str(identity_text).strip() or None

        return {
            "student_name": student_name,
            "student_id": student_id,
            "identity_text": identity_text,
            "confidence": confidence,
        }

    def _group_pages_by_identity(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[StudentPageGroup]:
        """
        Agrupa por identidad reconocida.

        Una identidad ausente recibe una clave única por página. Así nunca
        se unen dos estudiantes distintos solo porque el OCR no leyó el
        nombre.
        """

        buckets: Dict[str, List[DetectedPage]] = {}
        labels: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        for page in pages:
            identity_key = self._identity_key(page)

            if identity_key is None:
                identity_key = (
                    f"unresolved-page-{page.page_id}"
                )

            buckets.setdefault(identity_key, []).append(page)
            labels[identity_key] = (
                page.student_name,
                page.student_id,
            )

        groups: List[StudentPageGroup] = []

        for student_number, (
            identity_key,
            group_pages,
        ) in enumerate(
            buckets.items(),
            start=1,
        ):
            ordered_pages = sorted(
                group_pages,
                key=lambda page: (
                    page.visual_order
                    if page.visual_order is not None
                    else 10**9,
                    page.center_y,
                    page.center_x,
                ),
            )

            for page_number, page in enumerate(
                ordered_pages,
                start=1,
            ):
                page.student_number = student_number
                page.page_number = page_number

            student_name, student_id = labels[
                identity_key
            ]

            confidences = [
                page.identity_confidence
                for page in ordered_pages
                if page.identity_confidence is not None
            ]

            identity_confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else None
            )

            complete = (
                len(ordered_pages)
                == self.pages_per_student
            )

            warnings: List[str] = []

            if not (student_name or student_id):
                warnings.append(
                    "No se reconoció el nombre o código "
                    "en el encabezado."
                )

            if not complete:
                warnings.append(
                    f"Se esperaban {self.pages_per_student} "
                    f"página(s) y se detectaron "
                    f"{len(ordered_pages)}."
                )

            group = StudentPageGroup(
                student_number=student_number,
                pages=ordered_pages,
                complete=complete,
                student_id=student_id,
                student_name=student_name,
                identity_confidence=identity_confidence,
                identity_status=(
                    "recognized"
                    if student_name or student_id
                    else "pending_confirmation"
                ),
                warnings=warnings,
                metadata={
                    "identity_key": identity_key,
                    "grouping_method": (
                        "recognized_identity"
                        if student_name or student_id
                        else "independent_unresolved_page"
                    ),
                },
            )
            groups.append(group)

        return groups

    def _identity_key(
        self,
        page: DetectedPage,
    ) -> Optional[str]:
        if page.student_id:
            normalized_id = self._normalize_key(
                page.student_id
            )
            if normalized_id:
                return f"id:{normalized_id}"

        if page.student_name:
            normalized_name = self._normalize_key(
                page.student_name
            )
            if normalized_name:
                return f"name:{normalized_name}"

        return None

    @staticmethod
    def _normalize_key(
        value: str,
    ) -> str:
        normalized = unicodedata.normalize(
            "NFKD",
            str(value),
        )
        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        normalized = normalized.casefold()
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized,
        )
        return " ".join(
            normalized.split()
        )

    @staticmethod
    def _clean_identity_text(
        value: str,
    ) -> Optional[str]:
        text = str(value).replace("\n", " ")
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip(" :-_\t")

        # Elimina rótulos frecuentes sin alterar el nombre.
        text = re.sub(
            r"^(nombre|estudiante|alumno|alumna)\s*[:\-]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if not text:
            return None

        # Evita aceptar párrafos completos como nombre.
        words = text.split()
        if len(words) > 8:
            return None

        return text

    @staticmethod
    def _first_nonempty(
        data: Dict[str, Any],
        keys: Sequence[str],
    ) -> Optional[str]:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            normalized = str(value).strip()
            if normalized:
                return normalized
        return None

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _create_page_organizer(
        self,
    ) -> PageOrganizer:
        try:
            return PageOrganizer(
                pages_per_student=self.pages_per_student,
                grouping_mode="sequential",
            )
        except TypeError:
            try:
                return PageOrganizer(
                    pages_per_student=self.pages_per_student
                )
            except TypeError:
                return PageOrganizer()

    @staticmethod
    def _extract_pages_from_detector_result(
        detected: Any,
    ) -> List[DetectedPage]:
        if detected is None:
            return []
        if isinstance(detected, list):
            return detected
        if isinstance(detected, tuple):
            if (
                len(detected) > 0
                and isinstance(detected[0], list)
            ):
                return detected[0]
            return list(detected)

        pages_attribute = getattr(
            detected,
            "pages",
            None,
        )
        if pages_attribute is not None:
            return list(pages_attribute)

        detected_pages_attribute = getattr(
            detected,
            "detected_pages",
            None,
        )
        if detected_pages_attribute is not None:
            return list(detected_pages_attribute)

        raise TypeError(
            "El resultado de PageDetector no contiene una "
            "lista reconocible de páginas."
        )

    @staticmethod
    def _validate_detected_pages(
        pages: Sequence[DetectedPage],
    ) -> None:
        for index, page in enumerate(
            pages,
            start=1,
        ):
            if not isinstance(page, DetectedPage):
                raise TypeError(
                    "PageDetector devolvió un objeto incompatible "
                    f"en la posición {index}. Se esperaba DetectedPage."
                )

    @staticmethod
    def _validate_and_normalize_image(
        image: np.ndarray,
    ) -> np.ndarray:
        if image is None:
            raise ValueError("No se recibió una imagen.")
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "La imagen debe ser un numpy.ndarray."
            )
        if image.size == 0:
            raise ValueError(
                "La imagen recibida está vacía."
            )
        if image.ndim not in (2, 3):
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

        return np.ascontiguousarray(normalized)

    @staticmethod
    def _convert_to_uint8(
        image: np.ndarray,
    ) -> np.ndarray:
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
        minimum = float(np.min(finite_image))
        maximum = float(np.max(finite_image))

        if minimum >= 0.0 and maximum <= 1.0:
            finite_image = finite_image * 255.0

        finite_image = np.clip(
            finite_image,
            0.0,
            255.0,
        )
        return finite_image.astype(np.uint8)

    def _collect_warnings(
        self,
        quality: QualityReport,
        groups: Sequence[StudentPageGroup],
    ) -> List[str]:
        warnings: List[str] = []

        for warning in quality.warnings:
            if warning not in warnings:
                warnings.append(warning)

        for group in groups:
            for warning in group.warnings:
                contextual_warning = (
                    f"{group.identity_label}: {warning}"
                )
                if contextual_warning not in warnings:
                    warnings.append(contextual_warning)

        if (
            self.require_identity
            and self.identity_reader is None
        ):
            warning = (
                "No hay un identity_reader conectado al OCR; "
                "las hojas se mantuvieron separadas para "
                "confirmación manual."
            )
            if warning not in warnings:
                warnings.append(warning)

        return warnings

    def _handle_exception(
        self,
        error: Exception,
        stage: str,
    ) -> CaptureResult:
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
                "exception_type": type(error).__name__,
                "pages_per_student": self.pages_per_student,
                "traceback": traceback.format_exc(),
            }
        )

        return result

    def result_to_api_dict(
        self,
        result: CaptureResult,
        include_preview_base64: bool = False,
        preview_format: str = "jpeg",
    ) -> Dict[str, Any]:
        response = result.to_dict(
            include_images=False,
            include_preview=False,
        )

        if (
            include_preview_base64
            and result.has_preview
        ):
            normalized_format = (
                preview_format.strip().lower()
            )

            if normalized_format in ("jpg", "jpeg"):
                encoded = self.preview_generator.encode_jpeg(
                    result.preview
                )
                mime_type = "image/jpeg"

            elif normalized_format == "png":
                encoded = self.preview_generator.encode_png(
                    result.preview
                )
                mime_type = "image/png"

            else:
                raise ValueError(
                    "preview_format debe ser jpeg, jpg o png."
                )

            encoded_base64 = base64.b64encode(
                encoded
            ).decode("utf-8")

            response["preview"] = {
                "mime_type": mime_type,
                "base64": encoded_base64,
                "data_url": (
                    f"data:{mime_type};base64,"
                    f"{encoded_base64}"
                ),
            }

        return response
