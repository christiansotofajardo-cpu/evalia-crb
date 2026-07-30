"""
models.py
=========

Modelos de datos compartidos por Smart Capture.

Este archivo define las estructuras que conectan:

- QualityAnalyzer
- PageDetector
- PageOrganizer
- PreviewGenerator
- CaptureAssistant

No contiene lógica de procesamiento de imágenes.
Su función es mantener un contrato común entre los módulos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Tipos auxiliares
# ============================================================

Point = Tuple[float, float]
BoundingBox = Tuple[float, float, float, float]


# ============================================================
# Página detectada
# ============================================================

@dataclass
class DetectedPage:
    """
    Representa una hoja detectada dentro de una fotografía.

    Parameters
    ----------
    page_id:
        Identificador interno único dentro de la captura.

    corners:
        Cuatro vértices de la hoja en la imagen original.

    bounding_box:
        Caja delimitadora en formato:

        (x, y, width, height)

    width:
        Ancho aproximado de la página detectada.

    height:
        Alto aproximado de la página detectada.

    center_x:
        Coordenada horizontal del centro.

    center_y:
        Coordenada vertical del centro.

    area:
        Área del contorno detectado.

    confidence:
        Confianza aproximada de la detección entre 0 y 1.

    page_number:
        Número de página dentro del grupo del estudiante.

    cropped_image:
        Imagen recortada o rectificada de la hoja.

    student_name:
        Nombre leído desde el encabezado de la hoja.

    student_id:
        Código, matrícula u otro identificador reconocido.

    identity_text:
        Texto bruto devuelto por el OCR de identidad.

    identity_confidence:
        Confianza de la lectura de identidad entre 0 y 1.

    identity_status:
        Estado de la lectura: pending, recognized, not_recognized,
        reader_not_configured, reader_error o missing_crop.

    metadata:
        Información adicional del detector.
    """

    page_id: int
    corners: List[Point]
    bounding_box: BoundingBox
    width: float
    height: float
    center_x: float
    center_y: float

    area: float = 0.0
    confidence: float = 1.0
    page_number: Optional[int] = None
    cropped_image: Optional[np.ndarray] = None

    # Metadatos espaciales asignados por PageOrganizer.
    layout_row: Optional[int] = None
    layout_column: Optional[int] = None
    visual_order: Optional[int] = None

    # Identidad y pertenencia.
    student_number: Optional[int] = None
    student_name: Optional[str] = None
    student_id: Optional[str] = None
    identity_text: Optional[str] = None
    identity_confidence: Optional[float] = None
    identity_status: str = "pending"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def center(self) -> Point:
        """
        Devuelve el centro de la página.
        """

        return (
            self.center_x,
            self.center_y,
        )

    @property
    def bbox(self) -> BoundingBox:
        """
        Alias de bounding_box para compatibilidad.
        """

        return self.bounding_box

    @property
    def aspect_ratio(self) -> float:
        """
        Calcula la relación ancho/alto.
        """

        if self.height == 0:
            return 0.0

        return float(
            self.width / self.height
        )

    @property
    def has_crop(self) -> bool:
        """
        Indica si la página contiene una imagen recortada.
        """

        return (
            self.cropped_image is not None
            and isinstance(
                self.cropped_image,
                np.ndarray,
            )
            and self.cropped_image.size > 0
        )

    @property
    def has_identity(self) -> bool:
        """
        Indica si la página posee nombre o identificador reconocido.
        """

        return bool(
            (
                self.student_name
                and self.student_name.strip()
            )
            or (
                self.student_id
                and self.student_id.strip()
            )
        )

    def to_dict(
        self,
        include_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Convierte la página en un diccionario serializable.

        La imagen recortada no se incluye por defecto porque un
        numpy.ndarray no puede enviarse directamente como JSON.
        """

        result: Dict[str, Any] = {
            "page_id": self.page_id,
            "corners": [
                [
                    float(point[0]),
                    float(point[1]),
                ]
                for point in self.corners
            ],
            "bounding_box": [
                float(value)
                for value in self.bounding_box
            ],
            "width": float(self.width),
            "height": float(self.height),
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "area": float(self.area),
            "confidence": float(self.confidence),
            "page_number": self.page_number,
            "aspect_ratio": self.aspect_ratio,
            "has_crop": self.has_crop,
            "layout_row": self.layout_row,
            "layout_column": self.layout_column,
            "visual_order": self.visual_order,
            "student_number": self.student_number,
            "student_name": self.student_name,
            "student_id": self.student_id,
            "identity_text": self.identity_text,
            "identity_confidence": (
                self.identity_confidence
            ),
            "identity_status": self.identity_status,
            "has_identity": self.has_identity,
            "metadata": self.metadata,
        }

        if include_image:
            result["cropped_image"] = (
                self.cropped_image
            )

        return result


# ============================================================
# Grupo de páginas por estudiante
# ============================================================

@dataclass
class StudentPageGroup:
    """
    Agrupa las páginas atribuidas a un mismo estudiante.

    Parameters
    ----------
    student_number:
        Número secuencial asignado al estudiante dentro de
        la fotografía.

    pages:
        Lista ordenada de páginas del estudiante.

    complete:
        Indica si el grupo contiene el número esperado de páginas.

    student_id:
        Identificador real del estudiante, cuando posteriormente
        sea reconocido o ingresado por el profesor.

    student_name:
        Nombre reconocido desde el encabezado de sus hojas.

    identity_confidence:
        Confianza promedio de la identificación.

    identity_status:
        Estado consolidado de la identidad del grupo.

    confirmed:
        Indica si el profesor confirmó manualmente el grupo.

    warnings:
        Observaciones asociadas a la organización.

    metadata:
        Información adicional de la agrupación.
    """

    student_number: int
    pages: List[DetectedPage] = field(
        default_factory=list
    )
    complete: bool = False

    student_id: Optional[str] = None
    student_name: Optional[str] = None
    identity_confidence: Optional[float] = None
    identity_status: str = "pending"

    confirmed: bool = False
    warnings: List[str] = field(
        default_factory=list
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def page_count(self) -> int:
        """
        Número de páginas del grupo.
        """

        return len(self.pages)

    @property
    def identity_label(self) -> str:
        """
        Etiqueta legible para la interfaz.
        """

        if self.student_name:
            return self.student_name

        if self.student_id:
            return self.student_id

        return f"Estudiante {self.student_number}"

    @property
    def identity_resolved(self) -> bool:
        """
        Indica si el grupo posee nombre o código reconocido.
        """

        return bool(
            self.student_name
            or self.student_id
        )

    @property
    def requires_confirmation(self) -> bool:
        """
        Indica si el grupo requiere revisión del profesor.
        """

        return (
            not self.complete
            or bool(self.warnings)
            or not self.confirmed
        )

    def get_page(
        self,
        page_number: int,
    ) -> Optional[DetectedPage]:
        """
        Busca una página por su número dentro del grupo.
        """

        for page in self.pages:
            if page.page_number == page_number:
                return page

        return None

    def to_dict(
        self,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Convierte el grupo en un diccionario.
        """

        return {
            "student_number": self.student_number,
            "student_id": self.student_id,
            "student_name": self.student_name,
            "identity_label": self.identity_label,
            "identity_confidence": (
                self.identity_confidence
            ),
            "identity_status": self.identity_status,
            "identity_resolved": (
                self.identity_resolved
            ),
            "complete": self.complete,
            "confirmed": self.confirmed,
            "page_count": self.page_count,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
            "pages": [
                page.to_dict(
                    include_image=include_images
                )
                for page in self.pages
            ],
        }


# ============================================================
# Informe de calidad
# ============================================================

@dataclass
class QualityReport:
    """
    Resultado del análisis técnico de la fotografía.

    Los indicadores se conservan por separado para permitir
    calibraciones futuras con datos reales.
    """

    acceptable: bool
    quality_score: float

    width: int
    height: int

    blur_score: float
    brightness: float
    contrast: float

    dark_ratio: float
    bright_ratio: float

    warnings: List[str] = field(
        default_factory=list
    )

    recommendations: List[str] = field(
        default_factory=list
    )

    message: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def resolution(self) -> Tuple[int, int]:
        """
        Resolución en formato ancho por alto.
        """

        return (
            self.width,
            self.height,
        )

    @property
    def requires_new_photo(self) -> bool:
        """
        Indica si conviene solicitar una nueva fotografía.
        """

        return not self.acceptable

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el informe a un diccionario serializable.
        """

        return {
            "acceptable": self.acceptable,
            "quality_score": float(
                self.quality_score
            ),
            "width": int(self.width),
            "height": int(self.height),
            "resolution": [
                int(self.width),
                int(self.height),
            ],
            "blur_score": float(
                self.blur_score
            ),
            "brightness": float(
                self.brightness
            ),
            "contrast": float(
                self.contrast
            ),
            "dark_ratio": float(
                self.dark_ratio
            ),
            "bright_ratio": float(
                self.bright_ratio
            ),
            "warnings": list(
                self.warnings
            ),
            "recommendations": list(
                self.recommendations
            ),
            "message": self.message,
            "requires_new_photo": (
                self.requires_new_photo
            ),
            "metadata": self.metadata,
        }


# ============================================================
# Resultado general de Smart Capture
# ============================================================

@dataclass
class CaptureResult:
    """
    Resultado completo de una inspección Smart Capture.

    Reúne:

    - calidad de la fotografía;
    - páginas detectadas;
    - grupos por estudiante;
    - vista previa;
    - estado general del procesamiento.
    """

    success: bool

    quality: QualityReport

    pages_detected: int
    students_detected: int

    pages: List[DetectedPage] = field(
        default_factory=list
    )

    students: List[StudentPageGroup] = field(
        default_factory=list
    )

    preview: Optional[np.ndarray] = None

    message: str = ""

    errors: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def has_preview(self) -> bool:
        """
        Indica si se generó una vista previa.
        """

        return (
            self.preview is not None
            and isinstance(
                self.preview,
                np.ndarray,
            )
            and self.preview.size > 0
        )

    @property
    def complete_students(self) -> int:
        """
        Número de estudiantes con grupos completos.
        """

        return sum(
            1
            for student in self.students
            if student.complete
        )

    @property
    def incomplete_students(self) -> int:
        """
        Número de estudiantes con grupos incompletos.
        """

        return sum(
            1
            for student in self.students
            if not student.complete
        )

    @property
    def identified_students(self) -> int:
        """
        Número de estudiantes con identidad reconocida.
        """

        return sum(
            1
            for student in self.students
            if student.identity_resolved
        )

    @property
    def unidentified_students(self) -> int:
        """
        Número de estudiantes pendientes de identificación.
        """

        return (
            self.students_detected
            - self.identified_students
        )

    @property
    def requires_confirmation(self) -> bool:
        """
        Indica si el profesor debe revisar la organización.
        """

        if not self.success:
            return True

        if not self.quality.acceptable:
            return True

        if self.pages_detected == 0:
            return True

        if self.incomplete_students > 0:
            return True

        return any(
            student.requires_confirmation
            for student in self.students
        )

    @property
    def ready_for_ocr(self) -> bool:
        """
        Indica si el resultado está técnicamente listo para OCR.

        En esta etapa no exige confirmación manual, porque esa
        confirmación se realizará posteriormente en la interfaz.
        """

        return (
            self.success
            and self.quality.acceptable
            and self.pages_detected > 0
            and self.incomplete_students == 0
        )

    def get_cropped_pages(
        self,
    ) -> List[np.ndarray]:
        """
        Entrega las imágenes recortadas disponibles para OCR.
        """

        cropped_pages: List[np.ndarray] = []

        for page in self.pages:
            if page.has_crop:
                cropped_pages.append(
                    page.cropped_image
                )

        return cropped_pages

    def to_dict(
        self,
        include_images: bool = False,
        include_preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Convierte el resultado en un diccionario.

        Las imágenes no se incluyen por defecto porque requieren
        codificación previa, por ejemplo en JPEG y Base64.
        """

        result: Dict[str, Any] = {
            "success": self.success,
            "message": self.message,
            "pages_detected": (
                self.pages_detected
            ),
            "students_detected": (
                self.students_detected
            ),
            "identified_students": (
                self.identified_students
            ),
            "unidentified_students": (
                self.unidentified_students
            ),
            "complete_students": (
                self.complete_students
            ),
            "incomplete_students": (
                self.incomplete_students
            ),
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "ready_for_ocr": (
                self.ready_for_ocr
            ),
            "quality": self.quality.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": self.metadata,
            "pages": [
                page.to_dict(
                    include_image=include_images
                )
                for page in self.pages
            ],
            "students": [
                student.to_dict(
                    include_images=include_images
                )
                for student in self.students
            ],
            "has_preview": self.has_preview,
        }

        if include_preview:
            result["preview"] = self.preview

        return result


# ============================================================
# Funciones auxiliares
# ============================================================

def create_failed_quality_report(
    message: str,
) -> QualityReport:
    """
    Crea un informe de calidad vacío para casos en que la imagen
    no pudo analizarse.
    """

    return QualityReport(
        acceptable=False,
        quality_score=0.0,
        width=0,
        height=0,
        blur_score=0.0,
        brightness=0.0,
        contrast=0.0,
        dark_ratio=0.0,
        bright_ratio=0.0,
        warnings=[
            message
        ],
        recommendations=[
            "Revisa el archivo e intenta cargar nuevamente la fotografía."
        ],
        message=message,
    )


def create_failed_capture_result(
    message: str,
    error: Optional[str] = None,
) -> CaptureResult:
    """
    Crea un resultado fallido consistente.

    Permite que CaptureAssistant entregue siempre un CaptureResult,
    incluso cuando ocurre una excepción.
    """

    errors = []

    if error:
        errors.append(error)

    return CaptureResult(
        success=False,
        quality=create_failed_quality_report(
            message
        ),
        pages_detected=0,
        students_detected=0,
        pages=[],
        students=[],
        preview=None,
        message=message,
        errors=errors,
        warnings=[],
    )
