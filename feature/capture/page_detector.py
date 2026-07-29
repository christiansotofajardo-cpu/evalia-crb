"""
Detector de páginas para Evalia Smart Capture.

Responsabilidades:
- recibir una imagen OpenCV;
- detectar posibles hojas mediante contornos;
- conservar cuadriláteros de tamaño razonable;
- ordenar las esquinas;
- crear objetos DetectedPage compatibles con models.py;
- ofrecer una interfaz orientada a objetos para CaptureAssistant.

Esta etapa todavía no ejecuta OCR.
"""

from __future__ import annotations

from typing import Any, List

import cv2
import numpy as np

from .models import DetectedPage


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Ordena cuatro esquinas en este orden:

    1. superior izquierda;
    2. superior derecha;
    3. inferior derecha;
    4. inferior izquierda.
    """

    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    point_difference = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_difference)]
    ordered[3] = points[np.argmax(point_difference)]

    return ordered


class PageDetector:
    """
    Detecta hojas dentro de una fotografía.

    Parameters
    ----------
    minimum_area_ratio:
        Área mínima de una hoja respecto del área total de la imagen.

    maximum_area_ratio:
        Área máxima de una hoja respecto del área total de la imagen.

    canny_threshold1:
        Umbral inferior del detector de bordes Canny.

    canny_threshold2:
        Umbral superior del detector de bordes Canny.

    polygon_epsilon_ratio:
        Precisión usada para aproximar cada contorno a un polígono.

    generate_crops:
        Si es True, genera un recorte simple de cada hoja detectada.
        En esta versión no aplica todavía corrección de perspectiva.
    """

    def __init__(
        self,
        minimum_area_ratio: float = 0.035,
        maximum_area_ratio: float = 0.45,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        polygon_epsilon_ratio: float = 0.02,
        generate_crops: bool = True,
    ) -> None:
        if minimum_area_ratio <= 0:
            raise ValueError(
                "minimum_area_ratio debe ser mayor que 0."
            )

        if maximum_area_ratio <= minimum_area_ratio:
            raise ValueError(
                "maximum_area_ratio debe ser mayor que "
                "minimum_area_ratio."
            )

        if polygon_epsilon_ratio <= 0:
            raise ValueError(
                "polygon_epsilon_ratio debe ser mayor que 0."
            )

        self.minimum_area_ratio = float(minimum_area_ratio)
        self.maximum_area_ratio = float(maximum_area_ratio)

        self.canny_threshold1 = int(canny_threshold1)
        self.canny_threshold2 = int(canny_threshold2)

        self.polygon_epsilon_ratio = float(
            polygon_epsilon_ratio
        )

        self.generate_crops = bool(generate_crops)

    def detect(
        self,
        image: np.ndarray,
    ) -> List[DetectedPage]:
        """
        Detecta hojas dentro de una fotografía.

        Returns
        -------
        List[DetectedPage]
            Hojas detectadas, ordenadas primero por posición vertical
            y luego por posición horizontal.
        """

        normalized_image = self._validate_image(image)

        image_height, image_width = normalized_image.shape[:2]
        image_area = float(image_height * image_width)

        gray = cv2.cvtColor(
            normalized_image,
            cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        edges = cv2.Canny(
            blurred,
            threshold1=self.canny_threshold1,
            threshold2=self.canny_threshold2,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        )

        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: List[DetectedPage] = []

        for contour in contours:
            page = self._contour_to_page(
                contour=contour,
                image=normalized_image,
                image_area=image_area,
            )

            if page is not None:
                candidates.append(page)

        candidates.sort(
            key=lambda page: (
                page.center_y,
                page.center_x,
            )
        )

        for index, page in enumerate(
            candidates,
            start=1,
        ):
            page.page_id = index

        return candidates

    def detect_pages(
        self,
        image: np.ndarray,
    ) -> List[DetectedPage]:
        """
        Alias de detect() para compatibilidad.
        """

        return self.detect(image)

    def process(
        self,
        image: np.ndarray,
    ) -> List[DetectedPage]:
        """
        Alias de detect() para compatibilidad.
        """

        return self.detect(image)

    def _contour_to_page(
        self,
        contour: np.ndarray,
        image: np.ndarray,
        image_area: float,
    ) -> DetectedPage | None:
        """
        Convierte un contorno válido en un objeto DetectedPage.
        """

        area = float(
            cv2.contourArea(contour)
        )

        if area <= 0:
            return None

        area_ratio = area / image_area

        if area_ratio < self.minimum_area_ratio:
            return None

        if area_ratio > self.maximum_area_ratio:
            return None

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            return None

        polygon = cv2.approxPolyDP(
            contour,
            epsilon=(
                self.polygon_epsilon_ratio
                * perimeter
            ),
            closed=True,
        )

        if len(polygon) != 4:
            return None

        if not cv2.isContourConvex(polygon):
            return None

        ordered_corners = order_corners(
            polygon
        )

        x, y, width, height = cv2.boundingRect(
            polygon
        )

        if width <= 0 or height <= 0:
            return None

        center_x = x + width / 2.0
        center_y = y + height / 2.0

        bounding_area = max(
            float(width * height),
            1.0,
        )

        rectangularity = area / bounding_area

        confidence = max(
            0.0,
            min(
                1.0,
                rectangularity,
            ),
        )

        cropped_image = None

        if self.generate_crops:
            cropped_image = self._create_crop(
                image=image,
                x=x,
                y=y,
                width=width,
                height=height,
            )

        return DetectedPage(
            page_id=0,
            corners=[
                (
                    float(point_x),
                    float(point_y),
                )
                for point_x, point_y in ordered_corners
            ],
            bounding_box=(
                float(x),
                float(y),
                float(width),
                float(height),
            ),
            width=float(width),
            height=float(height),
            center_x=float(center_x),
            center_y=float(center_y),
            area=float(area),
            confidence=float(confidence),
            page_number=None,
            cropped_image=cropped_image,
            metadata={
                "area_ratio": float(area_ratio),
                "rectangularity": float(
                    rectangularity
                ),
                "detection_method": (
                    "external_contour_quadrilateral"
                ),
                "perspective_corrected": False,
            },
        )

    @staticmethod
    def _create_crop(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray | None:
        """
        Genera un recorte rectangular simple.

        Todavía no realiza rectificación de perspectiva.
        """

        image_height, image_width = image.shape[:2]

        x_start = max(
            0,
            int(x),
        )

        y_start = max(
            0,
            int(y),
        )

        x_end = min(
            image_width,
            int(x + width),
        )

        y_end = min(
            image_height,
            int(y + height),
        )

        if (
            x_end <= x_start
            or y_end <= y_start
        ):
            return None

        crop = image[
            y_start:y_end,
            x_start:x_end,
        ].copy()

        if crop.size == 0:
            return None

        return crop

    @staticmethod
    def _validate_image(
        image: Any,
    ) -> np.ndarray:
        """
        Valida y normaliza la imagen a formato BGR uint8.
        """

        if image is None:
            raise ValueError(
                "La imagen no puede ser None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "La imagen debe ser un arreglo NumPy "
                "compatible con OpenCV."
            )

        if image.size == 0:
            raise ValueError(
                "La imagen está vacía."
            )

        if image.ndim not in (
            2,
            3,
        ):
            raise ValueError(
                "La imagen tiene un formato no compatible."
            )

        normalized = image

        if normalized.dtype != np.uint8:
            normalized = np.nan_to_num(
                normalized,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            if (
                float(np.min(normalized)) >= 0.0
                and float(np.max(normalized)) <= 1.0
            ):
                normalized = normalized * 255.0

            normalized = np.clip(
                normalized,
                0,
                255,
            ).astype(np.uint8)

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
                "La imagen debe estar en formato gris, "
                "BGR o BGRA."
            )

        return np.ascontiguousarray(
            normalized
        )


def detect_pages(
    image: Any,
    minimum_area_ratio: float = 0.035,
    maximum_area_ratio: float = 0.45,
) -> List[DetectedPage]:
    """
    Función de compatibilidad con la implementación inicial.

    Internamente utiliza PageDetector.
    """

    detector = PageDetector(
        minimum_area_ratio=minimum_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
    )

    return detector.detect(image)


def draw_detected_pages(
    image: Any,
    pages: List[DetectedPage],
) -> np.ndarray:
    """
    Dibuja los contornos detectados sobre una copia de la fotografía.

    Esta función se conserva para pruebas y compatibilidad.
    """

    if image is None:
        raise ValueError(
            "La imagen no puede ser None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "La imagen debe ser un arreglo NumPy."
        )

    preview = image.copy()

    if preview.ndim == 2:
        preview = cv2.cvtColor(
            preview,
            cv2.COLOR_GRAY2BGR,
        )

    for page in pages:
        polygon = np.array(
            page.corners,
            dtype=np.int32,
        ).reshape(
            (-1, 1, 2)
        )

        cv2.polylines(
            preview,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=4,
        )

        label = f"Hoja {page.page_id}"

        cv2.putText(
            preview,
            label,
            (
                int(page.center_x),
                int(page.center_y),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return preview
