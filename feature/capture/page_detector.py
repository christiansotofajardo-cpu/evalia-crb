"""
Detector de páginas para Evalia Smart Capture.

Primera implementación:
- recibe una imagen OpenCV;
- detecta posibles hojas mediante contornos;
- conserva solamente cuadriláteros de tamaño razonable;
- devuelve sus coordenadas y propiedades geométricas.

Todavía no realiza corrección de perspectiva ni OCR.
"""

from typing import Any

import cv2
import numpy as np

from .models import DetectedPage


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Ordena cuatro esquinas en este orden:

    superior izquierda,
    superior derecha,
    inferior derecha,
    inferior izquierda.
    """

    points = points.reshape(4, 2).astype(np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    point_difference = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_difference)]
    ordered[3] = points[np.argmax(point_difference)]

    return ordered


def detect_pages(
    image: Any,
    minimum_area_ratio: float = 0.035,
    maximum_area_ratio: float = 0.45,
) -> list[DetectedPage]:
    """
    Detecta hojas dentro de una fotografía.

    Parameters
    ----------
    image:
        Imagen en formato OpenCV, BGR.
    minimum_area_ratio:
        Área mínima de una hoja respecto del área total.
    maximum_area_ratio:
        Área máxima permitida respecto del área total.

    Returns
    -------
    list[DetectedPage]:
        Hojas detectadas, ordenadas inicialmente por posición vertical
        y luego horizontal.
    """

    if image is None:
        raise ValueError("La imagen no puede ser None.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "La imagen debe ser un arreglo numpy compatible con OpenCV."
        )

    if image.ndim not in (2, 3):
        raise ValueError("La imagen tiene un formato no compatible.")

    image_height, image_width = image.shape[:2]
    image_area = float(image_height * image_width)

    if image_area <= 0:
        raise ValueError("La imagen no tiene dimensiones válidas.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blurred,
        threshold1=50,
        threshold2=150,
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

    candidates: list[DetectedPage] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        if area_ratio < minimum_area_ratio:
            continue

        if area_ratio > maximum_area_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)

        polygon = cv2.approxPolyDP(
            contour,
            epsilon=0.02 * perimeter,
            closed=True,
        )

        if len(polygon) != 4:
            continue

        if not cv2.isContourConvex(polygon):
            continue

        ordered_corners = order_corners(polygon)

        x, y, width, height = cv2.boundingRect(polygon)

        x_center = x + width / 2
        y_center = y + height / 2

        rectangularity = area / max(float(width * height), 1.0)

        confidence = max(
            0.0,
            min(1.0, rectangularity),
        )

        candidates.append(
            DetectedPage(
                page_id=0,
                image=None,
                corners=[
                    (float(px), float(py))
                    for px, py in ordered_corners
                ],
                confidence=confidence,
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
            )
        )

    candidates.sort(
        key=lambda page: (
            page.y_center,
            page.x_center,
        )
    )

    for index, page in enumerate(candidates, start=1):
        page.page_id = index

    return candidates


def draw_detected_pages(
    image: Any,
    pages: list[DetectedPage],
) -> Any:
    """
    Dibuja sobre una copia de la fotografía los contornos detectados.
    """

    if image is None:
        raise ValueError("La imagen no puede ser None.")

    preview = image.copy()

    for page in pages:
        polygon = np.array(
            page.corners,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            preview,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=4,
        )

        cv2.putText(
            preview,
            str(page.page_id),
            (
                int(page.x_center),
                int(page.y_center),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return preview
