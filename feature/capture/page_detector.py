"""
Detector de páginas para Evalia Smart Capture.

Responsabilidades:
- recibir una imagen OpenCV;
- detectar una o varias hojas mediante estrategias complementarias;
- tolerar fotografías de celular con fondos claros u oscuros;
- aceptar hojas que ocupan gran parte de la imagen;
- eliminar detecciones duplicadas;
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
    """Ordena cuatro esquinas: SI, SD, ID, II."""
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
    """Detector robusto de una o varias hojas en fotografías de celular."""

    def __init__(
        self,
        minimum_area_ratio: float = 0.025,
        maximum_area_ratio: float = 0.97,
        canny_threshold1: int = 40,
        canny_threshold2: int = 140,
        polygon_epsilon_ratio: float = 0.025,
        minimum_rectangularity: float = 0.52,
        minimum_aspect_ratio: float = 0.30,
        maximum_aspect_ratio: float = 2.50,
        duplicate_iou_threshold: float = 0.72,
        generate_crops: bool = True,
        max_pages: int = 20,
    ) -> None:
        if minimum_area_ratio <= 0:
            raise ValueError("minimum_area_ratio debe ser mayor que 0.")
        if maximum_area_ratio <= minimum_area_ratio:
            raise ValueError("maximum_area_ratio debe ser mayor que minimum_area_ratio.")
        if maximum_area_ratio > 1:
            raise ValueError("maximum_area_ratio no puede ser mayor que 1.")
        if polygon_epsilon_ratio <= 0:
            raise ValueError("polygon_epsilon_ratio debe ser mayor que 0.")

        self.minimum_area_ratio = float(minimum_area_ratio)
        self.maximum_area_ratio = float(maximum_area_ratio)
        self.canny_threshold1 = int(canny_threshold1)
        self.canny_threshold2 = int(canny_threshold2)
        self.polygon_epsilon_ratio = float(polygon_epsilon_ratio)
        self.minimum_rectangularity = float(minimum_rectangularity)
        self.minimum_aspect_ratio = float(minimum_aspect_ratio)
        self.maximum_aspect_ratio = float(maximum_aspect_ratio)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.generate_crops = bool(generate_crops)
        self.max_pages = max(1, int(max_pages))

    def detect(self, image: np.ndarray) -> List[DetectedPage]:
        normalized_image = self._validate_image(image)
        image_height, image_width = normalized_image.shape[:2]
        image_area = float(image_height * image_width)

        gray = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        raw_candidates: list[dict] = []
        for mask_name, mask in self._build_detection_masks(gray):
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                candidate = self._contour_to_candidate(
                    contour=contour,
                    image=normalized_image,
                    image_area=image_area,
                    detection_method=mask_name,
                )
                if candidate is not None:
                    raw_candidates.append(candidate)

        unique_candidates = self._remove_duplicates(raw_candidates)

        # Evita conservar un gran contorno exterior cuando dentro de él ya se
        # detectaron dos o más hojas individuales. Esto es especialmente útil
        # en fotografías de 2, 4 o más páginas sobre una misma mesa.
        unique_candidates = self._suppress_enclosing_candidates(
            unique_candidates
        )

        unique_candidates.sort(
            key=lambda item: (
                item["page"].center_y,
                item["page"].center_x,
            )
        )

        pages = [item["page"] for item in unique_candidates[: self.max_pages]]
        for index, page in enumerate(pages, start=1):
            page.page_id = index
        return pages

    def detect_pages(self, image: np.ndarray) -> List[DetectedPage]:
        return self.detect(image)

    def process(self, image: np.ndarray) -> List[DetectedPage]:
        return self.detect(image)

    def _build_detection_masks(
        self,
        gray: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Construye máscaras complementarias para dos escenarios:

        1. una hoja grande, donde conviene cerrar bordes moderadamente;
        2. varias hojas cercanas, donde se deben preservar las separaciones
           y evitar que la morfología una todo el conjunto en un solo bloque.

        Las máscaras se combinan después mediante candidatos + deduplicación.
        """
        masks: list[tuple[str, np.ndarray]] = []

        median_value = float(np.median(gray))
        automatic_lower = int(max(0, 0.66 * median_value))
        automatic_upper = int(min(255, 1.33 * median_value))
        if automatic_upper <= automatic_lower:
            automatic_lower = self.canny_threshold1
            automatic_upper = self.canny_threshold2

        # --------------------------------------------------------
        # A. Canny detallado: prioriza separaciones entre páginas.
        # --------------------------------------------------------
        edges_detail = cv2.Canny(
            gray,
            automatic_lower,
            automatic_upper,
        )
        edges_detail = cv2.morphologyEx(
            edges_detail,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        masks.append(("canny_multipage_detail", edges_detail))

        # --------------------------------------------------------
        # B. Canny balanceado: conserva robustez para una hoja.
        # Menos agresivo que la versión anterior.
        # --------------------------------------------------------
        edges_balanced = cv2.Canny(
            gray,
            automatic_lower,
            automatic_upper,
        )
        edges_balanced = cv2.morphologyEx(
            edges_balanced,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        masks.append(("canny_balanced", edges_balanced))

        # --------------------------------------------------------
        # C. Umbral adaptativo normal e invertido.
        # La apertura elimina puentes delgados entre hojas; el cierre
        # recompone bordes sin fusionar páginas vecinas.
        # --------------------------------------------------------
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            7,
        )
        adaptive = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        adaptive = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        masks.append(("adaptive_light_multipage", adaptive))

        adaptive_inverse = cv2.bitwise_not(adaptive)
        adaptive_inverse = cv2.morphologyEx(
            adaptive_inverse,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        masks.append(("adaptive_dark_multipage", adaptive_inverse))

        # --------------------------------------------------------
        # D. Otsu normal e invertido, con morfología ligera.
        # --------------------------------------------------------
        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        otsu = cv2.morphologyEx(
            otsu,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        otsu = cv2.morphologyEx(
            otsu,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        masks.append(("otsu_light_multipage", otsu))

        otsu_inverse = cv2.bitwise_not(otsu)
        otsu_inverse = cv2.morphologyEx(
            otsu_inverse,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        masks.append(("otsu_dark_multipage", otsu_inverse))

        return masks


    def _contour_to_candidate(
        self,
        contour: np.ndarray,
        image: np.ndarray,
        image_area: float,
        detection_method: str,
    ) -> dict | None:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            return None

        area_ratio = contour_area / image_area
        if area_ratio < self.minimum_area_ratio or area_ratio > self.maximum_area_ratio:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        polygon = cv2.approxPolyDP(
            contour,
            epsilon=self.polygon_epsilon_ratio * perimeter,
            closed=True,
        )

        used_rotated_rectangle = False
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            corners = polygon.reshape(4, 2)
        else:
            rotated_rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rotated_rectangle)
            box_area = float(cv2.contourArea(box.astype(np.float32)))
            if box_area <= 0:
                return None
            contour_fill_ratio = contour_area / box_area
            if contour_fill_ratio < self.minimum_rectangularity:
                return None
            corners = box
            used_rotated_rectangle = True

        ordered_corners = order_corners(corners)
        x, y, width, height = cv2.boundingRect(ordered_corners.astype(np.int32))
        if width <= 0 or height <= 0:
            return None

        shorter_side = max(min(width, height), 1)
        longer_side = max(width, height)
        aspect_ratio = float(shorter_side) / float(longer_side)
        inverse_aspect_ratio = float(longer_side) / float(shorter_side)
        if aspect_ratio < self.minimum_aspect_ratio:
            return None
        if inverse_aspect_ratio > self.maximum_aspect_ratio:
            return None

        bounding_area = max(float(width * height), 1.0)
        rectangularity = contour_area / bounding_area
        if rectangularity < self.minimum_rectangularity:
            return None

        center_x = x + width / 2.0
        center_y = y + height / 2.0

        border_margin = max(2, int(min(image.shape[:2]) * 0.005))
        touches_border = bool(
            x <= border_margin
            or y <= border_margin
            or x + width >= image.shape[1] - border_margin
            or y + height >= image.shape[0] - border_margin
        )

        confidence = (
            0.50 * min(1.0, rectangularity)
            + 0.30 * min(1.0, area_ratio / 0.50)
            + 0.20 * (0.75 if touches_border else 1.0)
        )
        confidence = max(0.0, min(1.0, confidence))

        cropped_image = None
        perspective_corrected = False
        if self.generate_crops:
            cropped_image = self._create_perspective_crop(
                image=image,
                ordered_corners=ordered_corners,
            )
            perspective_corrected = cropped_image is not None
            if cropped_image is None:
                cropped_image = self._create_crop(
                    image=image,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )

        page = DetectedPage(
            page_id=0,
            corners=[
                (float(point_x), float(point_y))
                for point_x, point_y in ordered_corners
            ],
            bounding_box=(float(x), float(y), float(width), float(height)),
            width=float(width),
            height=float(height),
            center_x=float(center_x),
            center_y=float(center_y),
            area=float(contour_area),
            confidence=float(confidence),
            page_number=None,
            cropped_image=cropped_image,
            metadata={
                "area_ratio": float(area_ratio),
                "rectangularity": float(rectangularity),
                "aspect_ratio": float(aspect_ratio),
                "detection_method": detection_method,
                "rotated_rectangle_fallback": used_rotated_rectangle,
                "touches_border": touches_border,
                "perspective_corrected": perspective_corrected,
            },
        )

        return {
            "page": page,
            "box": (float(x), float(y), float(width), float(height)),
            "score": float(confidence),
        }

    def _suppress_enclosing_candidates(
        self,
        candidates: list[dict],
    ) -> list[dict]:
        """
        Elimina un candidato exterior cuando contiene al menos dos candidatos
        internos plausibles. Así se evita interpretar 2–4 hojas como una sola
        página gigante.

        Se conserva el candidato grande cuando no existen suficientes páginas
        internas, manteniendo la compatibilidad con fotografías de una hoja.
        """
        if len(candidates) < 3:
            return candidates

        kept: list[dict] = []

        for outer in candidates:
            outer_box = outer["box"]
            outer_area = max(
                float(outer_box[2] * outer_box[3]),
                1.0,
            )

            contained: list[dict] = []
            for inner in candidates:
                if inner is outer:
                    continue

                inner_box = inner["box"]
                inner_area = max(
                    float(inner_box[2] * inner_box[3]),
                    1.0,
                )

                # Una página interna debe ser claramente menor que el marco.
                if inner_area >= outer_area * 0.78:
                    continue

                containment = self._box_containment_ratio(
                    inner_box,
                    outer_box,
                )
                if containment >= 0.88:
                    contained.append(inner)

            if len(contained) >= 2:
                combined_inner_area = sum(
                    float(item["box"][2] * item["box"][3])
                    for item in contained
                )

                # Solo se elimina el marco si las páginas internas explican
                # una parte sustantiva de su superficie.
                if combined_inner_area >= outer_area * 0.38:
                    continue

            kept.append(outer)

        return kept

    @staticmethod
    def _box_containment_ratio(
        inner_box: tuple[float, float, float, float],
        outer_box: tuple[float, float, float, float],
    ) -> float:
        """Fracción del rectángulo interno contenida dentro del externo."""
        ix, iy, iw, ih = inner_box
        ox, oy, ow, oh = outer_box

        intersection_left = max(ix, ox)
        intersection_top = max(iy, oy)
        intersection_right = min(ix + iw, ox + ow)
        intersection_bottom = min(iy + ih, oy + oh)

        intersection_width = max(
            0.0,
            intersection_right - intersection_left,
        )
        intersection_height = max(
            0.0,
            intersection_bottom - intersection_top,
        )
        intersection_area = intersection_width * intersection_height
        inner_area = max(float(iw * ih), 1.0)

        return float(intersection_area / inner_area)

    def _remove_duplicates(self, candidates: list[dict]) -> list[dict]:
        ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
        selected: list[dict] = []
        for candidate in ordered:
            duplicate = False
            for accepted in selected:
                if self._bounding_box_iou(candidate["box"], accepted["box"]) >= self.duplicate_iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)
        return selected

    @staticmethod
    def _bounding_box_iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        intersection_left = max(ax, bx)
        intersection_top = max(ay, by)
        intersection_right = min(ax + aw, bx + bw)
        intersection_bottom = min(ay + ah, by + bh)
        intersection_width = max(0.0, intersection_right - intersection_left)
        intersection_height = max(0.0, intersection_bottom - intersection_top)
        intersection_area = intersection_width * intersection_height
        union_area = aw * ah + bw * bh - intersection_area
        return 0.0 if union_area <= 0 else float(intersection_area / union_area)

    @staticmethod
    def _create_perspective_crop(
        image: np.ndarray,
        ordered_corners: np.ndarray,
    ) -> np.ndarray | None:
        top_left, top_right, bottom_right, bottom_left = ordered_corners
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        height_left = np.linalg.norm(bottom_left - top_left)
        target_width = int(round(max(width_top, width_bottom)))
        target_height = int(round(max(height_right, height_left)))
        if target_width < 40 or target_height < 40:
            return None

        destination = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(
            ordered_corners.astype(np.float32),
            destination,
        )
        warped = cv2.warpPerspective(
            image,
            transform,
            (target_width, target_height),
        )
        return None if warped.size == 0 else warped

    @staticmethod
    def _create_crop(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray | None:
        image_height, image_width = image.shape[:2]
        x_start = max(0, int(x))
        y_start = max(0, int(y))
        x_end = min(image_width, int(x + width))
        y_end = min(image_height, int(y + height))
        if x_end <= x_start or y_end <= y_start:
            return None
        crop = image[y_start:y_end, x_start:x_end].copy()
        return None if crop.size == 0 else crop

    @staticmethod
    def _validate_image(image: Any) -> np.ndarray:
        if image is None:
            raise ValueError("La imagen no puede ser None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("La imagen debe ser un arreglo NumPy compatible con OpenCV.")
        if image.size == 0:
            raise ValueError("La imagen está vacía.")
        if image.ndim not in (2, 3):
            raise ValueError("La imagen tiene un formato no compatible.")

        normalized = image
        if normalized.dtype != np.uint8:
            normalized = np.nan_to_num(
                normalized,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )
            if float(np.min(normalized)) >= 0.0 and float(np.max(normalized)) <= 1.0:
                normalized = normalized * 255.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        if normalized.ndim == 2:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        elif normalized.shape[2] == 1:
            normalized = cv2.cvtColor(normalized[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif normalized.shape[2] == 4:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGRA2BGR)
        elif normalized.shape[2] != 3:
            raise ValueError("La imagen debe estar en formato gris, BGR o BGRA.")

        return np.ascontiguousarray(normalized)


def detect_pages(
    image: Any,
    minimum_area_ratio: float = 0.025,
    maximum_area_ratio: float = 0.97,
) -> List[DetectedPage]:
    detector = PageDetector(
        minimum_area_ratio=minimum_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
    )
    return detector.detect(image)


def draw_detected_pages(
    image: Any,
    pages: List[DetectedPage],
) -> np.ndarray:
    if image is None:
        raise ValueError("La imagen no puede ser None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("La imagen debe ser un arreglo NumPy.")

    preview = image.copy()
    if preview.ndim == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    for page in pages:
        polygon = np.array(page.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            preview,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=4,
        )
        cv2.putText(
            preview,
            f"Hoja {page.page_id}",
            (int(page.center_x), int(page.center_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return preview


