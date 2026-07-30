"""
Separador multipágina para Evalia Smart Capture.

Responsabilidades:
- recibir la imagen original y una página candidata grande;
- analizar si esa región contiene 2 o 4 hojas;
- buscar separaciones internas verticales, horizontales e inclinadas;
- crear páginas independientes compatibles con DetectedPage;
- conservar la página original cuando no existe evidencia suficiente;
- mantener separado este proceso de PageDetector y del OCR.

Integración prevista en CaptureAssistant:

    from .multi_page_splitter import MultiPageSplitter

    pages = self.page_detector.detect(normalized_image)
    pages = self.multi_page_splitter.split_pages(
        image=normalized_image,
        pages=pages,
    )
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import cv2
import numpy as np

from .models import DetectedPage


class MultiPageSplitter:
    """
    Divide una gran detección en varias hojas cuando encuentra separaciones
    internas plausibles.

    La primera versión está orientada a:
    - dos hojas lado a lado;
    - dos hojas una sobre otra;
    - cuatro hojas en una cuadrícula 2 × 2;
    - dos hojas separadas por una línea inclinada.

    Si la evidencia no es suficiente, devuelve la página original sin cambios.
    """

    def __init__(
        self,
        minimum_parent_area_ratio: float = 0.28,
        minimum_child_area_ratio: float = 0.055,
        minimum_child_side: int = 120,
        center_search_start: float = 0.28,
        center_search_end: float = 0.72,
        minimum_gap_width_ratio: float = 0.006,
        maximum_gap_width_ratio: float = 0.10,
        valley_quantile: float = 0.35,
        minimum_valley_depth: float = 0.10,
        minimum_split_confidence: float = 0.58,
        edge_margin_ratio: float = 0.015,
        hough_min_line_length_ratio: float = 0.36,
        hough_max_line_gap_ratio: float = 0.035,
        hough_center_tolerance_ratio: float = 0.24,
        hough_minimum_confidence: float = 0.56,
        hough_angle_tolerance_degrees: float = 28.0,
        max_output_pages: int = 8,
    ) -> None:
        if not 0 < minimum_parent_area_ratio <= 1:
            raise ValueError(
                "minimum_parent_area_ratio debe estar entre 0 y 1."
            )
        if not 0 < minimum_child_area_ratio <= 1:
            raise ValueError(
                "minimum_child_area_ratio debe estar entre 0 y 1."
            )
        if not 0 < center_search_start < center_search_end < 1:
            raise ValueError(
                "El intervalo central de búsqueda debe estar dentro de 0–1."
            )
        if minimum_gap_width_ratio <= 0:
            raise ValueError(
                "minimum_gap_width_ratio debe ser mayor que 0."
            )
        if maximum_gap_width_ratio <= minimum_gap_width_ratio:
            raise ValueError(
                "maximum_gap_width_ratio debe superar minimum_gap_width_ratio."
            )

        self.minimum_parent_area_ratio = float(
            minimum_parent_area_ratio
        )
        self.minimum_child_area_ratio = float(
            minimum_child_area_ratio
        )
        self.minimum_child_side = max(
            40,
            int(minimum_child_side),
        )
        self.center_search_start = float(center_search_start)
        self.center_search_end = float(center_search_end)
        self.minimum_gap_width_ratio = float(
            minimum_gap_width_ratio
        )
        self.maximum_gap_width_ratio = float(
            maximum_gap_width_ratio
        )
        self.valley_quantile = float(
            np.clip(valley_quantile, 0.05, 0.80)
        )
        self.minimum_valley_depth = float(
            np.clip(minimum_valley_depth, 0.0, 1.0)
        )
        self.minimum_split_confidence = float(
            np.clip(minimum_split_confidence, 0.0, 1.0)
        )
        self.edge_margin_ratio = float(
            np.clip(edge_margin_ratio, 0.0, 0.10)
        )
        self.hough_min_line_length_ratio = float(
            np.clip(hough_min_line_length_ratio, 0.10, 0.95)
        )
        self.hough_max_line_gap_ratio = float(
            np.clip(hough_max_line_gap_ratio, 0.001, 0.20)
        )
        self.hough_center_tolerance_ratio = float(
            np.clip(hough_center_tolerance_ratio, 0.05, 0.45)
        )
        self.hough_minimum_confidence = float(
            np.clip(hough_minimum_confidence, 0.0, 1.0)
        )
        self.hough_angle_tolerance_degrees = float(
            np.clip(hough_angle_tolerance_degrees, 5.0, 45.0)
        )
        self.max_output_pages = max(1, int(max_output_pages))

    def split_pages(
        self,
        image: np.ndarray,
        pages: Sequence[DetectedPage],
    ) -> List[DetectedPage]:
        """
        Examina cada página detectada y la divide solo cuando la evidencia
        multipágina supera los umbrales de seguridad.
        """
        normalized_image = self._validate_image(image)
        input_pages = list(pages or [])

        if not input_pages:
            return []

        image_area = float(
            normalized_image.shape[0] * normalized_image.shape[1]
        )

        output: List[DetectedPage] = []

        for page in input_pages:
            page_area = float(
                max(
                    getattr(page, "area", 0.0),
                    getattr(page, "width", 0.0)
                    * getattr(page, "height", 0.0),
                )
            )
            parent_ratio = page_area / max(image_area, 1.0)

            # Una detección pequeña es casi con seguridad una hoja individual.
            if parent_ratio < self.minimum_parent_area_ratio:
                output.append(page)
                continue

            split_result = self.split_if_needed(
                image=normalized_image,
                page=page,
            )

            output.extend(split_result)

            if len(output) >= self.max_output_pages:
                break

        output = output[: self.max_output_pages]
        output.sort(
            key=lambda item: (
                float(getattr(item, "center_y", 0.0)),
                float(getattr(item, "center_x", 0.0)),
            )
        )

        for index, page in enumerate(output, start=1):
            page.page_id = index

        return output

    def split_if_needed(
        self,
        image: np.ndarray,
        page: DetectedPage,
    ) -> List[DetectedPage]:
        """
        Intenta dividir una página candidata grande.

        Orden de prueba:
        1. cuadrícula 2 × 2;
        2. separación vertical;
        3. separación horizontal;
        4. separación inclinada mediante Hough;
        5. fallback seguro a la página original.
        """
        normalized_image = self._validate_image(image)
        region = self._extract_parent_region(
            image=normalized_image,
            page=page,
        )

        if region is None:
            return [page]

        crop, offset_x, offset_y = region
        if crop.shape[0] < self.minimum_child_side * 2:
            horizontal_possible = False
        else:
            horizontal_possible = True

        if crop.shape[1] < self.minimum_child_side * 2:
            vertical_possible = False
        else:
            vertical_possible = True

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        evidence = self._build_split_evidence(gray)

        vertical_split = (
            self._find_split_axis(
                profile=evidence["vertical_profile"],
                axis_length=crop.shape[1],
            )
            if vertical_possible
            else None
        )

        horizontal_split = (
            self._find_split_axis(
                profile=evidence["horizontal_profile"],
                axis_length=crop.shape[0],
            )
            if horizontal_possible
            else None
        )

        # Primero se prueba una cuadrícula 2 × 2.
        if (
            vertical_split is not None
            and horizontal_split is not None
        ):
            grid_confidence = min(
                vertical_split["confidence"],
                horizontal_split["confidence"],
            )
            children = self._build_grid_children(
                image=normalized_image,
                page=page,
                crop=crop,
                offset_x=offset_x,
                offset_y=offset_y,
                vertical_split=vertical_split,
                horizontal_split=horizontal_split,
                split_confidence=grid_confidence,
            )
            if self._children_are_valid(
                children=children,
                image=normalized_image,
                expected_count=4,
            ):
                return children

        # Dos hojas lado a lado.
        if vertical_split is not None:
            children = self._build_axis_children(
                image=normalized_image,
                page=page,
                crop=crop,
                offset_x=offset_x,
                offset_y=offset_y,
                split=vertical_split,
                axis="vertical",
            )
            if self._children_are_valid(
                children=children,
                image=normalized_image,
                expected_count=2,
            ):
                return children

        # Dos hojas apiladas.
        if horizontal_split is not None:
            children = self._build_axis_children(
                image=normalized_image,
                page=page,
                crop=crop,
                offset_x=offset_x,
                offset_y=offset_y,
                split=horizontal_split,
                axis="horizontal",
            )
            if self._children_are_valid(
                children=children,
                image=normalized_image,
                expected_count=2,
            ):
                return children

        # Fallback geométrico: separación central inclinada.
        hough_split = self._find_hough_separator(
            gray=gray,
        )
        if hough_split is not None:
            children = self._build_hough_children(
                image=normalized_image,
                page=page,
                crop=crop,
                offset_x=offset_x,
                offset_y=offset_y,
                split=hough_split,
            )
            if self._children_are_valid(
                children=children,
                image=normalized_image,
                expected_count=2,
            ):
                return children

        return [page]

    def _find_hough_separator(
        self,
        gray: np.ndarray,
    ) -> Optional[dict]:
        """
        Busca una línea larga, central y aproximadamente vertical u horizontal.

        Se usa como fallback cuando los perfiles por ejes no detectan el hueco
        porque la separación entre hojas aparece inclinada por perspectiva.
        """
        height, width = gray.shape[:2]
        if (
            height < self.minimum_child_side * 2
            or width < self.minimum_child_side * 2
        ):
            return None

        # Realza bordes largos y reduce ruido de escritura.
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 45, 135)

        # Cierre ligero para unir segmentos de la separación sin fusionar texto.
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        min_dimension = min(height, width)
        min_line_length = int(
            round(
                max(height, width)
                * self.hough_min_line_length_ratio
            )
        )
        max_line_gap = int(
            round(
                min_dimension
                * self.hough_max_line_gap_ratio
            )
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(45, int(min_dimension * 0.045)),
            minLineLength=max(80, min_line_length),
            maxLineGap=max(8, max_line_gap),
        )

        if lines is None:
            return None

        center_x = width / 2.0
        center_y = height / 2.0
        candidates = []

        for raw_line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in raw_line]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length <= 0:
                continue

            angle = float(
                np.degrees(np.arctan2(dy, dx))
            )
            normalized_angle = abs(angle) % 180.0

            # Distancia angular a vertical y horizontal.
            distance_to_horizontal = min(
                normalized_angle,
                abs(180.0 - normalized_angle),
            )
            distance_to_vertical = abs(
                90.0 - normalized_angle
            )

            orientation: Optional[str] = None
            angle_distance = 999.0

            if (
                distance_to_vertical
                <= self.hough_angle_tolerance_degrees
            ):
                orientation = "vertical"
                angle_distance = distance_to_vertical

            if (
                distance_to_horizontal
                <= self.hough_angle_tolerance_degrees
                and distance_to_horizontal < angle_distance
            ):
                orientation = "horizontal"
                angle_distance = distance_to_horizontal

            if orientation is None:
                continue

            midpoint_x = (x1 + x2) / 2.0
            midpoint_y = (y1 + y2) / 2.0

            if orientation == "vertical":
                center_distance = abs(
                    midpoint_x - center_x
                ) / max(width, 1.0)
                span_ratio = abs(dy) / max(height, 1.0)
            else:
                center_distance = abs(
                    midpoint_y - center_y
                ) / max(height, 1.0)
                span_ratio = abs(dx) / max(width, 1.0)

            if (
                center_distance
                > self.hough_center_tolerance_ratio
            ):
                continue

            centrality = 1.0 - min(
                1.0,
                center_distance
                / max(
                    self.hough_center_tolerance_ratio,
                    1e-6,
                ),
            )
            length_score = min(
                1.0,
                length / max(
                    min_line_length,
                    1.0,
                ),
            )
            span_score = min(
                1.0,
                span_ratio / 0.58,
            )
            angle_score = 1.0 - min(
                1.0,
                angle_distance
                / max(
                    self.hough_angle_tolerance_degrees,
                    1e-6,
                ),
            )

            confidence = (
                0.34 * length_score
                + 0.28 * span_score
                + 0.23 * centrality
                + 0.15 * angle_score
            )

            candidates.append(
                {
                    "orientation": orientation,
                    "line": (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ),
                    "midpoint_x": float(midpoint_x),
                    "midpoint_y": float(midpoint_y),
                    "angle": float(angle),
                    "length": float(length),
                    "confidence": float(
                        np.clip(confidence, 0.0, 1.0)
                    ),
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda item: item["confidence"],
        )

        if (
            best["confidence"]
            < self.hough_minimum_confidence
        ):
            return None

        return best

    def _build_hough_children(
        self,
        image: np.ndarray,
        page: DetectedPage,
        crop: np.ndarray,
        offset_x: int,
        offset_y: int,
        split: dict,
    ) -> List[DetectedPage]:
        """
        Convierte una línea inclinada en dos regiones poligonales.

        Para mantener compatibilidad con DetectedPage, cada lado se convierte
        finalmente en su rectángulo envolvente seguro dentro de la región padre.
        """
        height, width = crop.shape[:2]
        x1, y1, x2, y2 = split["line"]

        # Normaliza la dirección para que la línea atraviese la región completa.
        if split["orientation"] == "vertical":
            if abs(y2 - y1) < 1e-6:
                return []

            slope_x_per_y = (x2 - x1) / (y2 - y1)
            x_top = x1 + (0.0 - y1) * slope_x_per_y
            x_bottom = x1 + (
                (height - 1.0) - y1
            ) * slope_x_per_y

            split_margin = max(
                3,
                int(round(width * 0.008)),
            )

            left_polygon = np.array(
                [
                    [0.0, 0.0],
                    [
                        np.clip(
                            x_top - split_margin,
                            0,
                            width - 1,
                        ),
                        0.0,
                    ],
                    [
                        np.clip(
                            x_bottom - split_margin,
                            0,
                            width - 1,
                        ),
                        float(height - 1),
                    ],
                    [0.0, float(height - 1)],
                ],
                dtype=np.float32,
            )
            right_polygon = np.array(
                [
                    [
                        np.clip(
                            x_top + split_margin,
                            0,
                            width - 1,
                        ),
                        0.0,
                    ],
                    [float(width - 1), 0.0],
                    [
                        float(width - 1),
                        float(height - 1),
                    ],
                    [
                        np.clip(
                            x_bottom + split_margin,
                            0,
                            width - 1,
                        ),
                        float(height - 1),
                    ],
                ],
                dtype=np.float32,
            )
            polygons = [
                left_polygon,
                right_polygon,
            ]
        else:
            if abs(x2 - x1) < 1e-6:
                return []

            slope_y_per_x = (y2 - y1) / (x2 - x1)
            y_left = y1 + (0.0 - x1) * slope_y_per_x
            y_right = y1 + (
                (width - 1.0) - x1
            ) * slope_y_per_x

            split_margin = max(
                3,
                int(round(height * 0.008)),
            )

            top_polygon = np.array(
                [
                    [0.0, 0.0],
                    [float(width - 1), 0.0],
                    [
                        float(width - 1),
                        np.clip(
                            y_right - split_margin,
                            0,
                            height - 1,
                        ),
                    ],
                    [
                        0.0,
                        np.clip(
                            y_left - split_margin,
                            0,
                            height - 1,
                        ),
                    ],
                ],
                dtype=np.float32,
            )
            bottom_polygon = np.array(
                [
                    [
                        0.0,
                        np.clip(
                            y_left + split_margin,
                            0,
                            height - 1,
                        ),
                    ],
                    [
                        float(width - 1),
                        np.clip(
                            y_right + split_margin,
                            0,
                            height - 1,
                        ),
                    ],
                    [
                        float(width - 1),
                        float(height - 1),
                    ],
                    [0.0, float(height - 1)],
                ],
                dtype=np.float32,
            )
            polygons = [
                top_polygon,
                bottom_polygon,
            ]

        pages: List[DetectedPage] = []

        for child_index, polygon in enumerate(
            polygons,
            start=1,
        ):
            x, y, child_width, child_height = cv2.boundingRect(
                polygon.astype(np.int32)
            )

            if (
                child_width < self.minimum_child_side
                or child_height < self.minimum_child_side
            ):
                continue

            local_crop = self._safe_crop(
                image=crop,
                x=x,
                y=y,
                width=child_width,
                height=child_height,
            )
            if local_crop is None:
                continue

            global_x = int(offset_x + x)
            global_y = int(offset_y + y)

            # Esquinas aproximadas del polígono real en coordenadas globales.
            global_polygon = polygon.copy()
            global_polygon[:, 0] += float(offset_x)
            global_polygon[:, 1] += float(offset_y)

            parent_confidence = float(
                getattr(page, "confidence", 0.70) or 0.70
            )
            child_confidence = float(
                np.clip(
                    0.55 * parent_confidence
                    + 0.45 * float(
                        split["confidence"]
                    ),
                    0.0,
                    1.0,
                )
            )

            metadata = dict(
                getattr(page, "metadata", {}) or {}
            )
            metadata.update(
                {
                    "split_from_parent": True,
                    "parent_page_id": getattr(
                        page,
                        "page_id",
                        None,
                    ),
                    "split_method": (
                        "hough_inclined_"
                        + str(split["orientation"])
                    ),
                    "split_confidence": float(
                        split["confidence"]
                    ),
                    "split_child_index": child_index,
                    "separator_angle": float(
                        split["angle"]
                    ),
                    "separator_line": [
                        float(v)
                        for v in split["line"]
                    ],
                    "perspective_corrected": False,
                }
            )

            pages.append(
                DetectedPage(
                    page_id=0,
                    corners=[
                        (
                            float(point[0]),
                            float(point[1]),
                        )
                        for point in global_polygon
                    ],
                    bounding_box=(
                        float(global_x),
                        float(global_y),
                        float(child_width),
                        float(child_height),
                    ),
                    width=float(child_width),
                    height=float(child_height),
                    center_x=float(
                        global_x + child_width / 2.0
                    ),
                    center_y=float(
                        global_y + child_height / 2.0
                    ),
                    area=float(
                        cv2.contourArea(
                            polygon.astype(np.float32)
                        )
                    ),
                    confidence=child_confidence,
                    page_number=None,
                    cropped_image=local_crop,
                    metadata=metadata,
                )
            )

        return pages

    def _build_split_evidence(
        self,
        gray: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Combina tres señales:

        - intensidad: el espacio entre hojas suele diferir del papel;
        - densidad de bordes: una separación tiene menos texto y trazos;
        - variación local: el hueco suele ser más uniforme.

        El perfil final se normaliza entre 0 y 1; valores bajos son mejores
        candidatos a separación.
        """
        gray_float = gray.astype(np.float32) / 255.0

        edges = cv2.Canny(gray, 50, 150)
        edge_density_vertical = np.mean(
            edges.astype(np.float32) / 255.0,
            axis=0,
        )
        edge_density_horizontal = np.mean(
            edges.astype(np.float32) / 255.0,
            axis=1,
        )

        local_mean = cv2.blur(gray_float, (15, 15))
        local_sq_mean = cv2.blur(
            gray_float * gray_float,
            (15, 15),
        )
        local_variance = np.maximum(
            local_sq_mean - local_mean * local_mean,
            0.0,
        )

        variance_vertical = np.mean(local_variance, axis=0)
        variance_horizontal = np.mean(local_variance, axis=1)

        # Distancia respecto de un blanco típico de papel. No se presupone
        # fondo blanco: se usa como una señal secundaria, no exclusiva.
        paper_distance = np.abs(gray_float - 0.88)
        intensity_vertical = np.mean(paper_distance, axis=0)
        intensity_horizontal = np.mean(paper_distance, axis=1)

        vertical_profile = (
            0.48 * self._normalize_profile(edge_density_vertical)
            + 0.34 * self._normalize_profile(variance_vertical)
            + 0.18 * self._normalize_profile(intensity_vertical)
        )
        horizontal_profile = (
            0.48 * self._normalize_profile(edge_density_horizontal)
            + 0.34 * self._normalize_profile(variance_horizontal)
            + 0.18 * self._normalize_profile(intensity_horizontal)
        )

        vertical_profile = self._smooth_profile(
            vertical_profile,
            window=max(5, int(gray.shape[1] * 0.012)),
        )
        horizontal_profile = self._smooth_profile(
            horizontal_profile,
            window=max(5, int(gray.shape[0] * 0.012)),
        )

        return {
            "vertical_profile": vertical_profile,
            "horizontal_profile": horizontal_profile,
        }

    def _find_split_axis(
        self,
        profile: np.ndarray,
        axis_length: int,
    ) -> Optional[dict]:
        """
        Busca un valle ancho y suficientemente profundo en la zona central.
        """
        profile = np.asarray(
            profile,
            dtype=np.float32,
        ).reshape(-1)

        if profile.size < self.minimum_child_side * 2:
            return None

        search_start = int(
            round(profile.size * self.center_search_start)
        )
        search_end = int(
            round(profile.size * self.center_search_end)
        )

        if search_end <= search_start:
            return None

        central = profile[search_start:search_end]
        if central.size == 0:
            return None

        threshold = float(
            np.quantile(central, self.valley_quantile)
        )
        low_mask = central <= threshold

        runs = self._true_runs(low_mask)
        if not runs:
            return None

        min_gap = max(
            2,
            int(round(axis_length * self.minimum_gap_width_ratio)),
        )
        max_gap = max(
            min_gap,
            int(round(axis_length * self.maximum_gap_width_ratio)),
        )

        global_reference = float(np.median(profile))
        candidates = []

        for run_start, run_end in runs:
            width = run_end - run_start
            if width < min_gap or width > max_gap:
                continue

            absolute_start = search_start + run_start
            absolute_end = search_start + run_end
            center = (absolute_start + absolute_end) / 2.0

            left_fraction = center / max(profile.size, 1)
            right_fraction = 1.0 - left_fraction

            # Se exige espacio suficiente para construir dos hojas.
            minimum_fraction = (
                self.minimum_child_side / max(axis_length, 1)
            )
            if (
                left_fraction < minimum_fraction
                or right_fraction < minimum_fraction
            ):
                continue

            valley_value = float(
                np.mean(profile[absolute_start:absolute_end])
            )
            valley_depth = (
                global_reference - valley_value
            ) / max(global_reference, 1e-6)

            if valley_depth < self.minimum_valley_depth:
                continue

            centrality = 1.0 - min(
                1.0,
                abs((center / profile.size) - 0.5) / 0.22,
            )
            width_score = min(
                1.0,
                width / max(min_gap * 2.0, 1.0),
            )
            depth_score = min(
                1.0,
                valley_depth / max(
                    self.minimum_valley_depth * 2.2,
                    1e-6,
                ),
            )

            confidence = (
                0.52 * depth_score
                + 0.28 * centrality
                + 0.20 * width_score
            )

            candidates.append(
                {
                    "start": int(absolute_start),
                    "end": int(absolute_end),
                    "center": int(round(center)),
                    "width": int(width),
                    "depth": float(valley_depth),
                    "confidence": float(
                        np.clip(confidence, 0.0, 1.0)
                    ),
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda item: item["confidence"],
        )

        if best["confidence"] < self.minimum_split_confidence:
            return None

        return best

    def _build_axis_children(
        self,
        image: np.ndarray,
        page: DetectedPage,
        crop: np.ndarray,
        offset_x: int,
        offset_y: int,
        split: dict,
        axis: str,
    ) -> List[DetectedPage]:
        margin_x = max(
            2,
            int(round(crop.shape[1] * self.edge_margin_ratio)),
        )
        margin_y = max(
            2,
            int(round(crop.shape[0] * self.edge_margin_ratio)),
        )

        if axis == "vertical":
            boxes = [
                (
                    margin_x,
                    margin_y,
                    split["start"] - margin_x,
                    crop.shape[0] - 2 * margin_y,
                ),
                (
                    split["end"],
                    margin_y,
                    crop.shape[1] - split["end"] - margin_x,
                    crop.shape[0] - 2 * margin_y,
                ),
            ]
        elif axis == "horizontal":
            boxes = [
                (
                    margin_x,
                    margin_y,
                    crop.shape[1] - 2 * margin_x,
                    split["start"] - margin_y,
                ),
                (
                    margin_x,
                    split["end"],
                    crop.shape[1] - 2 * margin_x,
                    crop.shape[0] - split["end"] - margin_y,
                ),
            ]
        else:
            raise ValueError("axis debe ser 'vertical' u 'horizontal'.")

        return self._pages_from_local_boxes(
            image=image,
            parent=page,
            local_boxes=boxes,
            offset_x=offset_x,
            offset_y=offset_y,
            split_method=f"internal_{axis}_valley",
            split_confidence=float(split["confidence"]),
        )

    def _build_grid_children(
        self,
        image: np.ndarray,
        page: DetectedPage,
        crop: np.ndarray,
        offset_x: int,
        offset_y: int,
        vertical_split: dict,
        horizontal_split: dict,
        split_confidence: float,
    ) -> List[DetectedPage]:
        margin_x = max(
            2,
            int(round(crop.shape[1] * self.edge_margin_ratio)),
        )
        margin_y = max(
            2,
            int(round(crop.shape[0] * self.edge_margin_ratio)),
        )

        left_x = margin_x
        left_width = vertical_split["start"] - margin_x
        right_x = vertical_split["end"]
        right_width = (
            crop.shape[1]
            - vertical_split["end"]
            - margin_x
        )

        top_y = margin_y
        top_height = horizontal_split["start"] - margin_y
        bottom_y = horizontal_split["end"]
        bottom_height = (
            crop.shape[0]
            - horizontal_split["end"]
            - margin_y
        )

        local_boxes = [
            (left_x, top_y, left_width, top_height),
            (right_x, top_y, right_width, top_height),
            (left_x, bottom_y, left_width, bottom_height),
            (right_x, bottom_y, right_width, bottom_height),
        ]

        return self._pages_from_local_boxes(
            image=image,
            parent=page,
            local_boxes=local_boxes,
            offset_x=offset_x,
            offset_y=offset_y,
            split_method="internal_grid_2x2",
            split_confidence=float(split_confidence),
        )

    def _pages_from_local_boxes(
        self,
        image: np.ndarray,
        parent: DetectedPage,
        local_boxes: Sequence[tuple[int, int, int, int]],
        offset_x: int,
        offset_y: int,
        split_method: str,
        split_confidence: float,
    ) -> List[DetectedPage]:
        pages: List[DetectedPage] = []

        for local_index, (x, y, width, height) in enumerate(
            local_boxes,
            start=1,
        ):
            if (
                width < self.minimum_child_side
                or height < self.minimum_child_side
            ):
                continue

            global_x = int(offset_x + x)
            global_y = int(offset_y + y)
            global_width = int(width)
            global_height = int(height)

            crop = self._safe_crop(
                image=image,
                x=global_x,
                y=global_y,
                width=global_width,
                height=global_height,
            )
            if crop is None:
                continue

            corners = [
                (float(global_x), float(global_y)),
                (
                    float(global_x + global_width),
                    float(global_y),
                ),
                (
                    float(global_x + global_width),
                    float(global_y + global_height),
                ),
                (
                    float(global_x),
                    float(global_y + global_height),
                ),
            ]

            parent_confidence = float(
                getattr(parent, "confidence", 0.70) or 0.70
            )
            child_confidence = float(
                np.clip(
                    0.58 * parent_confidence
                    + 0.42 * split_confidence,
                    0.0,
                    1.0,
                )
            )

            metadata = dict(
                getattr(parent, "metadata", {}) or {}
            )
            metadata.update(
                {
                    "split_from_parent": True,
                    "parent_page_id": getattr(
                        parent,
                        "page_id",
                        None,
                    ),
                    "split_method": split_method,
                    "split_confidence": float(
                        split_confidence
                    ),
                    "split_child_index": local_index,
                    "perspective_corrected": False,
                }
            )

            pages.append(
                DetectedPage(
                    page_id=0,
                    corners=corners,
                    bounding_box=(
                        float(global_x),
                        float(global_y),
                        float(global_width),
                        float(global_height),
                    ),
                    width=float(global_width),
                    height=float(global_height),
                    center_x=float(
                        global_x + global_width / 2.0
                    ),
                    center_y=float(
                        global_y + global_height / 2.0
                    ),
                    area=float(global_width * global_height),
                    confidence=child_confidence,
                    page_number=None,
                    cropped_image=crop,
                    metadata=metadata,
                )
            )

        return pages

    def _children_are_valid(
        self,
        children: Sequence[DetectedPage],
        image: np.ndarray,
        expected_count: int,
    ) -> bool:
        if len(children) != expected_count:
            return False

        image_area = float(
            image.shape[0] * image.shape[1]
        )
        child_areas = []

        for child in children:
            width = float(getattr(child, "width", 0.0))
            height = float(getattr(child, "height", 0.0))
            area = max(
                float(getattr(child, "area", 0.0)),
                width * height,
            )

            if (
                width < self.minimum_child_side
                or height < self.minimum_child_side
            ):
                return False

            if area / max(image_area, 1.0) < (
                self.minimum_child_area_ratio
            ):
                return False

            shorter = max(min(width, height), 1.0)
            longer = max(width, height)
            ratio = shorter / longer

            # Tolerancia amplia para hojas fotografiadas en perspectiva.
            if ratio < 0.30:
                return False

            child_areas.append(area)

        # Evita divisiones absurdamente desbalanceadas.
        if child_areas:
            smallest = min(child_areas)
            largest = max(child_areas)
            if smallest / max(largest, 1.0) < 0.42:
                return False

        return True

    def _extract_parent_region(
        self,
        image: np.ndarray,
        page: DetectedPage,
    ) -> Optional[tuple[np.ndarray, int, int]]:
        box = getattr(page, "bounding_box", None)
        if box is None or len(box) != 4:
            return None

        x, y, width, height = box
        x = max(0, int(round(x)))
        y = max(0, int(round(y)))
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))

        x_end = min(image.shape[1], x + width)
        y_end = min(image.shape[0], y + height)

        if x_end <= x or y_end <= y:
            return None

        crop = image[y:y_end, x:x_end].copy()
        if crop.size == 0:
            return None

        return crop, x, y

    @staticmethod
    def _normalize_profile(profile: np.ndarray) -> np.ndarray:
        values = np.asarray(
            profile,
            dtype=np.float32,
        ).reshape(-1)

        if values.size == 0:
            return values

        low = float(np.percentile(values, 5))
        high = float(np.percentile(values, 95))

        if high <= low + 1e-8:
            return np.zeros_like(values)

        normalized = (values - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def _smooth_profile(
        profile: np.ndarray,
        window: int,
    ) -> np.ndarray:
        values = np.asarray(
            profile,
            dtype=np.float32,
        ).reshape(-1)

        if values.size == 0:
            return values

        window = max(3, int(window))
        if window % 2 == 0:
            window += 1
        window = min(
            window,
            values.size if values.size % 2 == 1 else values.size - 1,
        )
        if window < 3:
            return values

        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(values, kernel, mode="same")

    @staticmethod
    def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        runs: list[tuple[int, int]] = []

        start: Optional[int] = None
        for index, value in enumerate(mask):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None

        if start is not None:
            runs.append((start, len(mask)))

        return runs

    @staticmethod
    def _safe_crop(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        x_start = max(0, int(x))
        y_start = max(0, int(y))
        x_end = min(
            image.shape[1],
            int(x + width),
        )
        y_end = min(
            image.shape[0],
            int(y + height),
        )

        if x_end <= x_start or y_end <= y_start:
            return None

        crop = image[
            y_start:y_end,
            x_start:x_end,
        ].copy()

        return None if crop.size == 0 else crop

    @staticmethod
    def _validate_image(image: Any) -> np.ndarray:
        if image is None:
            raise ValueError("La imagen no puede ser None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "La imagen debe ser un arreglo NumPy compatible con OpenCV."
            )

        if image.size == 0:
            raise ValueError("La imagen está vacía.")

        if image.ndim not in (2, 3):
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
                "La imagen debe estar en formato gris, BGR o BGRA."
            )

        return np.ascontiguousarray(normalized)


def split_detected_pages(
    image: np.ndarray,
    pages: Sequence[DetectedPage],
) -> List[DetectedPage]:
    """Interfaz funcional de conveniencia."""
    splitter = MultiPageSplitter()
    return splitter.split_pages(
        image=image,
        pages=pages,
    )
