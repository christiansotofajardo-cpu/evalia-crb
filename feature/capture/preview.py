"""
preview.py
==========

Genera una vista previa visual de las páginas detectadas por
Smart Capture antes de enviarlas al OCR.

La vista previa permite al profesor confirmar:

- cuántas hojas fueron detectadas;
- cómo fueron agrupadas por estudiante;
- cuál corresponde a la página 1 y página 2;
- si existe algún grupo incompleto;
- si la detección requiere corrección manual.

Este módulo no ejecuta OCR ni modifica la imagen original.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import DetectedPage, StudentPageGroup


class PreviewGenerator:
    """
    Genera una imagen anotada con la organización de las páginas.

    Parameters
    ----------
    line_thickness:
        Grosor de los contornos dibujados.

    font_scale:
        Tamaño base del texto.

    show_page_id:
        Indica si se muestra el identificador interno de cada página.

    add_summary:
        Indica si se agrega un resumen superior con el número de
        estudiantes y páginas detectadas.
    """

    def __init__(
        self,
        line_thickness: int = 4,
        font_scale: float = 0.75,
        show_page_id: bool = False,
        add_summary: bool = True,
    ) -> None:

        if line_thickness < 1:
            raise ValueError(
                "line_thickness debe ser igual o mayor que 1."
            )

        if font_scale <= 0:
            raise ValueError(
                "font_scale debe ser mayor que 0."
            )

        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.show_page_id = show_page_id
        self.add_summary = add_summary

        # OpenCV utiliza colores BGR.
        self.complete_color = (40, 170, 40)
        self.incomplete_color = (0, 165, 255)
        self.page_color = (255, 120, 20)
        self.text_color = (255, 255, 255)
        self.text_background = (20, 20, 20)
        self.summary_background = (245, 245, 245)
        self.summary_text_color = (30, 30, 30)

    def render(
        self,
        image: np.ndarray,
        groups: Sequence[StudentPageGroup],
    ) -> np.ndarray:
        """
        Dibuja la vista previa sobre una copia de la fotografía.

        Parameters
        ----------
        image:
            Imagen original en formato OpenCV.

        groups:
            Grupos producidos por PageOrganizer.

        Returns
        -------
        numpy.ndarray
            Imagen anotada en formato BGR.
        """

        self._validate_image(image)

        preview = self._prepare_canvas(image)

        total_pages = 0
        incomplete_groups = 0

        for group_index, group in enumerate(
            groups,
            start=1,
        ):
            pages = self._get_group_pages(group)
            complete = self._is_group_complete(group)

            total_pages += len(pages)

            if not complete:
                incomplete_groups += 1

            group_number = self._get_group_number(
                group=group,
                fallback=group_index,
            )

            group_color = (
                self.complete_color
                if complete
                else self.incomplete_color
            )

            for page_index, page in enumerate(
                pages,
                start=1,
            ):
                page_number = self._get_page_number(
                    page=page,
                    fallback=page_index,
                )

                label = self._build_page_label(
                    group_number=group_number,
                    page_number=page_number,
                    page=page,
                    complete=complete,
                )

                self._draw_page(
                    canvas=preview,
                    page=page,
                    label=label,
                    color=group_color,
                )

        if self.add_summary:
            preview = self._add_summary_bar(
                image=preview,
                student_count=len(groups),
                page_count=total_pages,
                incomplete_count=incomplete_groups,
            )

        return preview

    def render_pages(
        self,
        image: np.ndarray,
        pages: Sequence[DetectedPage],
    ) -> np.ndarray:
        """
        Dibuja únicamente las páginas detectadas, sin agruparlas.

        Este método resulta útil durante las pruebas iniciales del
        detector de páginas.
        """

        self._validate_image(image)

        preview = self._prepare_canvas(image)

        for index, page in enumerate(
            pages,
            start=1,
        ):
            label = f"Hoja detectada {index}"

            self._draw_page(
                canvas=preview,
                page=page,
                label=label,
                color=self.page_color,
            )

        if self.add_summary:
            preview = self._add_summary_bar(
                image=preview,
                student_count=0,
                page_count=len(pages),
                incomplete_count=0,
                detector_only=True,
            )

        return preview

    def encode_jpeg(
        self,
        image: np.ndarray,
        quality: int = 88,
    ) -> bytes:
        """
        Codifica una vista previa como JPEG.

        Puede utilizarse directamente en la respuesta de FastAPI.
        """

        self._validate_image(image)

        quality = max(
            40,
            min(int(quality), 100),
        )

        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                quality,
            ],
        )

        if not success:
            raise ValueError(
                "No fue posible codificar la vista previa como JPEG."
            )

        return encoded.tobytes()

    def encode_png(
        self,
        image: np.ndarray,
    ) -> bytes:
        """
        Codifica una vista previa como PNG.
        """

        self._validate_image(image)

        success, encoded = cv2.imencode(
            ".png",
            image,
        )

        if not success:
            raise ValueError(
                "No fue posible codificar la vista previa como PNG."
            )

        return encoded.tobytes()

    def resize_for_web(
        self,
        image: np.ndarray,
        max_width: int = 1600,
        max_height: int = 1600,
    ) -> np.ndarray:
        """
        Reduce la imagen para mostrarla en la interfaz web.

        La proporción original se mantiene.
        """

        self._validate_image(image)

        if max_width < 1 or max_height < 1:
            raise ValueError(
                "Las dimensiones máximas deben ser mayores que 0."
            )

        height, width = image.shape[:2]

        scale = min(
            max_width / width,
            max_height / height,
            1.0,
        )

        if scale >= 1.0:
            return image.copy()

        resized_width = max(
            1,
            int(round(width * scale)),
        )

        resized_height = max(
            1,
            int(round(height * scale)),
        )

        return cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    def _draw_page(
        self,
        canvas: np.ndarray,
        page: DetectedPage,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """
        Dibuja el contorno y la etiqueta de una página.
        """

        corners = self._get_corners(page)

        if corners is not None and len(corners) >= 4:
            contour = np.array(
                corners,
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            cv2.polylines(
                canvas,
                [contour],
                isClosed=True,
                color=color,
                thickness=self.line_thickness,
                lineType=cv2.LINE_AA,
            )

            anchor_x = int(
                min(point[0] for point in corners)
            )

            anchor_y = int(
                min(point[1] for point in corners)
            )

        else:
            x, y, width, height = self._get_bounding_box(page)

            cv2.rectangle(
                canvas,
                (int(x), int(y)),
                (
                    int(x + width),
                    int(y + height),
                ),
                color,
                self.line_thickness,
                lineType=cv2.LINE_AA,
            )

            anchor_x = int(x)
            anchor_y = int(y)

        self._draw_label(
            canvas=canvas,
            text=label,
            anchor=(
                anchor_x,
                anchor_y,
            ),
            color=color,
        )

    def _draw_label(
        self,
        canvas: np.ndarray,
        text: str,
        anchor: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """
        Dibuja una etiqueta legible encima de una página.
        """

        image_height, image_width = canvas.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(
            1,
            int(round(self.font_scale * 2)),
        )

        text_size, baseline = cv2.getTextSize(
            text,
            font,
            self.font_scale,
            thickness,
        )

        text_width, text_height = text_size

        padding_x = 9
        padding_y = 7

        box_width = text_width + padding_x * 2
        box_height = (
            text_height
            + baseline
            + padding_y * 2
        )

        anchor_x, anchor_y = anchor

        box_x = max(
            0,
            min(
                anchor_x,
                image_width - box_width,
            ),
        )

        box_y = anchor_y - box_height - 6

        if box_y < 0:
            box_y = min(
                image_height - box_height,
                anchor_y + 6,
            )

        box_y = max(
            0,
            box_y,
        )

        overlay = canvas.copy()

        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (
                box_x + box_width,
                box_y + box_height,
            ),
            self.text_background,
            thickness=-1,
        )

        alpha = 0.78

        cv2.addWeighted(
            overlay,
            alpha,
            canvas,
            1.0 - alpha,
            0,
            canvas,
        )

        cv2.rectangle(
            canvas,
            (box_x, box_y),
            (
                box_x + box_width,
                box_y + box_height,
            ),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        text_x = box_x + padding_x
        text_y = (
            box_y
            + padding_y
            + text_height
        )

        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            font,
            self.font_scale,
            self.text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    def _add_summary_bar(
        self,
        image: np.ndarray,
        student_count: int,
        page_count: int,
        incomplete_count: int,
        detector_only: bool = False,
    ) -> np.ndarray:
        """
        Agrega una barra informativa en la parte superior.
        """

        image_height, image_width = image.shape[:2]

        bar_height = max(
            72,
            int(round(image_height * 0.07)),
        )

        output = np.full(
            (
                image_height + bar_height,
                image_width,
                3,
            ),
            self.summary_background,
            dtype=np.uint8,
        )

        output[
            bar_height:
            bar_height + image_height,
            :
        ] = image

        if detector_only:
            summary = (
                f"Evalia Smart Capture | "
                f"{page_count} hojas detectadas"
            )

        else:
            summary = (
                f"Evalia Smart Capture | "
                f"{student_count} estudiantes | "
                f"{page_count} hojas"
            )

            if incomplete_count > 0:
                summary += (
                    f" | {incomplete_count} grupo(s) incompleto(s)"
                )
            else:
                summary += " | Organización completa"

        font = cv2.FONT_HERSHEY_SIMPLEX

        dynamic_scale = min(
            1.0,
            max(
                0.52,
                image_width / 1700.0,
            ),
        )

        thickness = max(
            1,
            int(round(dynamic_scale * 2)),
        )

        text_size, _ = cv2.getTextSize(
            summary,
            font,
            dynamic_scale,
            thickness,
        )

        text_width, text_height = text_size

        text_x = max(
            18,
            int(
                (image_width - text_width) / 2
            ),
        )

        text_y = int(
            (bar_height + text_height) / 2
        )

        cv2.putText(
            output,
            summary,
            (text_x, text_y),
            font,
            dynamic_scale,
            self.summary_text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        return output

    def _build_page_label(
        self,
        group_number: int,
        page_number: int,
        page: DetectedPage,
        complete: bool,
    ) -> str:
        """
        Construye la etiqueta que se mostrará sobre la hoja.
        """

        label = (
            f"Estudiante {group_number} - "
            f"Pagina {page_number}"
        )

        if not complete:
            label += " - INCOMPLETO"

        if self.show_page_id:
            page_id = getattr(
                page,
                "page_id",
                None,
            )

            if page_id is not None:
                label += f" [{page_id}]"

        return label

    @staticmethod
    def _prepare_canvas(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normaliza la imagen a formato BGR y devuelve una copia.
        """

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        if image.shape[2] == 1:
            return cv2.cvtColor(
                image[:, :, 0],
                cv2.COLOR_GRAY2BGR,
            )

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR,
            )

        return image.copy()

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> None:
        """
        Verifica que la entrada sea una imagen compatible.
        """

        if image is None:
            raise ValueError(
                "No se recibió una imagen para generar la vista previa."
            )

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
                "La imagen debe estar en escala de grises, BGR o BGRA."
            )

        if (
            image.ndim == 3
            and image.shape[2] not in (1, 3, 4)
        ):
            raise ValueError(
                "La imagen contiene un número de canales no compatible."
            )

    @staticmethod
    def _get_group_pages(
        group: StudentPageGroup,
    ) -> List[DetectedPage]:
        """
        Extrae las páginas de un grupo.
        """

        pages = getattr(
            group,
            "pages",
            [],
        )

        if pages is None:
            return []

        return list(pages)

    @staticmethod
    def _is_group_complete(
        group: StudentPageGroup,
    ) -> bool:
        """
        Determina si el grupo contiene todas sus páginas.
        """

        return bool(
            getattr(
                group,
                "complete",
                False,
            )
        )

    @staticmethod
    def _get_group_number(
        group: StudentPageGroup,
        fallback: int,
    ) -> int:
        """
        Obtiene el número de estudiante.
        """

        value = getattr(
            group,
            "student_number",
            fallback,
        )

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return fallback

    @staticmethod
    def _get_page_number(
        page: DetectedPage,
        fallback: int,
    ) -> int:
        """
        Obtiene el número de página dentro del grupo.
        """

        value = getattr(
            page,
            "page_number",
            fallback,
        )

        if value is None:
            return fallback

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return fallback

    @staticmethod
    def _get_corners(
        page: DetectedPage,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Obtiene los cuatro vértices de una página.

        Admite los atributos:

        - corners
        - contour
        - points
        - polygon
        """

        candidates = (
            "corners",
            "contour",
            "points",
            "polygon",
        )

        raw_points: Any = None

        for attribute in candidates:
            raw_points = getattr(
                page,
                attribute,
                None,
            )

            if raw_points is not None:
                break

        if raw_points is None:
            return None

        points_array = np.asarray(raw_points)

        if points_array.size < 8:
            return None

        try:
            points_array = points_array.reshape(
                (-1, 2)
            )

        except ValueError:
            return None

        normalized: List[Tuple[int, int]] = []

        for point in points_array:
            try:
                normalized.append(
                    (
                        int(round(float(point[0]))),
                        int(round(float(point[1]))),
                    )
                )

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

        if len(normalized) < 4:
            return None

        return normalized[:4]

    @staticmethod
    def _get_bounding_box(
        page: DetectedPage,
    ) -> Tuple[float, float, float, float]:
        """
        Obtiene la caja delimitadora de una página.
        """

        bounding_box = getattr(
            page,
            "bounding_box",
            None,
        )

        if bounding_box is None:
            bounding_box = getattr(
                page,
                "bbox",
                None,
            )

        if bounding_box is not None:
            try:
                x, y, width, height = bounding_box

                return (
                    float(x),
                    float(y),
                    float(width),
                    float(height),
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        x = getattr(page, "x", None)
        y = getattr(page, "y", None)
        width = getattr(page, "width", None)
        height = getattr(page, "height", None)

        if None not in (
            x,
            y,
            width,
            height,
        ):
            return (
                float(x),
                float(y),
                float(width),
                float(height),
            )

        corners = PreviewGenerator._get_corners(
            page
        )

        if corners:
            x_values = [
                point[0]
                for point in corners
            ]

            y_values = [
                point[1]
                for point in corners
            ]

            min_x = min(x_values)
            max_x = max(x_values)
            min_y = min(y_values)
            max_y = max(y_values)

            return (
                float(min_x),
                float(min_y),
                float(max_x - min_x),
                float(max_y - min_y),
            )

        raise ValueError(
            "No fue posible obtener la ubicación "
            "de una página para dibujar la vista previa."
        )
