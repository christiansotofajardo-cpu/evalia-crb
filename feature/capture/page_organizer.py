"""
page_organizer.py
=================

Organiza espacialmente las páginas detectadas por Smart Capture.

Objetivos iniciales:

1. Ordenar las hojas de arriba hacia abajo.
2. Ordenarlas de izquierda a derecha dentro de cada fila.
3. Agrupar dos páginas consecutivas por estudiante.
4. Detectar grupos incompletos cuando falta una hoja.

Supuesto inicial de Evalia:
cada estudiante entrega dos páginas impresas.

Este supuesto podrá configurarse posteriormente.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .models import DetectedPage, StudentPageGroup


class PageOrganizer:
    """
    Organiza páginas detectadas y las agrupa por estudiante.

    Parameters
    ----------
    pages_per_student:
        Número esperado de páginas por estudiante.

    row_tolerance:
        Tolerancia vertical relativa para decidir si dos páginas
        pertenecen a la misma fila de la fotografía.
    """

    def __init__(
        self,
        pages_per_student: int = 2,
        row_tolerance: float = 0.45,
    ) -> None:

        if pages_per_student < 1:
            raise ValueError(
                "pages_per_student debe ser igual o mayor que 1."
            )

        if row_tolerance <= 0:
            raise ValueError(
                "row_tolerance debe ser mayor que 0."
            )

        self.pages_per_student = pages_per_student
        self.row_tolerance = row_tolerance

    def group(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[StudentPageGroup]:
        """
        Ordena las páginas y las agrupa por estudiante.

        Parameters
        ----------
        pages:
            Páginas detectadas por PageDetector.

        Returns
        -------
        list[StudentPageGroup]
            Grupos de páginas ordenados por estudiante.
        """

        if not pages:
            return []

        ordered_pages = self.order_pages(pages)

        groups: List[StudentPageGroup] = []

        for start_index in range(
            0,
            len(ordered_pages),
            self.pages_per_student,
        ):
            group_pages = ordered_pages[
                start_index:
                start_index + self.pages_per_student
            ]

            student_number = len(groups) + 1
            complete = (
                len(group_pages)
                == self.pages_per_student
            )

            self._assign_page_numbers(group_pages)

            group = self._create_student_group(
                student_number=student_number,
                pages=group_pages,
                complete=complete,
            )

            groups.append(group)

        return groups

    def order_pages(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[DetectedPage]:
        """
        Ordena las hojas por filas visuales.

        Primero identifica filas de páginas usando sus centros
        verticales. Después ordena cada fila de izquierda a derecha.
        """

        pages_list = list(pages)

        if len(pages_list) <= 1:
            return pages_list

        page_data = [
            {
                "page": page,
                "center_x": self._get_center(page)[0],
                "center_y": self._get_center(page)[1],
                "height": self._get_dimensions(page)[1],
            }
            for page in pages_list
        ]

        page_data.sort(
            key=lambda item: (
                item["center_y"],
                item["center_x"],
            )
        )

        median_height = self._median(
            [
                max(float(item["height"]), 1.0)
                for item in page_data
            ]
        )

        vertical_tolerance = (
            median_height
            * self.row_tolerance
        )

        rows: List[List[dict]] = []

        for item in page_data:

            matching_row = None

            for row in rows:
                row_center_y = self._mean(
                    [
                        float(row_item["center_y"])
                        for row_item in row
                    ]
                )

                if (
                    abs(
                        float(item["center_y"])
                        - row_center_y
                    )
                    <= vertical_tolerance
                ):
                    matching_row = row
                    break

            if matching_row is None:
                rows.append([item])
            else:
                matching_row.append(item)

        rows.sort(
            key=lambda row: self._mean(
                [
                    float(item["center_y"])
                    for item in row
                ]
            )
        )

        ordered_pages: List[DetectedPage] = []

        for row in rows:

            row.sort(
                key=lambda item: float(
                    item["center_x"]
                )
            )

            ordered_pages.extend(
                item["page"]
                for item in row
            )

        return ordered_pages

    def regroup(
        self,
        pages: Sequence[DetectedPage],
        pages_per_student: int,
    ) -> List[StudentPageGroup]:
        """
        Permite reorganizar temporalmente usando otro número
        de páginas por estudiante.
        """

        if pages_per_student < 1:
            raise ValueError(
                "pages_per_student debe ser igual o mayor que 1."
            )

        original_value = self.pages_per_student

        try:
            self.pages_per_student = pages_per_student
            return self.group(pages)

        finally:
            self.pages_per_student = original_value

    def validate_groups(
        self,
        groups: Sequence[StudentPageGroup],
    ) -> dict:
        """
        Resume el estado de la organización.

        Resulta útil para la interfaz de confirmación del profesor.
        """

        total_groups = len(groups)

        complete_groups = sum(
            1
            for group in groups
            if self._group_is_complete(group)
        )

        incomplete_groups = (
            total_groups
            - complete_groups
        )

        total_pages = sum(
            len(self._group_pages(group))
            for group in groups
        )

        return {
            "total_pages": total_pages,
            "total_students": total_groups,
            "complete_students": complete_groups,
            "incomplete_students": incomplete_groups,
            "pages_per_student": self.pages_per_student,
            "requires_confirmation": (
                incomplete_groups > 0
            ),
        }

    def _assign_page_numbers(
        self,
        pages: Sequence[DetectedPage],
    ) -> None:
        """
        Asigna número de página dentro del grupo cuando el modelo
        permite modificar ese atributo.
        """

        for index, page in enumerate(
            pages,
            start=1,
        ):
            self._safe_setattr(
                page,
                "page_number",
                index,
            )

    def _create_student_group(
        self,
        student_number: int,
        pages: Sequence[DetectedPage],
        complete: bool,
    ) -> StudentPageGroup:
        """
        Construye un StudentPageGroup.

        Los nombres usados aquí deben coincidir con models.py.
        """

        return StudentPageGroup(
            student_number=student_number,
            pages=list(pages),
            complete=complete,
        )

    @staticmethod
    def _get_center(
        page: DetectedPage,
    ) -> Tuple[float, float]:
        """
        Obtiene el centro geométrico de una página.

        Admite distintos nombres de atributos para facilitar
        la evolución del modelo.
        """

        center = getattr(
            page,
            "center",
            None,
        )

        if center is not None:
            try:
                return (
                    float(center[0]),
                    float(center[1]),
                )
            except (
                TypeError,
                IndexError,
                ValueError,
            ):
                pass

        center_x = getattr(
            page,
            "center_x",
            None,
        )

        center_y = getattr(
            page,
            "center_y",
            None,
        )

        if (
            center_x is not None
            and center_y is not None
        ):
            return (
                float(center_x),
                float(center_y),
            )

        x, y, width, height = (
            PageOrganizer._get_bounding_box(page)
        )

        return (
            x + width / 2.0,
            y + height / 2.0,
        )

    @staticmethod
    def _get_dimensions(
        page: DetectedPage,
    ) -> Tuple[float, float]:
        """
        Obtiene ancho y alto de una página.
        """

        width = getattr(
            page,
            "width",
            None,
        )

        height = getattr(
            page,
            "height",
            None,
        )

        if (
            width is not None
            and height is not None
        ):
            return (
                float(width),
                float(height),
            )

        _, _, box_width, box_height = (
            PageOrganizer._get_bounding_box(page)
        )

        return (
            box_width,
            box_height,
        )

    @staticmethod
    def _get_bounding_box(
        page: DetectedPage,
    ) -> Tuple[float, float, float, float]:
        """
        Extrae la caja delimitadora de una página.

        Admite:

        - page.bounding_box
        - page.bbox
        - atributos x, y, width, height
        - puntos del cuadrilátero
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

        points = getattr(
            page,
            "corners",
            None,
        )

        if points is None:
            points = getattr(
                page,
                "contour",
                None,
            )

        if points is None:
            points = getattr(
                page,
                "points",
                None,
            )

        if points is not None:
            normalized_points = []

            for point in points:

                try:
                    point_x = float(point[0])
                    point_y = float(point[1])

                except (
                    TypeError,
                    IndexError,
                    ValueError,
                ):
                    continue

                normalized_points.append(
                    (
                        point_x,
                        point_y,
                    )
                )

            if normalized_points:

                x_values = [
                    point[0]
                    for point in normalized_points
                ]

                y_values = [
                    point[1]
                    for point in normalized_points
                ]

                min_x = min(x_values)
                max_x = max(x_values)
                min_y = min(y_values)
                max_y = max(y_values)

                return (
                    min_x,
                    min_y,
                    max_x - min_x,
                    max_y - min_y,
                )

        raise ValueError(
            "No fue posible obtener la posición "
            "de una página detectada."
        )

    @staticmethod
    def _group_is_complete(
        group: StudentPageGroup,
    ) -> bool:
        """
        Obtiene el estado de completitud del grupo.
        """

        complete = getattr(
            group,
            "complete",
            None,
        )

        if complete is not None:
            return bool(complete)

        return False

    @staticmethod
    def _group_pages(
        group: StudentPageGroup,
    ) -> List[DetectedPage]:
        """
        Obtiene las páginas de un grupo.
        """

        pages = getattr(
            group,
            "pages",
            [],
        )

        return list(pages)

    @staticmethod
    def _safe_setattr(
        instance: object,
        attribute: str,
        value: object,
    ) -> None:
        """
        Intenta asignar un atributo sin interrumpir el flujo
        cuando el modelo sea inmutable.
        """

        try:
            setattr(
                instance,
                attribute,
                value,
            )

        except (
            AttributeError,
            TypeError,
        ):
            pass

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:
        """
        Calcula la media sin depender de NumPy.
        """

        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _median(
        values: Sequence[float],
    ) -> float:
        """
        Calcula la mediana sin depender de NumPy.
        """

        if not values:
            return 0.0

        sorted_values = sorted(values)
        middle = len(sorted_values) // 2

        if len(sorted_values) % 2 == 1:
            return float(
                sorted_values[middle]
            )

        return float(
            (
                sorted_values[middle - 1]
                + sorted_values[middle]
            )
            / 2.0
        )
