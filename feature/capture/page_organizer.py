"""
page_organizer.py
=================

Organiza espacialmente las páginas detectadas por Smart Capture.

Responsabilidades
-----------------
1. Ordenar las hojas de manera determinista.
2. Detectar filas y columnas visuales.
3. Asignar metadatos espaciales cuando el modelo lo permite.
4. Agrupar las páginas por estudiante.
5. Detectar grupos incompletos o distribuciones ambiguas.

Supuesto inicial de Evalia:
cada estudiante entrega dos páginas impresas.

Para una fotografía con cuatro hojas en una cuadrícula 2 x 2,
el modo automático interpreta normalmente cada columna como un
estudiante:

    Estudiante 1       Estudiante 2
    página 1           página 1
    página 2           página 2

La API pública anterior se conserva:
- PageOrganizer.group(...)
- PageOrganizer.order_pages(...)
- PageOrganizer.regroup(...)
- PageOrganizer.validate_groups(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .models import DetectedPage, StudentPageGroup


@dataclass
class _PageLayoutItem:
    """Representación interna de una página dentro del layout."""

    page: DetectedPage
    center_x: float
    center_y: float
    width: float
    height: float
    row: int = 0
    column: int = 0


class PageOrganizer:
    """
    Organiza páginas detectadas y las agrupa por estudiante.

    Parameters
    ----------
    pages_per_student:
        Número esperado de páginas por estudiante.

    row_tolerance:
        Tolerancia vertical relativa, calculada sobre la altura mediana,
        para decidir si dos páginas pertenecen a la misma fila.

    column_tolerance:
        Tolerancia horizontal relativa, calculada sobre el ancho mediano,
        para decidir si dos páginas pertenecen a la misma columna.

    grouping_mode:
        Estrategia de agrupación:

        - ``"auto"``: usa columnas cuando cada columna contiene exactamente
          ``pages_per_student`` páginas; en otros casos usa orden visual.
        - ``"columns"``: agrupa por columnas, de arriba hacia abajo.
        - ``"rows"``: agrupa por filas, de izquierda a derecha.
        - ``"sequential"``: agrupa consecutivamente según orden visual.
    """

    VALID_GROUPING_MODES = {
        "auto",
        "columns",
        "rows",
        "sequential",
    }

    def __init__(
        self,
        pages_per_student: int = 2,
        row_tolerance: float = 0.35,
        column_tolerance: float = 0.35,
        grouping_mode: str = "auto",
    ) -> None:

        if pages_per_student < 1:
            raise ValueError(
                "pages_per_student debe ser igual o mayor que 1."
            )

        if row_tolerance <= 0:
            raise ValueError(
                "row_tolerance debe ser mayor que 0."
            )

        if column_tolerance <= 0:
            raise ValueError(
                "column_tolerance debe ser mayor que 0."
            )

        if grouping_mode not in self.VALID_GROUPING_MODES:
            valid_modes = ", ".join(
                sorted(self.VALID_GROUPING_MODES)
            )
            raise ValueError(
                f"grouping_mode debe ser uno de: {valid_modes}."
            )

        self.pages_per_student = pages_per_student
        self.row_tolerance = row_tolerance
        self.column_tolerance = column_tolerance
        self.grouping_mode = grouping_mode

        self._last_layout: List[_PageLayoutItem] = []
        self._last_grouping_strategy: str = "none"
        self._last_warnings: List[str] = []

    def group(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[StudentPageGroup]:
        """
        Ordena las páginas y las agrupa por estudiante.

        El orden espacial se calcula siempre desde las coordenadas reales;
        nunca se confía en el orden entregado por OpenCV.
        """

        self._last_warnings = []

        if not pages:
            self._last_layout = []
            self._last_grouping_strategy = "none"
            return []

        layout = self._build_layout(pages)
        self._last_layout = layout

        grouped_pages = self._select_grouping(layout)

        groups: List[StudentPageGroup] = []

        for student_number, group_pages in enumerate(
            grouped_pages,
            start=1,
        ):
            complete = (
                len(group_pages)
                == self.pages_per_student
            )

            self._assign_page_numbers(group_pages)
            self._assign_student_metadata(
                group_pages,
                student_number,
            )

            group = self._create_student_group(
                student_number=student_number,
                pages=group_pages,
                complete=complete,
            )

            groups.append(group)

        if any(
            not self._group_is_complete(group)
            for group in groups
        ):
            self._last_warnings.append(
                "Hay uno o más estudiantes con un número "
                "incompleto de páginas."
            )

        return groups

    def order_pages(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[DetectedPage]:
        """
        Devuelve las páginas en orden visual estable:

        arriba izquierda -> arriba derecha ->
        siguiente fila de izquierda a derecha.
        """

        if not pages:
            self._last_layout = []
            return []

        layout = self._build_layout(pages)
        self._last_layout = layout

        ordered_layout = sorted(
            layout,
            key=lambda item: (
                item.row,
                item.column,
                item.center_y,
                item.center_x,
            ),
        )

        return [
            item.page
            for item in ordered_layout
        ]

    def regroup(
        self,
        pages: Sequence[DetectedPage],
        pages_per_student: int,
    ) -> List[StudentPageGroup]:
        """
        Reorganiza temporalmente usando otro número de páginas
        por estudiante.
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
        Resume el estado de la organización para la interfaz docente.
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
            "grouping_strategy": self._last_grouping_strategy,
            "rows_detected": self._count_layout_values("row"),
            "columns_detected": self._count_layout_values("column"),
            "warnings": list(self._last_warnings),
            "requires_confirmation": (
                incomplete_groups > 0
                or bool(self._last_warnings)
            ),
        }

    def get_layout_summary(self) -> dict:
        """
        Entrega información espacial útil para depuración y vista previa.
        """

        return {
            "strategy": self._last_grouping_strategy,
            "rows": self._count_layout_values("row"),
            "columns": self._count_layout_values("column"),
            "pages": [
                {
                    "row": item.row + 1,
                    "column": item.column + 1,
                    "center_x": item.center_x,
                    "center_y": item.center_y,
                }
                for item in sorted(
                    self._last_layout,
                    key=lambda current: (
                        current.row,
                        current.column,
                    ),
                )
            ],
            "warnings": list(self._last_warnings),
        }

    def _build_layout(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[_PageLayoutItem]:
        """
        Construye filas y columnas estables desde los centros geométricos.
        """

        items: List[_PageLayoutItem] = []

        for page in pages:
            center_x, center_y = self._get_center(page)
            width, height = self._get_dimensions(page)

            items.append(
                _PageLayoutItem(
                    page=page,
                    center_x=center_x,
                    center_y=center_y,
                    width=max(width, 1.0),
                    height=max(height, 1.0),
                )
            )

        if len(items) == 1:
            self._apply_layout_metadata(items)
            return items

        median_height = self._median(
            [item.height for item in items]
        )
        median_width = self._median(
            [item.width for item in items]
        )

        row_threshold = (
            median_height
            * self.row_tolerance
        )
        column_threshold = (
            median_width
            * self.column_tolerance
        )

        row_clusters = self._cluster_axis(
            items=items,
            axis="y",
            tolerance=row_threshold,
        )

        column_clusters = self._cluster_axis(
            items=items,
            axis="x",
            tolerance=column_threshold,
        )

        for row_index, cluster in enumerate(row_clusters):
            for item in cluster:
                item.row = row_index

        for column_index, cluster in enumerate(column_clusters):
            for item in cluster:
                item.column = column_index

        self._apply_layout_metadata(items)
        return items

    def _select_grouping(
        self,
        layout: Sequence[_PageLayoutItem],
    ) -> List[List[DetectedPage]]:
        """
        Selecciona la estrategia de agrupación configurada.
        """

        if self.grouping_mode == "columns":
            return self._group_by_columns(layout)

        if self.grouping_mode == "rows":
            return self._group_by_rows(layout)

        if self.grouping_mode == "sequential":
            return self._group_sequential(layout)

        # Modo automático.
        columns = self._items_by_index(
            layout,
            attribute="column",
        )

        column_sizes = [
            len(column_items)
            for column_items in columns.values()
        ]

        if (
            len(columns) >= 1
            and column_sizes
            and all(
                size == self.pages_per_student
                for size in column_sizes
            )
        ):
            self._last_grouping_strategy = "columns"
            return self._group_by_columns(layout)

        self._last_grouping_strategy = "sequential"

        if len(layout) > self.pages_per_student:
            self._last_warnings.append(
                "La distribución no formó columnas completas; "
                "se aplicó agrupación visual consecutiva."
            )

        return self._group_sequential(layout)

    def _group_by_columns(
        self,
        layout: Sequence[_PageLayoutItem],
    ) -> List[List[DetectedPage]]:
        """
        Cada columna representa un estudiante.

        Dentro de la columna, las páginas se ordenan de arriba abajo.
        """

        columns = self._items_by_index(
            layout,
            attribute="column",
        )

        groups: List[List[DetectedPage]] = []

        for column_index in sorted(columns):
            column_items = sorted(
                columns[column_index],
                key=lambda item: (
                    item.row,
                    item.center_y,
                ),
            )

            pages = [
                item.page
                for item in column_items
            ]

            groups.extend(
                self._split_pages(pages)
            )

        self._last_grouping_strategy = "columns"
        return groups

    def _group_by_rows(
        self,
        layout: Sequence[_PageLayoutItem],
    ) -> List[List[DetectedPage]]:
        """
        Cada fila se procesa de izquierda a derecha.
        """

        rows = self._items_by_index(
            layout,
            attribute="row",
        )

        ordered_pages: List[DetectedPage] = []

        for row_index in sorted(rows):
            row_items = sorted(
                rows[row_index],
                key=lambda item: (
                    item.column,
                    item.center_x,
                ),
            )

            ordered_pages.extend(
                item.page
                for item in row_items
            )

        self._last_grouping_strategy = "rows"
        return self._split_pages(ordered_pages)

    def _group_sequential(
        self,
        layout: Sequence[_PageLayoutItem],
    ) -> List[List[DetectedPage]]:
        """
        Agrupa consecutivamente usando el orden visual estable.
        """

        ordered_items = sorted(
            layout,
            key=lambda item: (
                item.row,
                item.column,
                item.center_y,
                item.center_x,
            ),
        )

        ordered_pages = [
            item.page
            for item in ordered_items
        ]

        self._last_grouping_strategy = "sequential"
        return self._split_pages(ordered_pages)

    def _split_pages(
        self,
        pages: Sequence[DetectedPage],
    ) -> List[List[DetectedPage]]:
        """
        Divide una secuencia en grupos de pages_per_student.
        """

        return [
            list(
                pages[
                    start_index:
                    start_index + self.pages_per_student
                ]
            )
            for start_index in range(
                0,
                len(pages),
                self.pages_per_student,
            )
        ]

    @staticmethod
    def _items_by_index(
        layout: Sequence[_PageLayoutItem],
        attribute: str,
    ) -> Dict[int, List[_PageLayoutItem]]:

        result: Dict[int, List[_PageLayoutItem]] = {}

        for item in layout:
            index = int(
                getattr(item, attribute)
            )
            result.setdefault(index, []).append(item)

        return result

    @staticmethod
    def _cluster_axis(
        items: Sequence[_PageLayoutItem],
        axis: str,
        tolerance: float,
    ) -> List[List[_PageLayoutItem]]:
        """
        Agrupa elementos por proximidad en un eje.

        El centro del clúster se recalcula después de cada incorporación,
        evitando depender del primer elemento detectado.
        """

        if axis not in {"x", "y"}:
            raise ValueError(
                "axis debe ser 'x' o 'y'."
            )

        coordinate_name = (
            "center_x"
            if axis == "x"
            else "center_y"
        )

        ordered = sorted(
            items,
            key=lambda item: float(
                getattr(item, coordinate_name)
            ),
        )

        clusters: List[List[_PageLayoutItem]] = []

        for item in ordered:
            coordinate = float(
                getattr(item, coordinate_name)
            )

            best_cluster: Optional[
                List[_PageLayoutItem]
            ] = None
            best_distance: Optional[float] = None

            for cluster in clusters:
                cluster_center = PageOrganizer._mean(
                    [
                        float(
                            getattr(
                                current,
                                coordinate_name,
                            )
                        )
                        for current in cluster
                    ]
                )

                distance = abs(
                    coordinate
                    - cluster_center
                )

                if (
                    distance <= tolerance
                    and (
                        best_distance is None
                        or distance < best_distance
                    )
                ):
                    best_cluster = cluster
                    best_distance = distance

            if best_cluster is None:
                clusters.append([item])
            else:
                best_cluster.append(item)

        clusters.sort(
            key=lambda cluster: PageOrganizer._mean(
                [
                    float(
                        getattr(
                            item,
                            coordinate_name,
                        )
                    )
                    for item in cluster
                ]
            )
        )

        return clusters

    def _apply_layout_metadata(
        self,
        layout: Sequence[_PageLayoutItem],
    ) -> None:
        """
        Guarda fila, columna y orden visual en el modelo cuando es mutable.
        """

        ordered = sorted(
            layout,
            key=lambda item: (
                item.row,
                item.column,
                item.center_y,
                item.center_x,
            ),
        )

        for visual_index, item in enumerate(
            ordered,
            start=1,
        ):
            self._safe_setattr(
                item.page,
                "layout_row",
                item.row + 1,
            )
            self._safe_setattr(
                item.page,
                "layout_column",
                item.column + 1,
            )
            self._safe_setattr(
                item.page,
                "visual_order",
                visual_index,
            )

    def _assign_page_numbers(
        self,
        pages: Sequence[DetectedPage],
    ) -> None:

        for index, page in enumerate(
            pages,
            start=1,
        ):
            self._safe_setattr(
                page,
                "page_number",
                index,
            )

    def _assign_student_metadata(
        self,
        pages: Sequence[DetectedPage],
        student_number: int,
    ) -> None:

        for page in pages:
            self._safe_setattr(
                page,
                "student_number",
                student_number,
            )

    def _create_student_group(
        self,
        student_number: int,
        pages: Sequence[DetectedPage],
        complete: bool,
    ) -> StudentPageGroup:

        return StudentPageGroup(
            student_number=student_number,
            pages=list(pages),
            complete=complete,
        )

    def _count_layout_values(
        self,
        attribute: str,
    ) -> int:

        if not self._last_layout:
            return 0

        return len(
            {
                int(getattr(item, attribute))
                for item in self._last_layout
            }
        )

    @staticmethod
    def _get_center(
        page: DetectedPage,
    ) -> Tuple[float, float]:

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

        try:
            setattr(
                instance,
                attribute,
                value,
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            pass

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:

        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _median(
        values: Sequence[float],
    ) -> float:

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

