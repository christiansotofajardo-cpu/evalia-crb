"""
quality_analyzer.py
===================

Analiza la calidad técnica de una fotografía antes de enviarla
al detector de páginas.

Evalúa:

- resolución;
- nitidez o desenfoque;
- iluminación promedio;
- contraste;
- zonas demasiado oscuras;
- zonas sobreexpuestas.

Este módulo no rechaza automáticamente la fotografía:
entrega un diagnóstico y recomendaciones para el profesor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .models import QualityReport


class QualityAnalyzer:
    """
    Analizador de calidad para fotografías de evaluaciones impresas.

    Los umbrales iniciales son conservadores y podrán calibrarse
    posteriormente con fotografías reales tomadas por profesores.
    """

    def __init__(
        self,
        min_width: int = 1200,
        min_height: int = 900,
        min_blur_score: float = 85.0,
        min_brightness: float = 45.0,
        max_brightness: float = 220.0,
        min_contrast: float = 28.0,
        max_dark_ratio: float = 0.35,
        max_bright_ratio: float = 0.25,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_dark_ratio = max_dark_ratio
        self.max_bright_ratio = max_bright_ratio

    def analyze(self, image: np.ndarray) -> QualityReport:
        """
        Analiza una fotografía y devuelve un QualityReport.

        Parameters
        ----------
        image:
            Imagen OpenCV en formato BGR o escala de grises.

        Returns
        -------
        QualityReport
            Resultado del análisis técnico.
        """

        self._validate_image(image)

        height, width = image.shape[:2]
        gray = self._to_gray(image)

        blur_score = self._calculate_blur_score(gray)
        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)
        dark_ratio = self._calculate_dark_ratio(gray)
        bright_ratio = self._calculate_bright_ratio(gray)

        warnings: List[str] = []
        recommendations: List[str] = []

        resolution_ok = (
            width >= self.min_width
            and height >= self.min_height
        )

        blur_ok = blur_score >= self.min_blur_score

        brightness_ok = (
            self.min_brightness
            <= brightness
            <= self.max_brightness
        )

        contrast_ok = contrast >= self.min_contrast
        shadows_ok = dark_ratio <= self.max_dark_ratio
        exposure_ok = bright_ratio <= self.max_bright_ratio

        if not resolution_ok:
            warnings.append("La resolución de la fotografía es baja.")
            recommendations.append(
                "Acerca la cámara o utiliza una resolución mayor."
            )

        if not blur_ok:
            warnings.append(
                "La fotografía presenta desenfoque o movimiento."
            )
            recommendations.append(
                "Mantén el teléfono firme y vuelve a tomar la fotografía."
            )

        if brightness < self.min_brightness:
            warnings.append("La fotografía está demasiado oscura.")
            recommendations.append(
                "Aumenta la iluminación y evita proyectar sombras sobre las hojas."
            )

        elif brightness > self.max_brightness:
            warnings.append("La fotografía está demasiado iluminada.")
            recommendations.append(
                "Reduce la luz directa o evita utilizar flash sobre el papel."
            )

        if not contrast_ok:
            warnings.append(
                "Existe poco contraste entre el papel y la escritura."
            )
            recommendations.append(
                "Usa una iluminación uniforme y procura que las hojas estén bien enfocadas."
            )

        if not shadows_ok:
            warnings.append(
                "Hay demasiadas zonas oscuras en la fotografía."
            )
            recommendations.append(
                "Evita cubrir las hojas con la mano, el teléfono u otros objetos."
            )

        if not exposure_ok:
            warnings.append(
                "Hay zonas sobreexpuestas o con reflejos intensos."
            )
            recommendations.append(
                "Cambia el ángulo de la cámara para eliminar reflejos."
            )

        critical_failures = sum(
            [
                not resolution_ok,
                not blur_ok,
                not brightness_ok,
            ]
        )

        secondary_failures = sum(
            [
                not contrast_ok,
                not shadows_ok,
                not exposure_ok,
            ]
        )

        acceptable = (
            critical_failures == 0
            and secondary_failures <= 1
        )

        quality_score = self._calculate_quality_score(
            resolution_ok=resolution_ok,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            dark_ratio=dark_ratio,
            bright_ratio=bright_ratio,
        )

        message = self._build_message(
            acceptable=acceptable,
            warnings=warnings,
            quality_score=quality_score,
        )

        return self._create_report(
            acceptable=acceptable,
            quality_score=quality_score,
            width=width,
            height=height,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            dark_ratio=dark_ratio,
            bright_ratio=bright_ratio,
            warnings=warnings,
            recommendations=recommendations,
            message=message,
        )

    def _validate_image(self, image: np.ndarray) -> None:
        """Verifica que la entrada sea una imagen válida."""

        if image is None:
            raise ValueError(
                "No se recibió una imagen para analizar."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "La imagen debe ser un arreglo numpy.ndarray."
            )

        if image.size == 0:
            raise ValueError(
                "La imagen recibida está vacía."
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                "La imagen debe estar en escala de grises, BGR o BGRA."
            )

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(
                "La imagen contiene un número de canales no compatible."
            )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convierte la imagen a escala de grises."""

        if image.ndim == 2:
            return image

        channels = image.shape[2]

        if channels == 1:
            return image[:, :, 0]

        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _calculate_blur_score(gray: np.ndarray) -> float:
        """
        Calcula nitidez mediante la varianza del Laplaciano.

        Un valor bajo suele indicar una imagen desenfocada.
        """

        score = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        return round(float(score), 2)

    @staticmethod
    def _calculate_brightness(gray: np.ndarray) -> float:
        """Calcula el brillo medio de la imagen."""

        return round(float(np.mean(gray)), 2)

    @staticmethod
    def _calculate_contrast(gray: np.ndarray) -> float:
        """Calcula contraste mediante desviación estándar."""

        return round(float(np.std(gray)), 2)

    @staticmethod
    def _calculate_dark_ratio(gray: np.ndarray) -> float:
        """
        Proporción de píxeles muy oscuros.

        Se consideran oscuros los valores inferiores a 35.
        """

        ratio = np.mean(gray < 35)
        return round(float(ratio), 4)

    @staticmethod
    def _calculate_bright_ratio(gray: np.ndarray) -> float:
        """
        Proporción de píxeles muy claros.

        Se consideran sobreexpuestos los valores superiores a 245.
        """

        ratio = np.mean(gray > 245)
        return round(float(ratio), 4)

    def _calculate_quality_score(
        self,
        resolution_ok: bool,
        blur_score: float,
        brightness: float,
        contrast: float,
        dark_ratio: float,
        bright_ratio: float,
    ) -> float:
        """
        Calcula una puntuación global aproximada entre 0 y 100.

        Esta puntuación sirve para orientar al profesor, pero no
        reemplaza los indicadores individuales.
        """

        resolution_component = 100.0 if resolution_ok else 45.0

        blur_component = min(
            100.0,
            max(
                0.0,
                blur_score / max(self.min_blur_score, 1.0) * 80.0,
            ),
        )

        ideal_brightness = 135.0
        brightness_distance = abs(
            brightness - ideal_brightness
        )

        brightness_component = max(
            0.0,
            100.0 - brightness_distance * 0.85,
        )

        contrast_component = min(
            100.0,
            max(
                0.0,
                contrast / max(self.min_contrast, 1.0) * 75.0,
            ),
        )

        shadows_component = max(
            0.0,
            100.0
            - (
                dark_ratio
                / max(self.max_dark_ratio, 0.01)
            )
            * 50.0,
        )

        exposure_component = max(
            0.0,
            100.0
            - (
                bright_ratio
                / max(self.max_bright_ratio, 0.01)
            )
            * 50.0,
        )

        weighted_score = (
            resolution_component * 0.15
            + blur_component * 0.30
            + brightness_component * 0.20
            + contrast_component * 0.15
            + shadows_component * 0.10
            + exposure_component * 0.10
        )

        return round(
            min(100.0, max(0.0, weighted_score)),
            1,
        )

    @staticmethod
    def _build_message(
        acceptable: bool,
        warnings: List[str],
        quality_score: float,
    ) -> str:
        """Genera un mensaje breve para la interfaz."""

        if acceptable and not warnings:
            return (
                "La fotografía presenta buenas condiciones "
                "para detectar y procesar las hojas."
            )

        if acceptable:
            return (
                "La fotografía puede procesarse, aunque Evalia "
                "detectó algunos aspectos que conviene revisar."
            )

        return (
            "La calidad de la fotografía podría afectar la "
            "detección de páginas. Se recomienda tomarla nuevamente."
        )

    @staticmethod
    def _create_report(
        acceptable: bool,
        quality_score: float,
        width: int,
        height: int,
        blur_score: float,
        brightness: float,
        contrast: float,
        dark_ratio: float,
        bright_ratio: float,
        warnings: List[str],
        recommendations: List[str],
        message: str,
    ) -> QualityReport:
        """
        Construye el QualityReport.

        Los nombres utilizados aquí deben coincidir con el modelo
        definido en models.py.
        """

        return QualityReport(
            acceptable=acceptable,
            quality_score=quality_score,
            width=width,
            height=height,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            dark_ratio=dark_ratio,
            bright_ratio=bright_ratio,
            warnings=warnings,
            recommendations=recommendations,
            message=message,
        )

    def get_thresholds(self) -> Dict[str, Any]:
        """
        Entrega los umbrales activos.

        Es útil para diagnóstico, pruebas y futuras configuraciones
        de administración.
        """

        return {
            "min_width": self.min_width,
            "min_height": self.min_height,
            "min_blur_score": self.min_blur_score,
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "min_contrast": self.min_contrast,
            "max_dark_ratio": self.max_dark_ratio,
            "max_bright_ratio": self.max_bright_ratio,
        }
