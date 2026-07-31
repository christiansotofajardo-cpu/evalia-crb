"""
Servicio OCR reutilizable de Evalia.

Orden de ejecución:

1. Mistral OCR, cuando existe MISTRAL_API_KEY.
2. EasyOCR, cuando está habilitado e instalado.
3. Tesseract, cuando está habilitado e instalado.
4. Fallback manual seguro.

Este módulo no depende de main.py. De esta manera puede ser utilizado tanto
por el OCR tradicional como por Smart Capture sin producir importaciones
circulares.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union


logger = logging.getLogger("evalia.ocr")

PathLike = Union[str, Path]


OCR_STATUS: Dict[str, Any] = {
    "engine": "manual_fallback",
    "available": False,
    "message": (
        "OCR automático no disponible; "
        "se habilita transcripción/edición manual."
    ),
}


try:
    from PIL import Image, ImageFilter, ImageOps

    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageFilter = None
    ImageOps = None
    PIL_AVAILABLE = False


def _log_event(event: str, **kwargs: Any) -> None:
    """
    Registra eventos del OCR sin depender del logger definido en main.py.
    """
    try:
        payload = " | ".join(
            f"{key}={value}"
            for key, value in kwargs.items()
        )

        logger.info(
            "%s | %s" if payload else "%s",
            event,
            payload,
        )
    except Exception:
        pass


def _path(path: PathLike) -> Path:
    """
    Convierte una ruta recibida como string o Path en un objeto Path.
    """
    return path if isinstance(path, Path) else Path(path)


def get_ocr_status() -> Dict[str, Any]:
    """
    Devuelve una copia del estado actual del servicio OCR.
    """
    return dict(OCR_STATUS)


def preprocess_image_for_ocr(path: PathLike) -> Path:
    """
    Realiza un preprocesamiento liviano para fotografías.

    Aplica:

    - corrección de orientación EXIF;
    - ampliación de imágenes pequeñas;
    - conversión a escala de grises;
    - mejora automática del contraste;
    - enfoque ligero.

    Si Pillow no está disponible o el procesamiento falla, devuelve la ruta
    original sin interrumpir Evalia.
    """
    source_path = _path(path)

    if not PIL_AVAILABLE:
        return source_path

    try:
        with Image.open(source_path) as original:
            image = ImageOps.exif_transpose(original)

            max_side = max(image.size)

            if max_side > 0 and max_side < 1800:
                scale = 1800 / max_side

                image = image.resize(
                    (
                        int(image.width * scale),
                        int(image.height * scale),
                    )
                )

            image = ImageOps.grayscale(image)
            image = ImageOps.autocontrast(image)
            image = image.filter(ImageFilter.SHARPEN)

            output_path = Path(f"{source_path}_pre.png")
            image.save(output_path)

        return output_path

    except Exception as exc:
        _log_event(
            "ocr_preprocess_failed",
            file=source_path.name,
            error=str(exc),
        )
        return source_path


def extract_text_from_mistral_response(payload: Any) -> str:
    """
    Extrae texto desde distintas variantes de respuesta de Mistral OCR.
    """
    if not isinstance(payload, dict):
        return ""

    parts = []

    for page in payload.get("pages", []) or []:
        if not isinstance(page, dict):
            continue

        text = (
            page.get("markdown")
            or page.get("text")
            or page.get("content")
            or ""
        )

        if text:
            parts.append(str(text))

    for key in (
        "markdown",
        "text",
        "content",
        "output_text",
    ):
        value = payload.get(key)

        if value:
            parts.append(str(value))

    for choice in payload.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue

        message = choice.get("message", {})

        if not isinstance(message, dict):
            continue

        content = message.get("content", "")

        if content:
            parts.append(str(content))

    return "\n\n".join(
        part.strip()
        for part in parts
        if str(part).strip()
    ).strip()


def run_mistral_ocr_on_file(
    path: PathLike,
) -> Optional[Dict[str, Any]]:
    """
    Ejecuta Mistral OCR como motor principal.

    Requiere la variable de entorno:

        MISTRAL_API_KEY

    El archivo se envía como base64, por lo que no necesita una URL pública.
    """
    started = time.time()
    source_path = _path(path)

    api_key = os.getenv(
        "MISTRAL_API_KEY",
        "",
    ).strip()

    if not api_key:
        return None

    try:
        import requests
    except Exception as exc:
        return {
            "text": "",
            "confidence": 0.0,
            "engine": "mistral_unavailable",
            "seconds": round(time.time() - started, 3),
            "message": (
                "MISTRAL_API_KEY existe, pero falta requests "
                f"en requirements.txt: {exc}"
            ),
        }

    try:
        file_bytes = source_path.read_bytes()

        mime_type = (
            mimetypes.guess_type(str(source_path))[0]
            or "image/jpeg"
        )

        encoded = base64.b64encode(
            file_bytes
        ).decode("utf-8")

        data_url = (
            f"data:{mime_type};base64,{encoded}"
        )

        if mime_type.startswith("image/"):
            document = {
                "type": "image_url",
                "image_url": data_url,
            }
        else:
            document = {
                "type": "document_url",
                "document_url": data_url,
            }

        model_name = os.getenv(
            "EVALIA_MISTRAL_OCR_MODEL",
            "mistral-ocr-latest",
        )

        timeout = int(
            os.getenv(
                "EVALIA_OCR_TIMEOUT",
                "90",
            )
        )

        request_payload = {
            "model": model_name,
            "document": document,
            "include_image_base64": False,
        }

        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=request_payload,
            timeout=timeout,
        )

        if response.status_code >= 400:
            message = response.text[:700]

            _log_event(
                "mistral_ocr_failed",
                file=source_path.name,
                status=response.status_code,
                error=message,
            )

            return {
                "text": "",
                "confidence": 0.0,
                "engine": "mistral_failed",
                "seconds": round(
                    time.time() - started,
                    3,
                ),
                "message": (
                    "Mistral OCR respondió "
                    f"{response.status_code}: {message}"
                ),
            }

        data = response.json()
        text = extract_text_from_mistral_response(data)

        if not text:
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "mistral_empty",
                "seconds": round(
                    time.time() - started,
                    3,
                ),
                "message": (
                    "Mistral OCR se ejecutó, "
                    "pero no devolvió texto útil."
                ),
            }

        OCR_STATUS.update(
            {
                "engine": "mistral_ocr",
                "available": True,
                "message": (
                    "OCR moderno activo con Mistral OCR."
                ),
            }
        )

        return {
            "text": text,
            "confidence": 0.88,
            "engine": "mistral_ocr",
            "seconds": round(
                time.time() - started,
                3,
            ),
            "message": (
                f"OCR moderno aplicado con {model_name}."
            ),
        }

    except Exception as exc:
        _log_event(
            "mistral_ocr_exception",
            file=source_path.name,
            error=str(exc),
        )

        return {
            "text": "",
            "confidence": 0.0,
            "engine": "mistral_exception",
            "seconds": round(
                time.time() - started,
                3,
            ),
            "message": (
                f"Mistral OCR falló: {exc}"
            ),
        }


def run_easyocr_on_image(
    path: PathLike,
) -> Optional[Dict[str, Any]]:
    """
    Ejecuta EasyOCR como respaldo local opcional.
    """
    started = time.time()
    source_path = _path(path)

    if not PIL_AVAILABLE:
        return None

    try:
        import easyocr
    except Exception:
        return None

    try:
        image_path = preprocess_image_for_ocr(
            source_path
        )

        languages = os.getenv(
            "EVALIA_EASYOCR_LANGS",
            "es,en",
        ).split(",")

        languages = [
            language.strip()
            for language in languages
            if language.strip()
        ]

        reader = easyocr.Reader(
            languages,
            gpu=False,
        )

        result = reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
        )

        rows = []
        confidences = []

        for box, text, confidence in result:
            text = str(text).strip()

            if not text:
                continue

            try:
                top = min(
                    point[1]
                    for point in box
                )
                left = min(
                    point[0]
                    for point in box
                )
            except Exception:
                top = 0
                left = 0

            numeric_confidence = float(
                confidence or 0
            )

            rows.append(
                {
                    "top": top,
                    "left": left,
                    "text": text,
                    "confidence": numeric_confidence,
                }
            )

            confidences.append(
                numeric_confidence
            )

        rows.sort(
            key=lambda row: (
                round(row["top"] / 18),
                row["left"],
            )
        )

        lines = []
        current_line = []
        current_y = None

        for row in rows:
            y_position = round(
                row["top"] / 18
            )

            if (
                current_y is None
                or y_position == current_y
            ):
                current_line.append(row)
                current_y = y_position
                continue

            line = " ".join(
                item["text"]
                for item in sorted(
                    current_line,
                    key=lambda item: item["left"],
                )
            ).strip()

            if line:
                lines.append(line)

            current_line = [row]
            current_y = y_position

        if current_line:
            line = " ".join(
                item["text"]
                for item in sorted(
                    current_line,
                    key=lambda item: item["left"],
                )
            ).strip()

            if line:
                lines.append(line)

        text = "\n".join(lines).strip()

        if not text:
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "easyocr_empty",
                "seconds": round(
                    time.time() - started,
                    3,
                ),
                "message": (
                    "EasyOCR se ejecutó, "
                    "pero no detectó texto útil."
                ),
            }

        average_confidence = round(
            sum(confidences)
            / max(len(confidences), 1),
            2,
        )

        OCR_STATUS.update(
            {
                "engine": "easyocr",
                "available": True,
                "message": (
                    "OCR local moderno activo con EasyOCR."
                ),
            }
        )

        return {
            "text": text,
            "confidence": average_confidence,
            "engine": "easyocr",
            "seconds": round(
                time.time() - started,
                3,
            ),
            "message": (
                "OCR local aplicado con EasyOCR."
            ),
        }

    except Exception as exc:
        _log_event(
            "easyocr_failed",
            file=source_path.name,
            error=str(exc),
        )

        return {
            "text": "",
            "confidence": 0.0,
            "engine": "easyocr_failed",
            "seconds": round(
                time.time() - started,
                3,
            ),
            "message": (
                f"EasyOCR falló: {exc}"
            ),
        }


def run_tesseract_ocr_on_image(
    path: PathLike,
) -> Optional[Dict[str, Any]]:
    """
    Ejecuta Tesseract como respaldo clásico.

    Conserva las líneas y los bloques detectados.
    """
    started = time.time()
    source_path = _path(path)

    if not PIL_AVAILABLE:
        return None

    try:
        import pytesseract

        image_path = preprocess_image_for_ocr(
            source_path
        )

        image = Image.open(image_path)

        config = os.getenv(
            "EVALIA_TESSERACT_CONFIG",
            "--oem 3 --psm 6",
        )

        language = os.getenv(
            "EVALIA_OCR_LANG",
            "spa+eng",
        )

        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        rows = []
        confidences = []

        number_of_items = len(
            data.get("text", [])
        )

        for index in range(number_of_items):
            text = str(
                data.get("text", [""])[index]
            ).strip()

            if not text:
                continue

            try:
                confidence = float(
                    data.get("conf", [0])[index]
                )

                if confidence >= 0:
                    confidences.append(
                        confidence
                    )
            except Exception:
                confidence = 0

            rows.append(
                {
                    "block": data.get(
                        "block_num",
                        [0],
                    )[index],
                    "paragraph": data.get(
                        "par_num",
                        [0],
                    )[index],
                    "line": data.get(
                        "line_num",
                        [0],
                    )[index],
                    "word": data.get(
                        "word_num",
                        [0],
                    )[index],
                    "left": data.get(
                        "left",
                        [0],
                    )[index],
                    "top": data.get(
                        "top",
                        [0],
                    )[index],
                    "text": text,
                    "confidence": confidence,
                }
            )

        grouped_rows = {}

        for row in rows:
            key = (
                row["block"],
                row["paragraph"],
                row["line"],
            )

            grouped_rows.setdefault(
                key,
                [],
            ).append(row)

        lines = []

        for key in sorted(grouped_rows):
            words = sorted(
                grouped_rows[key],
                key=lambda item: (
                    item["left"],
                    item["word"],
                ),
            )

            line = " ".join(
                word["text"]
                for word in words
            ).strip()

            if line:
                lines.append(line)

        text = "\n".join(lines).strip()

        average_confidence = (
            round(
                (
                    sum(confidences)
                    / len(confidences)
                )
                / 100,
                2,
            )
            if confidences
            else 0.0
        )

        if not text:
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "pytesseract_empty",
                "seconds": round(
                    time.time() - started,
                    3,
                ),
                "message": (
                    "Tesseract se ejecutó, "
                    "pero no detectó texto útil."
                ),
            }

        OCR_STATUS.update(
            {
                "engine": "pytesseract",
                "available": True,
                "message": (
                    "OCR clásico activo con pytesseract."
                ),
            }
        )

        return {
            "text": text,
            "confidence": average_confidence,
            "engine": "pytesseract_lines",
            "seconds": round(
                time.time() - started,
                3,
            ),
            "message": (
                "OCR clásico aplicado conservando líneas."
            ),
        }

    except Exception as exc:
        message = str(exc)
        hint = ""

        if "tesseract" in message.lower():
            hint = (
                " En Render esto suele indicar que falta "
                "el binario del sistema Tesseract, "
                "no solamente la librería pytesseract."
            )

        _log_event(
            "tesseract_unavailable_or_failed",
            file=source_path.name,
            error=message,
        )

        return {
            "text": "",
            "confidence": 0.0,
            "engine": "pytesseract_failed",
            "seconds": round(
                time.time() - started,
                3,
            ),
            "message": (
                "Tesseract no disponible o falló: "
                f"{message}.{hint}"
            ),
        }


def run_ocr_on_image(
    path: PathLike,
) -> Dict[str, Any]:
    """
    Ejecuta el router OCR de Evalia.

    Orden:

    1. Mistral OCR.
    2. EasyOCR.
    3. Tesseract.
    4. Fallback manual.

    La función siempre devuelve un diccionario y nunca interrumpe Evalia
    solamente porque un motor OCR no esté disponible.
    """
    started = time.time()
    source_path = _path(path)
    attempts = []

    if not source_path.exists():
        return {
            "text": "",
            "confidence": 0.0,
            "engine": "file_not_found",
            "seconds": round(
                time.time() - started,
                3,
            ),
            "message": (
                f"No se encontró la imagen: {source_path}"
            ),
            "attempts": [],
        }

    mistral_result = run_mistral_ocr_on_file(
        source_path
    )

    if mistral_result is not None:
        attempts.append(mistral_result)

        if str(
            mistral_result.get(
                "text",
                "",
            )
        ).strip():
            mistral_result["attempts"] = attempts
            return mistral_result

    use_easyocr = os.getenv(
        "EVALIA_USE_EASYOCR",
        "0",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if use_easyocr:
        easyocr_result = run_easyocr_on_image(
            source_path
        )

        if easyocr_result is not None:
            attempts.append(easyocr_result)

            if str(
                easyocr_result.get(
                    "text",
                    "",
                )
            ).strip():
                easyocr_result["attempts"] = attempts
                return easyocr_result

    use_tesseract = os.getenv(
        "EVALIA_USE_TESSERACT",
        "1",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if use_tesseract:
        tesseract_result = (
            run_tesseract_ocr_on_image(
                source_path
            )
        )

        if tesseract_result is not None:
            attempts.append(tesseract_result)

            if str(
                tesseract_result.get(
                    "text",
                    "",
                )
            ).strip():
                tesseract_result["attempts"] = attempts
                return tesseract_result

    detail = " | ".join(
        (
            f"{attempt.get('engine')}: "
            f"{attempt.get('message')}"
        )
        for attempt in attempts
    )

    OCR_STATUS.update(
        {
            "engine": "manual_fallback",
            "available": False,
            "message": (
                "OCR automático no produjo texto útil; "
                "se habilita revisión/transcripción manual."
            ),
        }
    )

    return {
        "text": "",
        "confidence": 0.0,
        "engine": "manual_fallback",
        "seconds": round(
            time.time() - started,
            3,
        ),
        "message": (
            "OCR automático no produjo texto útil. "
            + (
                detail
                if detail
                else (
                    "Configura MISTRAL_API_KEY "
                    "o revisa la calidad de imagen."
                )
            )
        ),
        "attempts": attempts,
    }
