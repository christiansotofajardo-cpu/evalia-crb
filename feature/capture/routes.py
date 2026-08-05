"""
Rutas móviles de Smart Capture para Evalia.

Este módulo:
- abre la cámara trasera del teléfono;
- recibe una fotografía;
- ejecuta CaptureAssistant.inspect_bytes();
- muestra calidad, páginas detectadas, organización y vista previa;
- mantiene separado Smart Capture del main.py.

Integración en main.py:

    from feature.capture.routes import register_capture_routes
    register_capture_routes(app)

Importante:
- register_capture_routes(app) debe ejecutarse después de crear:
      app = FastAPI(...)
"""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import shutil
import tempfile
import time
import uuid
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from .capture_assistant import CaptureAssistant
from feature.ocr.service import run_ocr_on_image


MAX_CAPTURE_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/tiff",
    "image/bmp",
}

RUBRIC_EXTENSIONS = {".json", ".xlsx", ".xls"}


def _rubric_directories() -> list[Path]:
    """Devuelve ubicaciones probables de la carpeta de rúbricas sin importar el cwd."""
    module_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "rubrics",
        module_path.parents[2] / "rubrics",
        module_path.parents[1] / "rubrics",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _available_rubrics() -> list[Dict[str, str]]:
    """Lista rúbricas guardadas que Smart Capture puede seleccionar."""
    found: Dict[str, Dict[str, str]] = {}
    for directory in _rubric_directories():
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in RUBRIC_EXTENSIONS:
                continue
            filename = path.name
            label = path.stem.replace("_", " ").replace("-", " ").strip()
            found.setdefault(filename, {
                "filename": filename,
                "label": label or filename,
            })
    return sorted(found.values(), key=lambda item: item["label"].lower())


def _valid_rubric_filename(filename: str) -> bool:
    selected = Path(str(filename or "")).name
    return bool(selected) and any(
        item["filename"] == selected for item in _available_rubrics()
    )

_capture_assistant: Optional[CaptureAssistant] = None

# Sesiones temporales de captura. Conservan los recortes entre la revisión
# docente y la confirmación que inicia el OCR. Se eliminan automáticamente.
_CAPTURE_SESSIONS: Dict[str, Dict[str, Any]] = {}
_CAPTURE_SESSION_TTL_SECONDS = 15 * 60

# Sesiones temporales entre OCR, revisión docente y evaluación final.
_EVALUATION_SESSIONS: Dict[str, Dict[str, Any]] = {}
_EVALUATION_SESSION_TTL_SECONDS = 30 * 60


def _cleanup_capture_sessions() -> None:
    """Elimina sesiones antiguas para no acumular imágenes en memoria."""
    now = time.time()
    expired = [
        token
        for token, payload in _CAPTURE_SESSIONS.items()
        if now - float(payload.get("created_at", 0)) > _CAPTURE_SESSION_TTL_SECONDS
    ]
    for token in expired:
        _CAPTURE_SESSIONS.pop(token, None)


def _cleanup_evaluation_sessions() -> None:
    """Elimina sesiones de revisión/evaluación que superaron su tiempo útil."""
    now = time.time()
    expired = [
        token
        for token, payload in _EVALUATION_SESSIONS.items()
        if now - float(payload.get("created_at", 0)) > _EVALUATION_SESSION_TTL_SECONDS
    ]
    for token in expired:
        _EVALUATION_SESSIONS.pop(token, None)


def _store_evaluation_session(
    *,
    filename: str,
    exam_name: str,
    rubric_filename: str,
    rubric: Dict[str, Any],
    students: list,
) -> str:
    _cleanup_evaluation_sessions()
    token = uuid.uuid4().hex
    _EVALUATION_SESSIONS[token] = {
        "created_at": time.time(),
        "filename": str(filename or "captura"),
        "exam_name": str(exam_name or "Evaluación"),
        "rubric_filename": Path(str(rubric_filename or "")).name,
        "rubric": rubric,
        "students": students,
    }
    return token


def _evaluation_services(app: FastAPI) -> Dict[str, Any]:
    """Obtiene el puente de servicios registrado por main.py."""
    services = getattr(getattr(app, "state", None), "evalia_services", None)
    if not isinstance(services, dict):
        raise RuntimeError(
            "El puente de evaluación no está disponible. "
            "Verifica que main.py registre app.state.evalia_services "
            "antes de llamar register_capture_routes(app)."
        )
    return services


def _required_service(services: Dict[str, Any], name: str):
    service = services.get(name)
    if not callable(service):
        raise RuntimeError(f"Falta el servicio de evaluación requerido: {name}")
    return service


def _store_capture_session(
    result: Any,
    filename: str,
    exam_name: str,
    rubric_filename: str,
) -> str:
    _cleanup_capture_sessions()
    token = uuid.uuid4().hex
    _CAPTURE_SESSIONS[token] = {
        "created_at": time.time(),
        "result": result,
        "filename": filename,
        "exam_name": str(exam_name or "").strip(),
        "rubric_filename": Path(str(rubric_filename or "")).name,
    }
    return token


def _extract_crop(page: Any) -> Optional[np.ndarray]:
    """Obtiene y normaliza el recorte de una página detectada."""
    crop = _get_value(
        page,
        "cropped_image",
        "crop",
        "page_image",
        "image",
        default=None,
    )
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        return None

    image = crop
    if image.dtype != np.uint8:
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.min(image)) >= 0.0 and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        return None

    return np.ascontiguousarray(image)


def _get_capture_assistant() -> CaptureAssistant:
    """Crea una sola instancia del asistente durante la vida de la aplicación."""
    global _capture_assistant
    if _capture_assistant is None:
        _capture_assistant = CaptureAssistant()
    return _capture_assistant


def _get_value(obj: Any, *names: str, default: Any = None) -> Any:
    """Lee una clave desde dict u objeto sin depender de una estructura rígida."""
    if obj is None:
        return default

    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)

        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _json_safe(value: Any, seen: Optional[set] = None) -> Any:
    """Convierte resultados internos en estructuras seguras para depuración."""
    if seen is None:
        seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    value_id = id(value)
    if value_id in seen:
        return "[referencia circular]"

    if isinstance(value, (dict, list, tuple, set)) or hasattr(value, "__dict__"):
        seen.add(value_id)

    if isinstance(value, dict):
        return {str(k): _json_safe(v, seen) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, seen) for v in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(), seen)
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict(), seen)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _json_safe(v, seen)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass

    return str(value)


def _percent(value: Any) -> str:
    try:
        number = float(value)
        if 0 <= number <= 1:
            number *= 100
        return f"{round(number)}%"
    except Exception:
        return "—"


def _preview_to_data_url(preview: Any) -> Optional[str]:
    """
    Convierte la salida del PreviewGenerator en una data URL compatible con <img>.

    Acepta:
    - numpy.ndarray de OpenCV;
    - data URL;
    - base64 puro;
    - bytes o bytearray;
    - ruta local;
    - dict/objeto con data_url, base64, image_bytes, path o preview.
    """
    if preview is None:
        return None

    # PreviewGenerator.render() devuelve normalmente un ndarray BGR.
    if isinstance(preview, np.ndarray):
        if preview.size == 0:
            return None

        image = preview

        if image.dtype != np.uint8:
            image = np.nan_to_num(
                image,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )
            if float(np.min(image)) >= 0.0 and float(np.max(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            return None

        success, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(image),
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if not success:
            return None

        encoded_base64 = base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
        return f"data:image/jpeg;base64,{encoded_base64}"

    nested = _get_value(
        preview,
        "data_url",
        "image_data_url",
        "preview_data_url",
        "base64",
        "image_base64",
        "preview_base64",
        "image_bytes",
        "bytes",
        "path",
        "file_path",
        "preview_path",
        "preview",
        default=None,
    )

    if nested is not None and nested is not preview:
        return _preview_to_data_url(nested)

    if isinstance(preview, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(preview)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    if isinstance(preview, Path):
        preview = str(preview)

    if isinstance(preview, str):
        text = preview.strip()
        if not text:
            return None

        if text.startswith("data:image/"):
            return text

        path = Path(text)
        if path.exists() and path.is_file():
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"

        # Base64 razonablemente largo.
        if len(text) > 200:
            try:
                base64.b64decode(text, validate=True)
                return f"data:image/jpeg;base64,{text}"
            except Exception:
                return None

    return None


def _quality_state(result: Any) -> Dict[str, Any]:
    quality = _get_value(
        result,
        "quality",
        "quality_result",
        "quality_analysis",
        "quality_report",
        default=None,
    )

    score = _get_value(
        quality,
        "score",
        "quality_score",
        "confidence",
        "overall_score",
        default=None,
    )
    if score is None:
        score = _get_value(
            result,
            "quality_score",
            "quality_confidence",
            default=None,
        )

    status = _get_value(
        quality,
        "status",
        "state",
        "level",
        "label",
        default="",
    )
    message = _get_value(
        quality,
        "message",
        "summary",
        "recommendation",
        "feedback",
        default="",
    )

    issues = _as_list(
        _get_value(
            quality,
            "issues",
            "warnings",
            "problems",
            "recommendations",
            default=[],
        )
    )

    numeric = None
    try:
        numeric = float(score)
        if numeric > 1:
            numeric /= 100
    except Exception:
        pass

    if numeric is not None:
        if numeric >= 0.78:
            color = "green"
            default_status = "Buena"
        elif numeric >= 0.48:
            color = "yellow"
            default_status = "Aceptable"
        else:
            color = "red"
            default_status = "Repetir fotografía"
    else:
        normalized = str(status).lower()
        if any(word in normalized for word in ("good", "ok", "alta", "buena", "green", "verde")):
            color = "green"
        elif any(word in normalized for word in ("bad", "low", "mala", "red", "rojo")):
            color = "red"
        else:
            color = "yellow"
        default_status = "Analizada"

    return {
        "score": score,
        "status": str(status or default_status),
        "message": str(message or ""),
        "issues": [str(x) for x in issues if str(x).strip()],
        "color": color,
    }


def _detected_pages(result: Any) -> list:
    pages = _get_value(
        result,
        "pages",
        "detected_pages",
        "page_detections",
        default=None,
    )

    if pages is None:
        detection = _get_value(
            result,
            "detection",
            "page_detection",
            "detector_result",
            default=None,
        )
        pages = _get_value(
            detection,
            "pages",
            "detected_pages",
            default=[],
        )

    return _as_list(pages)


def _organization(result: Any) -> Any:
    return _get_value(
        result,
        "organization",
        "students",
        "student_groups",
        "organized_pages",
        "page_organization",
        "groups",
        default=None,
    )


def _render_page_cards(pages: Iterable[Any]) -> str:
    cards = []

    for index, page in enumerate(pages, start=1):
        page_number = _get_value(
            page,
            "page_number",
            "number",
            "page",
            "index",
            default=index,
        )
        confidence = _get_value(
            page,
            "confidence",
            "score",
            "detection_confidence",
            default=None,
        )
        student = _get_value(
            page,
            "student_id",
            "student",
            "student_name",
            "group_id",
            default="",
        )
        bbox = _get_value(
            page,
            "bbox",
            "bounding_box",
            "corners",
            "polygon",
            default=None,
        )

        details = []
        if confidence is not None:
            details.append(f"Confianza: {_percent(confidence)}")
        if student:
            details.append(f"Estudiante/grupo: {escape(str(student))}")
        if bbox is not None:
            details.append("Área delimitada correctamente")

        cards.append(
            f"""
            <article class="mini-card">
                <div class="mini-number">{escape(str(page_number))}</div>
                <div>
                    <strong>Hoja detectada {index}</strong>
                    <p>{'<br>'.join(details) if details else 'Lista para revisión docente.'}</p>
                </div>
            </article>
            """
        )

    if not cards:
        return """
        <div class="notice warning">
            No se detectaron hojas con suficiente claridad. Prueba nuevamente con
            mejor iluminación y toda la superficie dentro del encuadre.
        </div>
        """

    return "\n".join(cards)


def _render_organization(organization: Any) -> str:
    if organization is None:
        return """
        <div class="notice neutral">
            La organización automática no entregó grupos visibles. Revisa la
            vista previa antes de continuar.
        </div>
        """

    groups = _get_value(
        organization,
        "groups",
        "student_groups",
        "organized_groups",
        default=None,
    )
    incomplete = _as_list(
        _get_value(
            organization,
            "incomplete_groups",
            "incomplete",
            "warnings",
            default=[],
        )
    )

    if groups is None and isinstance(organization, (list, tuple)):
        groups = organization

    group_rows = []
    for index, group in enumerate(_as_list(groups), start=1):
        student_number = _get_value(
            group,
            "student_number",
            default=index,
        )

        group_name = _get_value(
            group,
            "student_name",
            "student_id",
            "group_id",
            "name",
            default=None,
        )

        if group_name in (None, "", "None"):
            group_name = f"Estudiante {student_number}"
        group_pages = _as_list(
            _get_value(
                group,
                "pages",
                "page_ids",
                "items",
                default=[],
            )
        )
        group_rows.append(
            f"""
            <div class="group-row">
                <strong>{escape(str(group_name))}</strong>
                <span>{len(group_pages)} hoja(s)</span>
            </div>
            """
        )

    if not group_rows:
        safe_summary = escape(
            json.dumps(
                _json_safe(organization),
                ensure_ascii=False,
            )[:800]
        )
        group_rows.append(
            f"""
            <details>
                <summary>Ver resultado de organización</summary>
                <pre>{safe_summary}</pre>
            </details>
            """
        )

    incomplete_html = ""
    if incomplete:
        items = "".join(f"<li>{escape(str(x))}</li>" for x in incomplete)
        incomplete_html = f"""
        <div class="notice warning">
            <strong>Revisión necesaria</strong>
            <ul>{items}</ul>
        </div>
        """

    return "\n".join(group_rows) + incomplete_html


def _mobile_css() -> str:
    return """
    <style>
        :root {
            --blue: #1d4ed8;
            --blue-dark: #1e3a8a;
            --text: #172033;
            --muted: #64748b;
            --border: #dbe4f0;
            --background: #f4f7fb;
            --card: #ffffff;
            --green-bg: #ecfdf5;
            --green: #047857;
            --yellow-bg: #fffbeb;
            --yellow: #a16207;
            --red-bg: #fef2f2;
            --red: #b91c1c;
        }

        * { box-sizing: border-box; }

        html, body {
            margin: 0;
            padding: 0;
            background: var(--background);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, Helvetica, Arial, sans-serif;
        }

        body { min-height: 100vh; }

        .mobile-shell {
            width: min(100%, 760px);
            margin: 0 auto;
            padding: 16px;
        }

        .brand {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 4px 0 18px;
        }

        .brand strong {
            font-size: 1.05rem;
            color: var(--blue-dark);
        }

        .badge {
            padding: 7px 10px;
            border-radius: 999px;
            background: #dbeafe;
            color: var(--blue-dark);
            font-size: .78rem;
            font-weight: 800;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 20px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, .07);
            margin-bottom: 16px;
        }

        h1 {
            margin: 0 0 8px;
            font-size: clamp(1.75rem, 8vw, 2.5rem);
            line-height: 1.04;
        }

        h2 {
            margin: 0 0 12px;
            font-size: 1.15rem;
        }

        p {
            color: var(--muted);
            line-height: 1.5;
        }

        .steps {
            display: grid;
            gap: 8px;
            margin: 18px 0;
        }

        .step {
            display: flex;
            gap: 10px;
            align-items: center;
            padding: 10px 12px;
            border-radius: 13px;
            background: #f8fafc;
        }

        .step span {
            display: grid;
            place-items: center;
            min-width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #dbeafe;
            color: var(--blue-dark);
            font-weight: 900;
        }

        .field-label {
            display: block;
            margin: 14px 0 7px;
            color: var(--text);
            font-size: .9rem;
            font-weight: 800;
        }

        .text-input, .select-input {
            width: 100%;
            min-height: 52px;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 14px;
            background: white;
            color: var(--text);
            font-size: 1rem;
        }

        .text-input:focus, .select-input:focus {
            outline: 3px solid #dbeafe;
            border-color: var(--blue);
        }

        .session-summary {
            display: grid;
            gap: 8px;
            padding: 13px;
            margin: 12px 0;
            border-radius: 14px;
            background: #eff6ff;
            color: var(--blue-dark);
        }

        .camera-input {
            position: absolute;
            width: 1px;
            height: 1px;
            overflow: hidden;
            opacity: 0;
            pointer-events: none;
        }

        .primary-button, .secondary-button {
            display: block;
            width: 100%;
            min-height: 56px;
            border: 0;
            border-radius: 16px;
            padding: 15px 18px;
            font-size: 1.02rem;
            font-weight: 900;
            text-align: center;
            text-decoration: none;
            cursor: pointer;
        }

        .primary-button {
            background: var(--blue);
            color: white;
            box-shadow: 0 8px 20px rgba(29, 78, 216, .24);
        }

        .primary-button:disabled {
            opacity: .55;
            cursor: wait;
        }

        .secondary-button {
            background: #e2e8f0;
            color: #1e293b;
            margin-top: 10px;
        }

        .file-name {
            margin: 12px 0 0;
            min-height: 24px;
            font-size: .9rem;
            color: var(--muted);
            text-align: center;
        }

        .hint {
            margin-top: 14px;
            font-size: .87rem;
            text-align: center;
        }

        .preview {
            width: 100%;
            display: block;
            border-radius: 16px;
            border: 1px solid var(--border);
            background: #eef2f7;
        }

        .quality {
            border-radius: 16px;
            padding: 14px;
            margin-bottom: 14px;
        }

        .quality.green { background: var(--green-bg); color: var(--green); }
        .quality.yellow { background: var(--yellow-bg); color: var(--yellow); }
        .quality.red { background: var(--red-bg); color: var(--red); }

        .quality p {
            color: inherit;
            margin: 6px 0 0;
        }

        .mini-card {
            display: flex;
            gap: 12px;
            align-items: center;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 15px;
            margin-bottom: 9px;
        }

        .mini-number {
            display: grid;
            place-items: center;
            min-width: 42px;
            height: 42px;
            border-radius: 12px;
            background: #dbeafe;
            color: var(--blue-dark);
            font-weight: 900;
        }

        .mini-card p {
            margin: 3px 0 0;
            font-size: .88rem;
        }

        .group-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }

        .notice {
            padding: 13px;
            border-radius: 14px;
            line-height: 1.45;
            margin-top: 12px;
        }

        .notice.warning { background: var(--yellow-bg); color: var(--yellow); }
        .notice.neutral { background: #f1f5f9; color: #475569; }

        ul { margin-bottom: 0; }

        details { margin-top: 8px; }

        pre {
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: .75rem;
            background: #0f172a;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 12px;
        }

        .loading {
            display: none;
            text-align: center;
            color: var(--blue-dark);
            font-weight: 800;
            margin-top: 14px;
        }

        @media (min-width: 700px) {
            .mobile-shell { padding: 28px; }
            .card { padding: 28px; }
        }
    </style>
    """


def _capture_home_html(rubrics: list[Dict[str, str]]) -> str:
    if rubrics:
        rubric_options = "".join(
            f'<option value="{escape(item["filename"], quote=True)}">'
            f'{escape(item["label"])} · {escape(item["filename"])}</option>'
            for item in rubrics
        )
        rubric_control = f"""
        <label class="field-label" for="rubricFilename">Rúbrica</label>
        <select id="rubricFilename" class="select-input" name="rubric_filename" required>
            <option value="">Selecciona una rúbrica guardada</option>
            {rubric_options}
        </select>
        """
        rubric_notice = ""
        can_submit = "true"
    else:
        rubric_control = """
        <label class="field-label">Rúbrica</label>
        <div class="notice warning">
            No encontramos rúbricas guardadas en la carpeta <strong>rubrics</strong>.
            Sube o registra una rúbrica desde Evalia antes de iniciar Smart Capture.
        </div>
        """
        rubric_notice = '<a class="secondary-button" href="/">Ir a Evalia</a>'
        can_submit = "false"

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <meta name="theme-color" content="#1d4ed8">
        <title>Captura inteligente · Evalia</title>
        {_mobile_css()}
    </head>
    <body>
        <main class="mobile-shell">
            <div class="brand">
                <strong>Evalia</strong>
                <span class="badge">Smart Capture</span>
            </div>

            <section class="card">
                <h1>Prepara la corrección</h1>
                <p>
                    Indica la evaluación y elige una rúbrica guardada. Evalia
                    mantendrá esta selección durante toda la captura.
                </p>

                <form id="captureForm" action="/captura/analizar" method="post" enctype="multipart/form-data">
                    <label class="field-label" for="examName">Evaluación</label>
                    <input
                        id="examName"
                        class="text-input"
                        type="text"
                        name="exam_name"
                        maxlength="120"
                        placeholder="Ej.: Certamen 1 de Psicolingüística"
                        required
                    >

                    {rubric_control}

                    <div class="steps">
                        <div class="step"><span>1</span> Ilumina bien las hojas.</div>
                        <div class="step"><span>2</span> Incluye todos los bordes.</div>
                        <div class="step"><span>3</span> Evita sombras y movimiento.</div>
                    </div>

                    <input
                        id="capturePhoto"
                        class="camera-input"
                        type="file"
                        name="capture_photo"
                        accept="image/*"
                        capture="environment"
                        required
                    >

                    <label class="primary-button" for="capturePhoto">📷 Tomar fotografía</label>
                    <p id="fileName" class="file-name">Aún no has tomado una fotografía.</p>

                    <button id="analyzeButton" class="primary-button" type="submit" disabled>
                        Analizar fotografía
                    </button>

                    <div id="loading" class="loading">
                        Analizando calidad y detectando páginas…
                    </div>
                </form>

                <p class="hint">
                    En el teléfono se abrirá preferentemente la cámara trasera.
                    También puedes seleccionar una imagen ya existente.
                </p>
                {rubric_notice}
            </section>

            <a class="secondary-button" href="/ocr">Volver al OCR tradicional</a>
        </main>

        <script>
            const input = document.getElementById("capturePhoto");
            const fileName = document.getElementById("fileName");
            const button = document.getElementById("analyzeButton");
            const form = document.getElementById("captureForm");
            const loading = document.getElementById("loading");
            const examName = document.getElementById("examName");
            const rubricFilename = document.getElementById("rubricFilename");
            const rubricsAvailable = {can_submit};

            function updateButtonState() {{
                const hasFile = Boolean(input.files && input.files[0]);
                const hasExam = Boolean(examName.value.trim());
                const hasRubric = Boolean(rubricFilename && rubricFilename.value);
                button.disabled = !(rubricsAvailable && hasFile && hasExam && hasRubric);
            }}

            input.addEventListener("change", () => {{
                const file = input.files && input.files[0];
                if (!file) {{
                    fileName.textContent = "Aún no has tomado una fotografía.";
                    updateButtonState();
                    return;
                }}
                const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
                fileName.textContent = `${{file.name}} · ${{sizeMb}} MB`;
                updateButtonState();
            }});
            examName.addEventListener("input", updateButtonState);
            if (rubricFilename) rubricFilename.addEventListener("change", updateButtonState);

            form.addEventListener("submit", () => {{
                button.disabled = true;
                button.textContent = "Procesando…";
                loading.style.display = "block";
            }});
            updateButtonState();
        </script>
    </body>
    </html>
    """

def _result_html(
    result: Any,
    filename: str,
    capture_token: str,
    exam_name: str,
    rubric_filename: str,
) -> str:
    quality = _quality_state(result)
    pages = _detected_pages(result)
    organization = _organization(result)

    preview = _get_value(
        result,
        "preview",
        "preview_image",
        "preview_result",
        "annotated_image",
        default=None,
    )
    preview_url = _preview_to_data_url(preview)

    preview_html = (
        f'<img class="preview" src="{escape(preview_url, quote=True)}" '
        f'alt="Vista previa de las páginas detectadas">'
        if preview_url
        else """
        <div class="notice neutral">
            El análisis terminó, pero el generador no entregó una imagen de
            vista previa compatible. Los resultados técnicos aparecen abajo.
        </div>
        """
    )

    message_html = (
        f"<p>{escape(quality['message'])}</p>"
        if quality["message"]
        else ""
    )

    issues_html = ""
    if quality["issues"]:
        issues = "".join(
            f"<li>{escape(issue)}</li>"
            for issue in quality["issues"]
        )
        issues_html = f"<ul>{issues}</ul>"

    score_text = (
        f" · {_percent(quality['score'])}"
        if quality["score"] is not None
        else ""
    )

    raw_json = escape(
        json.dumps(
            _json_safe(result),
            ensure_ascii=False,
            indent=2,
        )[:12000]
    )

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1, viewport-fit=cover"
        >
        <meta name="theme-color" content="#1d4ed8">
        <title>Resultado de captura · Evalia</title>
        {_mobile_css()}
    </head>
    <body>
        <main class="mobile-shell">
            <div class="brand">
                <strong>Evalia</strong>
                <span class="badge">Revisión docente</span>
            </div>

            <section class="card">
                <h1>Revisa la captura</h1>
                <p>{escape(filename)}</p>
                <div class="session-summary">
                    <div><strong>Evaluación:</strong> {escape(exam_name)}</div>
                    <div><strong>Rúbrica:</strong> {escape(rubric_filename)}</div>
                </div>

                <div class="quality {quality['color']}">
                    <strong>
                        Calidad: {escape(quality['status'])}{score_text}
                    </strong>
                    {message_html}
                    {issues_html}
                </div>

                {preview_html}
            </section>

            <section class="card">
                <h2>Páginas detectadas</h2>
                {_render_page_cards(pages)}
            </section>

            <section class="card">
                <h2>Organización propuesta</h2>
                {_render_organization(organization)}
            </section>

            <section class="card">
                <h2>Confirmación docente</h2>
                <p>
                    Confirma la organización para enviar cada hoja recortada al
                    OCR de Evalia. El procesamiento puede tardar algunos segundos.
                </p>

                <form id="ocrForm" action="/captura/procesar-ocr" method="post">
                    <input type="hidden" name="capture_token"
                           value="{escape(capture_token, quote=True)}">
                    <button id="ocrButton" class="primary-button" type="submit">
                        Confirmar y ejecutar OCR
                    </button>
                    <div id="ocrLoading" class="loading">
                        Procesando las hojas con OCR… No cierres esta ventana.
                    </div>
                </form>

                <a class="secondary-button" href="/captura">
                    Tomar otra fotografía
                </a>
            </section>

            <details class="card">
                <summary>Información técnica del análisis</summary>
                <pre>{raw_json}</pre>
            </details>
        </main>
        <script>
            const ocrForm = document.getElementById("ocrForm");
            const ocrButton = document.getElementById("ocrButton");
            const ocrLoading = document.getElementById("ocrLoading");
            if (ocrForm) {{
                ocrForm.addEventListener("submit", () => {{
                    ocrButton.disabled = true;
                    ocrButton.textContent = "Procesando OCR…";
                    ocrLoading.style.display = "block";
                }});
            }}
        </script>
    </body>
    </html>
    """



def _ocr_review_html(
    *,
    filename: str,
    students: list,
    exam_name: str,
    rubric_filename: str,
    rubric: Dict[str, Any],
    evaluation_token: str,
    display_item_type,
) -> str:
    student_cards = []
    questions = rubric.get("questions", []) or []

    for student_index, student in enumerate(students):
        student_number = student.get("student_number", student_index + 1)
        page_number = student.get("page_number", student_index + 1)
        ocr = student.get("ocr", {}) or {}
        raw_text = str(ocr.get("text", "") or "").strip()
        segmentation = student.get("segmentation", {}) or {}
        segments = student.get("segments", {}) or {}
        error = str(student.get("error", "") or "")
        seg_conf = segmentation.get("confidence")
        seg_mode = segmentation.get("mode", "desconocido")

        question_fields = []
        for question in questions:
            qid = str(question.get("id", ""))
            prompt = str(question.get("prompt", "") or "")
            item_label = display_item_type(question.get("item_type", ""))
            answer = str(segments.get(qid, "") or "")
            field_name = f"answer__{student_index}__{qid}"
            question_fields.append(f"""
            <div class="mini-card" style="display:block;">
                <strong>{escape(qid)} · {escape(str(item_label))} · {escape(str(question.get("max_score", "")))} pts</strong>
                {f'<p>{escape(prompt)}</p>' if prompt else ''}
                <textarea
                    class="text-input"
                    name="{escape(field_name, quote=True)}"
                    rows="5"
                    style="margin-top:10px;min-height:120px;"
                >{escape(answer)}</textarea>
            </div>
            """)

        state_class = "green" if raw_text and not error else "yellow"
        seg_text = (
            f"Segmentación: {escape(str(seg_mode))}"
            + (f" · Confianza: {_percent(seg_conf)}" if seg_conf is not None else "")
        )

        student_cards.append(f"""
        <section class="card">
            <h2>Estudiante {escape(str(student_number))} · Hoja {escape(str(page_number))}</h2>
            <div class="quality {state_class}">
                <strong>{'OCR y segmentación preparados' if raw_text else 'Revisión manual necesaria'}</strong>
                <p>{seg_text}</p>
                {f'<p>{escape(error)}</p>' if error else ''}
            </div>

            <label class="field-label" for="studentName{student_index}">Nombre del estudiante</label>
            <input
                id="studentName{student_index}"
                class="text-input"
                type="text"
                name="student_name__{student_index}"
                value="{escape(str(student.get('student_name', '') or ''), quote=True)}"
                placeholder="Nombre detectado o corregido por el docente"
            >

            <details>
                <summary>Ver texto OCR bruto</summary>
                <textarea
                    class="text-input"
                    name="raw_text__{student_index}"
                    rows="7"
                    style="margin-top:10px;min-height:150px;"
                >{escape(raw_text)}</textarea>
            </details>

            <h2 style="margin-top:18px;">Respuestas por pregunta</h2>
            {''.join(question_fields)}
        </section>
        """)

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <meta name="theme-color" content="#1d4ed8">
        <title>Revisión OCR · Evalia</title>
        {_mobile_css()}
    </head>
    <body>
        <main class="mobile-shell">
            <div class="brand">
                <strong>Evalia</strong>
                <span class="badge">Revisión antes de evaluar</span>
            </div>

            <section class="card">
                <h1>Revisa las respuestas</h1>
                <p>{escape(filename)}</p>
                <div class="session-summary">
                    <div><strong>Evaluación:</strong> {escape(exam_name)}</div>
                    <div><strong>Rúbrica:</strong> {escape(rubric_filename)}</div>
                    <div><strong>Estudiantes/hojas:</strong> {len(students)}</div>
                </div>
                <div class="notice neutral">
                    Corrige nombres o respuestas cuando el OCR no haya quedado exacto.
                    Evalia no calificará automáticamente un texto vacío.
                </div>
            </section>

            <form id="evaluationForm" action="/captura/evaluar" method="post">
                <input type="hidden" name="evaluation_token"
                       value="{escape(evaluation_token, quote=True)}">
                {''.join(student_cards)}
                <section class="card">
                    <button id="evaluateButton" class="primary-button" type="submit">
                        Confirmar y evaluar
                    </button>
                    <div id="evaluateLoading" class="loading">
                        Aplicando la rúbrica y generando resultados…
                    </div>
                    <a class="secondary-button" href="/captura">Cancelar y comenzar otra captura</a>
                </section>
            </form>
        </main>
        <script>
            const form = document.getElementById("evaluationForm");
            const button = document.getElementById("evaluateButton");
            const loading = document.getElementById("evaluateLoading");
            form.addEventListener("submit", () => {{
                button.disabled = true;
                button.textContent = "Evaluando…";
                loading.style.display = "block";
            }});
        </script>
    </body>
    </html>
    """


def _evaluation_results_html(
    *,
    filename: str,
    exam_name: str,
    rubric_filename: str,
    results: list,
) -> str:
    cards = []
    valid_results = [item for item in results if not item.get("error")]

    for index, item in enumerate(results, start=1):
        name = str(item.get("student_name") or f"Estudiante {index}")
        error = str(item.get("error", "") or "")

        if error:
            cards.append(f"""
            <section class="card">
                <h2>{escape(name)}</h2>
                <div class="quality yellow">
                    <strong>No fue posible evaluar</strong>
                    <p>{escape(error)}</p>
                </div>
            </section>
            """)
            continue

        question_rows = []
        for question in item.get("questions", []) or []:
            question_rows.append(f"""
            <div class="mini-card" style="display:block;">
                <strong>
                    {escape(str(question.get('question_id', '')))}
                    · {escape(str(question.get('score', 0)))} /
                    {escape(str(question.get('max_score', 0)))} pts
                </strong>
                <p><strong>Estado:</strong> {escape(str(question.get('status', '')))}
                   · <strong>Confianza:</strong> {_percent(question.get('confidence'))}</p>
                <p>{escape(str(question.get('feedback', '') or ''))}</p>
                <details>
                    <summary>Respuesta evaluada</summary>
                    <pre>{escape(str(question.get('answer', '') or ''))}</pre>
                </details>
            </div>
            """)

        cards.append(f"""
        <section class="card">
            <h2>{escape(name)}</h2>
            <div class="quality green">
                <strong>
                    {escape(str(item.get('total_score', 0)))} /
                    {escape(str(item.get('max_score', 0)))} pts
                </strong>
                <p>
                    {escape(str(item.get('percentage', 0)))}%
                    · {escape(str(item.get('performance_level', '')))}
                </p>
            </div>
            {''.join(question_rows)}
        </section>
        """)

    average = 0.0
    if valid_results:
        average = round(
            sum(float(item.get("percentage", 0) or 0) for item in valid_results)
            / len(valid_results),
            2,
        )

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <meta name="theme-color" content="#1d4ed8">
        <title>Resultados · Evalia</title>
        {_mobile_css()}
    </head>
    <body>
        <main class="mobile-shell">
            <div class="brand">
                <strong>Evalia</strong>
                <span class="badge">Evaluación completada</span>
            </div>
            <section class="card">
                <h1>Resultados finales</h1>
                <p>{escape(filename)}</p>
                <div class="session-summary">
                    <div><strong>Evaluación:</strong> {escape(exam_name)}</div>
                    <div><strong>Rúbrica:</strong> {escape(rubric_filename)}</div>
                    <div><strong>Evaluados:</strong> {len(valid_results)} de {len(results)}</div>
                    <div><strong>Promedio del grupo:</strong> {average}%</div>
                </div>
            </section>
            {''.join(cards)}
            <section class="card">
                <a class="primary-button" href="/captura">Procesar otra fotografía</a>
                <a class="secondary-button" href="/">Volver a Evalia</a>
            </section>
        </main>
    </body>
    </html>
    """


def _error_html(title: str, message: str, detail: str = "") -> str:
    detail_html = (
        f"<details><summary>Detalle técnico</summary>"
        f"<pre>{escape(detail)}</pre></details>"
        if detail
        else ""
    )

    return f"""
    <!doctype html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1, viewport-fit=cover"
        >
        <title>Error de captura · Evalia</title>
        {_mobile_css()}
    </head>
    <body>
        <main class="mobile-shell">
            <div class="brand">
                <strong>Evalia</strong>
                <span class="badge">Smart Capture</span>
            </div>

            <section class="card">
                <h1>{escape(title)}</h1>
                <div class="notice warning">{escape(message)}</div>
                {detail_html}
                <a class="secondary-button" href="/captura">
                    Volver a intentarlo
                </a>
            </section>
        </main>
    </body>
    </html>
    """


def register_capture_routes(app: FastAPI) -> None:
    """
    Registra las rutas una sola vez.

    Uso en main.py:
        from feature.capture.routes import register_capture_routes
        register_capture_routes(app)
    """

    existing_paths = {
        getattr(route, "path", None)
        for route in getattr(app, "routes", [])
    }

    if "/captura" not in existing_paths:

        @app.get("/captura", response_class=HTMLResponse)
        async def capture_home() -> HTMLResponse:
            return HTMLResponse(_capture_home_html(_available_rubrics()))

    if "/captura/analizar" not in existing_paths:

        @app.post("/captura/analizar", response_class=HTMLResponse)
        async def analyze_capture(
            capture_photo: UploadFile = File(...),
            exam_name: str = Form(...),
            rubric_filename: str = Form(...),
        ) -> HTMLResponse:
            try:
                exam_name = str(exam_name or "").strip()
                rubric_filename = Path(str(rubric_filename or "")).name

                if not exam_name:
                    return HTMLResponse(
                        _error_html(
                            "Falta la evaluación",
                            "Escribe el nombre de la evaluación antes de tomar la fotografía.",
                        ),
                        status_code=400,
                    )

                if not _valid_rubric_filename(rubric_filename):
                    return HTMLResponse(
                        _error_html(
                            "Rúbrica no disponible",
                            "La rúbrica seleccionada ya no existe o no está registrada. Vuelve a seleccionarla.",
                        ),
                        status_code=400,
                    )

                if not capture_photo or not capture_photo.filename:
                    return HTMLResponse(
                        _error_html(
                            "Falta la fotografía",
                            "Toma o selecciona una imagen antes de continuar.",
                        ),
                        status_code=400,
                    )

                content_type = (
                    capture_photo.content_type or ""
                ).lower().strip()

                if content_type and content_type not in ALLOWED_IMAGE_TYPES:
                    return HTMLResponse(
                        _error_html(
                            "Formato no compatible",
                            "Usa una fotografía JPG, PNG, WebP, HEIC, TIFF o BMP.",
                            f"Tipo recibido: {content_type}",
                        ),
                        status_code=415,
                    )

                image_bytes = await capture_photo.read()

                if not image_bytes:
                    return HTMLResponse(
                        _error_html(
                            "Fotografía vacía",
                            "El teléfono no envió datos de imagen. Vuelve a tomarla.",
                        ),
                        status_code=400,
                    )

                if len(image_bytes) > MAX_CAPTURE_BYTES:
                    return HTMLResponse(
                        _error_html(
                            "Fotografía demasiado pesada",
                            "La imagen supera el máximo de 20 MB. "
                            "Usa la resolución normal de la cámara.",
                        ),
                        status_code=413,
                    )

                assistant = _get_capture_assistant()
                result = assistant.inspect_bytes(image_bytes)

                # Compatibilidad por si inspect_bytes se implementa como async.
                if inspect.isawaitable(result):
                    result = await result

                capture_token = _store_capture_session(
                    result=result,
                    filename=capture_photo.filename,
                    exam_name=exam_name,
                    rubric_filename=rubric_filename,
                )

                return HTMLResponse(
                    _result_html(
                        result=result,
                        filename=capture_photo.filename,
                        capture_token=capture_token,
                        exam_name=exam_name,
                        rubric_filename=rubric_filename,
                    )
                )

            except Exception as exc:
                return HTMLResponse(
                    _error_html(
                        "No pudimos analizar la fotografía",
                        "Evalia detuvo el proceso de forma segura. "
                        "Vuelve a intentarlo con mejor iluminación.",
                        str(exc),
                    ),
                    status_code=500,
                )

    if "/captura/procesar-ocr" not in existing_paths:

        @app.post("/captura/procesar-ocr", response_class=HTMLResponse)
        async def process_capture_ocr(
            capture_token: str = Form(...),
        ) -> HTMLResponse:
            _cleanup_capture_sessions()
            session = _CAPTURE_SESSIONS.pop(capture_token, None)

            if session is None:
                return HTMLResponse(
                    _error_html(
                        "La captura expiró",
                        "La revisión estuvo abierta demasiado tiempo o el servidor se reinició. "
                        "Toma nuevamente la fotografía.",
                    ),
                    status_code=410,
                )

            result = session.get("result")
            filename = str(session.get("filename") or "captura")
            exam_name = str(session.get("exam_name") or "Evaluación")
            rubric_filename = Path(str(session.get("rubric_filename") or "")).name
            pages = _detected_pages(result)

            if not _valid_rubric_filename(rubric_filename):
                return HTMLResponse(
                    _error_html(
                        "Rúbrica no disponible",
                        "La rúbrica asociada a esta captura ya no está disponible. Inicia nuevamente la sesión.",
                    ),
                    status_code=400,
                )

            if not pages:
                return HTMLResponse(
                    _error_html(
                        "No hay hojas para procesar",
                        "La captura confirmada no contiene páginas detectadas.",
                    ),
                    status_code=400,
                )

            temp_dir = Path(tempfile.mkdtemp(prefix="evalia_capture_"))
            ocr_results = []

            try:
                for index, page in enumerate(pages, start=1):
                    page_number = _get_value(
                        page,
                        "page_number",
                        "number",
                        "page",
                        "index",
                        default=index,
                    )
                    student_number = _get_value(
                        page,
                        "student_number",
                        "student_id",
                        "student",
                        "group_id",
                        default=index,
                    )

                    crop = _extract_crop(page)
                    if crop is None:
                        ocr_results.append({
                            "student_number": student_number,
                            "page_number": page_number,
                            "ocr": {},
                            "error": "El detector no entregó un recorte utilizable.",
                        })
                        continue

                    image_path = temp_dir / f"student_{index}_page_{page_number}.jpg"
                    written = cv2.imwrite(
                        str(image_path),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    if not written:
                        ocr_results.append({
                            "student_number": student_number,
                            "page_number": page_number,
                            "ocr": {},
                            "error": "No fue posible guardar temporalmente el recorte.",
                        })
                        continue

                    try:
                        ocr = await run_in_threadpool(run_ocr_on_image, image_path)
                        if not isinstance(ocr, dict):
                            ocr = {
                                "text": str(ocr or ""),
                                "confidence": 0.0,
                                "engine": "unknown",
                                "message": "El servicio OCR devolvió un formato no estándar.",
                            }
                        ocr_results.append({
                            "student_number": student_number,
                            "page_number": page_number,
                            "ocr": ocr,
                            "error": "",
                        })
                    except Exception as exc:
                        ocr_results.append({
                            "student_number": student_number,
                            "page_number": page_number,
                            "ocr": {},
                            "error": f"OCR falló para esta hoja: {exc}",
                        })

                services = _evaluation_services(app)
                load_rubric = _required_service(services, "load_selected_rubric")
                validate_rubric = _required_service(services, "validate_rubric_integrity")
                segment_text = _required_service(services, "segment_ocr_text_by_questions")
                display_item_type = _required_service(services, "display_item_type")

                rubric = load_rubric(rubric_filename)
                issues = validate_rubric(rubric)
                if issues:
                    return HTMLResponse(
                        _error_html(
                            "Rúbrica con problemas",
                            "La rúbrica seleccionada debe corregirse antes de evaluar.",
                            "; ".join(str(issue) for issue in issues),
                        ),
                        status_code=400,
                    )

                students = []
                for index, item in enumerate(ocr_results):
                    ocr = item.get("ocr", {}) or {}
                    raw_text = str(ocr.get("text", "") or "").strip()
                    segments, segmentation = segment_text(raw_text, rubric)
                    students.append({
                        **item,
                        "student_name": "",
                        "segments": segments,
                        "segmentation": segmentation,
                    })

                evaluation_token = _store_evaluation_session(
                    filename=filename,
                    exam_name=exam_name,
                    rubric_filename=rubric_filename,
                    rubric=rubric,
                    students=students,
                )

                return HTMLResponse(
                    _ocr_review_html(
                        filename=filename,
                        students=students,
                        exam_name=exam_name,
                        rubric_filename=rubric_filename,
                        rubric=rubric,
                        evaluation_token=evaluation_token,
                        display_item_type=display_item_type,
                    )
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    if "/captura/evaluar" not in existing_paths:

        @app.post("/captura/evaluar", response_class=HTMLResponse)
        async def evaluate_capture(request: Request) -> HTMLResponse:
            _cleanup_evaluation_sessions()
            form = await request.form()
            evaluation_token = str(form.get("evaluation_token", "") or "")
            session = _EVALUATION_SESSIONS.pop(evaluation_token, None)

            if session is None:
                return HTMLResponse(
                    _error_html(
                        "La revisión expiró",
                        "La sesión de evaluación ya no está disponible. "
                        "Vuelve a procesar la fotografía.",
                    ),
                    status_code=410,
                )

            try:
                services = _evaluation_services(app)
                score_answer = _required_service(services, "score_answer")
                semantic_diagnosis = _required_service(services, "semantic_diagnosis")
                cognitive_level = _required_service(
                    services,
                    "cognitive_level_from_score",
                )
                performance_level = _required_service(
                    services,
                    "performance_level",
                )
                segment_text = _required_service(
                    services,
                    "segment_ocr_text_by_questions",
                )

                rubric = session.get("rubric", {}) or {}
                questions = rubric.get("questions", []) or []
                total_score = (
                    float(rubric.get("total_score", 0) or 0)
                    or sum(float(q.get("max_score", 0) or 0) for q in questions)
                    or 1.0
                )

                evaluation_results = []
                for student_index, student in enumerate(
                    session.get("students", []) or []
                ):
                    student_name = str(
                        form.get(f"student_name__{student_index}", "")
                        or ""
                    ).strip()
                    if not student_name:
                        student_name = (
                            f"Estudiante "
                            f"{student.get('student_number', student_index + 1)}"
                        )

                    raw_text = str(
                        form.get(f"raw_text__{student_index}", "")
                        or ""
                    ).strip()
                    fallback_segments, fallback_info = segment_text(
                        raw_text,
                        rubric,
                    )

                    answers: Dict[str, str] = {}
                    for question in questions:
                        qid = str(question.get("id", ""))
                        direct = str(
                            form.get(
                                f"answer__{student_index}__{qid}",
                                "",
                            )
                            or ""
                        ).strip()
                        answers[qid] = (
                            direct
                            or str(fallback_segments.get(qid, "") or "").strip()
                        )

                    nonempty_answers = [
                        answer for answer in answers.values() if answer.strip()
                    ]
                    if not nonempty_answers:
                        evaluation_results.append({
                            "student_name": student_name,
                            "error": (
                                "No hay respuestas evaluables. "
                                "Evalia no asignó cero porque el problema puede "
                                "provenir del OCR o de la segmentación."
                            ),
                            "segmentation": fallback_info,
                        })
                        continue

                    total = 0.0
                    question_results = []
                    for question in questions:
                        qid = str(question.get("id", ""))
                        answer = answers.get(qid, "")
                        score, confidence, feedback, status = score_answer(
                            answer,
                            question,
                        )
                        diagnosis = semantic_diagnosis(
                            answer,
                            question,
                            score=score,
                            confidence=confidence,
                            status=status,
                        )
                        level = cognitive_level(
                            score,
                            question.get("max_score", 1),
                            confidence,
                            diagnosis,
                        )
                        total += float(score)

                        question_results.append({
                            "question_id": qid,
                            "answer": answer,
                            "score": score,
                            "max_score": question.get("max_score", 0),
                            "confidence": confidence,
                            "status": status,
                            "feedback": feedback,
                            "cognitive_level": level,
                            "diagnosis": diagnosis,
                        })

                    percentage = round((total / total_score) * 100, 2)
                    evaluation_results.append({
                        "student_name": student_name,
                        "student_number": student.get(
                            "student_number",
                            student_index + 1,
                        ),
                        "total_score": round(total, 2),
                        "max_score": total_score,
                        "percentage": percentage,
                        "performance_level": performance_level(percentage),
                        "questions": question_results,
                        "segmentation": fallback_info,
                    })

                return HTMLResponse(
                    _evaluation_results_html(
                        filename=str(session.get("filename") or "captura"),
                        exam_name=str(
                            session.get("exam_name") or "Evaluación"
                        ),
                        rubric_filename=str(
                            session.get("rubric_filename") or ""
                        ),
                        results=evaluation_results,
                    )
                )

            except Exception as exc:
                return HTMLResponse(
                    _error_html(
                        "No pudimos completar la evaluación",
                        "Evalia detuvo el proceso de forma segura.",
                        str(exc),
                    ),
                    status_code=500,
                )

