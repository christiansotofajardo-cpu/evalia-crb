
from __future__ import annotations

"""Evalia 3.2: contrato universal y control de evidencias aprobadas.

Traduce una pauta simple a Task/AssessmentSpec/Criterion sin asignar notas.
No extrae conceptos de listas de palabras ni inventa ponderaciones.
El profesor debe aprobar explícitamente la pauta antes de compilarla.
"""

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Sequence
import math
import re
import unicodedata

from .models import AssessmentSpec, Criterion, Task


VERSION = "evalia-rubric-3.0-1"  # Mantener compatibilidad del motor 3.1.
EVIDENCE_CONTRACT = "evalia-evidence-3.2-provisional"

TYPES = {
    "abierta": ("constructed_response", "semantic"),
    "enumeracion": ("enumeration", "semantic"),
    "explicacion_causal": ("causal_explanation", "semantic"),
    "argumentacion": ("argumentative_response", "semantic"),
    "verdadero_falso": ("true_false", "exact"),
    "seleccion_multiple": ("multiple_choice", "exact"),
    "completar": ("fill_in", "controlled"),
    "numerica": ("numeric_response", "numeric_specialist_required"),
    "procedimiento_matematico": (
        "mathematical_reasoning", "math_specialist_required"
    ),
}


class RubricValidationError(ValueError):
    """La pauta no se puede ejecutar sin aclaración del profesor."""


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RubricValidationError(f"Falta texto válido en '{field_name}'.")
    return value.strip()


def _points(value: Any, field_name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise RubricValidationError(f"'{field_name}' debe ser un número positivo.")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise RubricValidationError(f"'{field_name}' no es numérico.") from None
    if not number.is_finite() or number <= 0:
        raise RubricValidationError(f"'{field_name}' debe ser finito y mayor que cero.")
    if number.as_tuple().exponent < -4:
        raise RubricValidationError(f"'{field_name}' admite hasta cuatro decimales.")
    return number


def _strings(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RubricValidationError(f"'{field_name}' debe ser una lista de textos.")
    result: List[str] = []
    for item in value:
        result.append(_require_text(item, field_name))
    return list(dict.fromkeys(result))


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFD", value.casefold().strip())
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _key_for_closed(task_type: str, value: Any) -> str:
    raw = _require_text(value, "clave").strip()
    normalized = _normalize(raw)
    if task_type == "verdadero_falso":
        if normalized in {"v", "verdadero", "true"}:
            return "V"
        if normalized in {"f", "falso", "false"}:
            return "F"
        raise RubricValidationError("La clave V/F debe ser V o F.")
    if task_type == "seleccion_multiple":
        if len(normalized) != 1 or normalized.upper() not in "ABCDE":
            raise RubricValidationError("La clave de selección múltiple debe ser A-E.")
        return normalized.upper()
    return raw


def compile_question(
    question: Mapping[str, Any],
    *,
    assessment_id: str,
    domain: str = "general",
    language: str = "es",
) -> Task:
    """Compila una pregunta aprobada. No concede crédito ni evalúa respuestas."""
    if not isinstance(question, Mapping):
        raise RubricValidationError("La pregunta debe ser un objeto con campos.")
    if question.get("aprobada") is not True:
        raise RubricValidationError("La rúbrica requiere aprobación expresa del profesor.")

    q_id = _require_text(question.get("id"), "id")
    prompt = _require_text(question.get("enunciado"), "enunciado")
    q_type = _require_text(question.get("tipo"), "tipo")
    if q_type not in TYPES:
        raise RubricValidationError(f"Tipo '{q_type}' no reconocido.")
    max_points = _points(question.get("maximo"), "maximo")
    task_type, scoring_route = TYPES[q_type]
    raw_criteria = question.get("criterios", [])
    if not isinstance(raw_criteria, list):
        raise RubricValidationError("'criterios' debe ser una lista.")

    if q_type in {"verdadero_falso", "seleccion_multiple"}:
        if raw_criteria:
            raise RubricValidationError(
                "Para preguntas cerradas indique solo la clave; no criterios semánticos."
            )
        key = _key_for_closed(q_type, question.get("clave"))
        opposite = (["F"] if key == "V" else ["V"]) if q_type == "verdadero_falso" else []
        raw_criteria = [{
            "id": f"{q_id}_clave", "descripcion": key,
            "puntos": str(max_points), "aceptadas": [key],
            "contradicciones": opposite,
        }]

    if q_type == "completar" and not raw_criteria:
        expected = _require_text(question.get("respuesta_esperada"), "respuesta_esperada")
        raw_criteria = [{
            "id": f"{q_id}_respuesta", "descripcion": expected,
            "puntos": str(max_points),
            "aceptadas": [expected] + _strings(question.get("alternativas"), "alternativas"),
        }]

    if not raw_criteria:
        raise RubricValidationError(
            "Defina criterios y puntos explícitos; no se repartirán automáticamente."
        )

    criteria: List[Criterion] = []
    sum_points = Decimal("0")
    used_ids = set()
    for index, item in enumerate(raw_criteria, 1):
        if not isinstance(item, Mapping):
            raise RubricValidationError("Cada criterio debe ser un objeto.")
        c_id = _require_text(item.get("id", f"{q_id}_C{index}"), "criterio.id")
        if c_id in used_ids:
            raise RubricValidationError(f"Criterio repetido: {c_id}")
        used_ids.add(c_id)
        description = _require_text(item.get("descripcion"), "criterio.descripcion")
        points = _points(item.get("puntos"), f"{c_id}.puntos")
        variants = _strings(item.get("variantes"), f"{c_id}.variantes")
        accepted = _strings(item.get("aceptadas"), f"{c_id}.aceptadas")
        contradictory = _strings(item.get("contradicciones"), f"{c_id}.contradicciones")
        overlap = {_normalize(x) for x in accepted + variants + [description]} & {
            _normalize(x) for x in contradictory
        }
        if overlap:
            raise RubricValidationError(f"{c_id}: evidencia aceptada y contradictoria se solapan.")
        if not isinstance(item.get("obligatorio", False), bool):
            raise RubricValidationError(f"{c_id}.obligatorio debe ser verdadero o falso.")

        # Evidencias atomicas: solo se activan con aprobacion expresa.
        # Las propuestas externas de Psicolinguistica permanecen inertes.
        observables = item.get("observables", [])
        if not isinstance(observables, list):
            raise RubricValidationError(f"{c_id}.observables debe ser una lista.")

        credit = item.get("credito_por_unidad", False)
        if not isinstance(credit, bool):
            raise RubricValidationError(f"{c_id}.credito_por_unidad debe ser booleano.")

        aprobacion = item.get("evidencias_aprobadas", False)
        if not isinstance(aprobacion, bool):
            raise RubricValidationError(f"{c_id}.evidencias_aprobadas debe ser booleano.")

        if observables and not aprobacion:
            raise RubricValidationError(
                f"{c_id}: hay observables no aprobados expresamente por el docente."
            )
        if credit and not aprobacion:
            raise RubricValidationError(
                f"{c_id}: el credito por unidad requiere aprobacion expresa."
            )
        if credit and len(observables) < 2:
            raise RubricValidationError(
                f"{c_id}: el credito proporcional requiere dos o mas observables."
            )

        if q_type in {"verdadero_falso", "seleccion_multiple", "completar",
                      "numerica", "procedimiento_matematico"} and observables:
            raise RubricValidationError(
                f"{c_id}: los observables semanticos no se habilitan en esta ruta."
            )

        parsed_observables: List[Dict[str, Any]] = []
        seen_evidence = set()
        used_unit_ids = set()

        for oi, obs in enumerate(observables, 1):
            if not isinstance(obs, Mapping):
                raise RubricValidationError(f"{c_id}: observable {oi} invalido.")

            unit_id = _require_text(
                obs.get("id", f"{c_id}_U{oi}"),
                "observable.id"
            )
            if unit_id in used_unit_ids:
                raise RubricValidationError(
                    f"{c_id}: ID observable repetido: {unit_id}."
                )
            used_unit_ids.add(unit_id)

            obs_desc = _require_text(
                obs.get("descripcion"),
                "observable.descripcion"
            )
            obs_variants = _strings(
                obs.get("variantes"),
                "observable.variantes"
            )
            obs_accepted = _strings(
                obs.get("aceptadas"),
                "observable.aceptadas"
            )
            obs_weight = _points(
                obs.get("peso", 1),
                "observable.peso"
            )

            if obs.get("obligatorio", True) is not True:
                raise RubricValidationError(
                    f"{c_id}: los observables opcionales necesitan politica adicional."
                )

            # No acreditar dos unidades distintas por la misma formulacion exacta.
            unit_phrases = {
                _normalize(phrase)
                for phrase in [obs_desc] + obs_variants + obs_accepted
            }
            if seen_evidence.intersection(unit_phrases):
                raise RubricValidationError(
                    f"{c_id}: hay evidencia duplicada entre dos observables."
                )
            seen_evidence.update(unit_phrases)

            parsed_observables.append({
                "id": unit_id,
                "description": obs_desc,
                "variants": obs_variants + obs_accepted,
                "required": True,
                "weight": float(obs_weight),
            })

        if not parsed_observables and aprobacion:
            raise RubricValidationError(
                f"{c_id}: evidencias_aprobadas no puede ser verdadero sin observables."
            )

        criteria.append(Criterion(
            id=c_id,
            description=description,
            weight=float(points),
            required=item.get("obligatorio", False),
            semantic_variants=variants,
            accepted_values=accepted,
            contradictory_values=contradictory,
            metadata={
                "explicit_points": str(points),
                "rubric_version": VERSION,
                "observables": parsed_observables,
                "credito_por_unidad": credit,
                "observables_aprobados": (
                    bool(parsed_observables) and aprobacion
                ),
                "evidence_contract": EVIDENCE_CONTRACT,
            },
        ))
        sum_points += points

    if sum_points != max_points:
        raise RubricValidationError(
            f"{q_id}: la suma de criterios ({sum_points}) no coincide "
            f"con el máximo ({max_points})."
        )

    spec = AssessmentSpec(
        id=f"{assessment_id}:{q_id}",
        name=f"Rúbrica aprobada para {q_id}",
        language=language,
        domain=domain,
        criteria=criteria,
        scoring_policy={
            "rubric_version": VERSION,
            "scoring_route": scoring_route,
            "teacher_approved": True,
        },
        reliability_policy={
            "validation_approved": False,
            "auto_accept_enabled": False,
        },
        metadata={
            "rubric_version": VERSION,
            "explicit_max_points": str(max_points),
            "teacher_approved": True,
            "automated_scoring_validated": False,
            "evidence_contract": EVIDENCE_CONTRACT,
        },
    )

    return Task(
        id=q_id,
        prompt=prompt,
        task_type=task_type,
        max_score=float(max_points),
        language=language,
        domain=domain,
        assessment_spec=spec,
        metadata={
            "rubric_version": VERSION,
            "scoring_route": scoring_route,
        },
    )


def compile_assessment(assessment: Mapping[str, Any]) -> List[Task]:
    """Compila y valida el conjunto; exige ID únicos y aprobación por pregunta."""
    if not isinstance(assessment, Mapping):
        raise RubricValidationError("Se requiere una evaluación estructurada.")

    a_id = _require_text(assessment.get("id"), "evaluacion.id")
    domain = _require_text(assessment.get("dominio", "general"), "dominio")
    language = _require_text(assessment.get("idioma", "es"), "idioma")
    questions = assessment.get("preguntas")

    if not isinstance(questions, list) or not questions:
        raise RubricValidationError("La evaluación debe contener preguntas.")

    tasks = [
        compile_question(
            q,
            assessment_id=a_id,
            domain=domain,
            language=language
        )
        for q in questions
    ]

    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise RubricValidationError("Hay IDs de preguntas repetidos.")

    return tasks
