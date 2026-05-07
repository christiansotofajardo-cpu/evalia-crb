from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import pandas as pd
import json
import re
import unicodedata
from rapidfuzz import fuzz
from html import escape
from typing import Optional
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RUBRICS_DIR = BASE_DIR / "rubrics"
RUBRICS_DIR.mkdir(exist_ok=True)

LEGACY_RUBRIC_PATH = BASE_DIR / "rubric_psicolinguistica_2026.json"

app = FastAPI(title="Evalia CRB", version="2.3")


# ============================================================
# NORMALIZACIÓN GENERAL
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^a-z0-9ñ\s/._-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_values(value):
    if pd.isna(value) or value is None:
        return []
    txt = str(value).strip()
    if not txt:
        return []
    parts = re.split(r"\s*[|;]\s*", txt)
    return [p.strip() for p in parts if p.strip()]


def normalize_item_type(value):
    t = normalize_text(value).replace(" ", "_")

    mapping = {
        "vf": "true_false",
        "v_f": "true_false",
        "verdadero_falso": "true_false",
        "true_false": "true_false",
        "truefalse": "true_false",

        "completacion": "completion",
        "completación": "completion",
        "completion": "completion",
        "exacta": "short_exact_answer",
        "respuesta_exacta": "short_exact_answer",
        "short_exact_answer": "short_exact_answer",

        "enumeracion": "enumeration_conceptual",
        "enumeración": "enumeration_conceptual",
        "enumeration": "enumeration_conceptual",
        "enumeration_conceptual": "enumeration_conceptual",
        "lista": "enumeration_conceptual",

        "matching": "classification_matching",
        "emparejamiento": "classification_matching",
        "relacion": "classification_matching",
        "relación": "classification_matching",
        "classification_matching": "classification_matching",

        "criterios": "criteria",
        "criterio": "criteria",
        "abierta": "criteria",
        "respuesta_abierta": "criteria",
        "criteria": "criteria",
    }

    return mapping.get(t, t or "criteria")


def normalize_column_name(name):
    n = normalize_text(name)
    n = n.replace(" ", "")
    n = n.replace("-", "_")
    n = n.replace(".", "_")
    return n


def item_number(item_id):
    m = re.search(r"(\d+)", str(item_id))
    return m.group(1) if m else None


def item_aliases(item_id):
    """
    Permite que una pregunta definida como P1 pueda calzar con Q1,
    pregunta_1, pregunta1, item1, item_1, etc.
    """
    raw = str(item_id).strip()
    n = item_number(raw)

    aliases = {raw, raw.upper(), raw.lower(), normalize_column_name(raw)}

    if n:
        aliases.update({
            f"P{n}", f"p{n}",
            f"Q{n}", f"q{n}",
            f"pregunta{n}", f"pregunta_{n}",
            f"item{n}", f"item_{n}",
            f"i{n}", f"I{n}",
        })

    return {normalize_column_name(a) for a in aliases}


def find_column(df, desired_name, extra_aliases=None):
    normalized_cols = {normalize_column_name(c): c for c in df.columns}
    aliases = {normalize_column_name(desired_name)}

    if extra_aliases:
        aliases.update(normalize_column_name(a) for a in extra_aliases)

    for a in aliases:
        if a in normalized_cols:
            return normalized_cols[a]

    return None


def find_item_column(df, item_id):
    normalized_cols = {normalize_column_name(c): c for c in df.columns}
    for alias in item_aliases(item_id):
        if alias in normalized_cols:
            return normalized_cols[alias]
    return None


def get_student_id(row, df):
    col = find_column(df, "student_id", ["id", "alumno_id", "estudiante_id", "rut", "codigo", "código"])
    return row.get(col, "") if col else ""


def get_student_name(row, df):
    col = find_column(df, "nombre", ["name", "student_name", "alumno", "estudiante", "nombre_estudiante"])
    return row.get(col, "") if col else ""


# ============================================================
# RÚBRICAS JSON Y EXCEL
# ============================================================

def get_available_rubrics():
    rubrics = []

    for path in sorted(RUBRICS_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rubric_name = data.get("name") or data.get("title") or path.stem.replace("_", " ").title()
            rubrics.append({
                "filename": path.name,
                "name": rubric_name,
                "path": path
            })
        except Exception:
            continue

    if not rubrics and LEGACY_RUBRIC_PATH.exists():
        try:
            with open(LEGACY_RUBRIC_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            rubric_name = data.get("name") or data.get("title") or "Psicolingüística 2026"
            rubrics.append({
                "filename": LEGACY_RUBRIC_PATH.name,
                "name": rubric_name,
                "path": LEGACY_RUBRIC_PATH
            })
        except Exception:
            pass

    return rubrics


def safe_rubric_path(selected_filename):
    for r in get_available_rubrics():
        if r["filename"] == selected_filename:
            return r["path"]
    return None


def load_rubric_from_json_path(path):
    with open(path, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    rubric["_rubric_filename"] = path.name
    rubric["_rubric_display_name"] = rubric.get("name") or rubric.get("title") or path.stem.replace("_", " ").title()
    return rubric


def load_selected_rubric(selected_filename=None):
    available = get_available_rubrics()

    if not available:
        raise FileNotFoundError(
            "No se encontraron rúbricas JSON en /rubrics ni rúbrica antigua en la raíz."
        )

    if selected_filename:
        path = safe_rubric_path(selected_filename)
        if path is None:
            raise FileNotFoundError("La rúbrica seleccionada no existe o no está permitida.")
    else:
        path = available[0]["path"]

    return load_rubric_from_json_path(path)


def build_question_from_excel_row(row):
    """
    Formato simple de rúbrica Excel esperado:

    pregunta | tipo | max_score | respuestas | criterios | required_items | prompt

    Ejemplos:
    P1 | criterios | 3 | lenguaje; mente; psicolingüística | lenguaje; mente; psicolingüística | |
    P2 | VF | 1 | V | | |
    P3 | enumeracion | 2 | memoria; atención; comprensión | | 2 |
    P4 | matching | 3 | Broca:producción; Wernicke:comprensión | | |
    """
    qid = str(row.get("pregunta", "")).strip() or str(row.get("id", "")).strip()
    item_type = normalize_item_type(row.get("tipo", row.get("item_type", "criteria")))
    max_score = float(row.get("max_score", row.get("puntaje", row.get("puntaje_maximo", 1))) or 1)
    prompt = "" if pd.isna(row.get("prompt", "")) else str(row.get("prompt", ""))

    respuestas = split_values(row.get("respuestas", row.get("accepted_answers", "")))
    criterios = split_values(row.get("criterios", row.get("criteria", "")))

    question = {
        "id": qid,
        "item_type": item_type,
        "max_score": max_score,
        "prompt": prompt
    }

    if item_type == "true_false":
        question["accepted_answers"] = respuestas or ["V", "Verdadero"]

    elif item_type in ["completion", "short_exact_answer"]:
        question["accepted_answers"] = respuestas

    elif item_type in ["enumeration", "enumeration_conceptual", "enumeration_closed", "enumeration_categorized"]:
        question["accepted_answers"] = respuestas
        required = row.get("required_items", row.get("required_number_of_items", ""))
        if not pd.isna(required) and str(required).strip():
            question["constraints"] = {"required_number_of_items": int(float(required))}

    elif item_type == "classification_matching":
        pairs = []
        for pair_txt in respuestas:
            if ":" in pair_txt:
                left, right = pair_txt.split(":", 1)
            elif "->" in pair_txt:
                left, right = pair_txt.split("->", 1)
            else:
                continue
            pairs.append({
                "prompt_value": left.strip(),
                "correct_match": right.strip(),
                "weight": max_score / max(len(respuestas), 1)
            })
        question["pairs"] = pairs

    else:
        # criteria / abierta
        concepts = criterios or respuestas
        criteria = []
        weight = max_score / max(len(concepts), 1)
        for c in concepts:
            criteria.append({
                "concept": c,
                "weight": weight,
                "semantic_variants": [c],
                "accepted_values": []
            })
        question["item_type"] = "criteria"
        question["criteria"] = criteria

    return question


def load_rubric_from_excel(path):
    df = pd.read_excel(path)

    # Normalizar encabezados para aceptar nombres simples.
    original_cols = list(df.columns)
    col_map = {}
    for c in original_cols:
        nc = normalize_column_name(c)
        if nc in ["pregunta", "id", "item", "item_id"]:
            col_map[c] = "pregunta"
        elif nc in ["tipo", "item_type", "tipo_item"]:
            col_map[c] = "tipo"
        elif nc in ["max_score", "puntaje", "puntajemaximo", "puntaje_maximo"]:
            col_map[c] = "max_score"
        elif nc in ["respuestas", "accepted_answers", "respuesta", "respuestascorrectas", "respuesta_correcta"]:
            col_map[c] = "respuestas"
        elif nc in ["criterios", "criteria", "conceptos", "concepts"]:
            col_map[c] = "criterios"
        elif nc in ["required_items", "required_number_of_items", "numero_requerido", "n_requerido"]:
            col_map[c] = "required_items"
        elif nc in ["prompt", "enunciado", "pregunta_texto"]:
            col_map[c] = "prompt"

    df = df.rename(columns=col_map)

    if "pregunta" not in df.columns:
        raise ValueError("La rúbrica Excel debe incluir una columna llamada 'pregunta'.")

    questions = []
    total_score = 0.0

    for _, row in df.iterrows():
        if pd.isna(row.get("pregunta", "")) or not str(row.get("pregunta", "")).strip():
            continue
        q = build_question_from_excel_row(row)
        questions.append(q)
        total_score += float(q.get("max_score", 0))

    if not questions:
        raise ValueError("La rúbrica Excel no contiene preguntas válidas.")

    required_columns = ["student_id", "nombre"] + [q["id"] for q in questions]

    rubric = {
        "name": path.stem.replace("_", " ").title(),
        "total_score": total_score,
        "input_format": {
            "required_columns": required_columns
        },
        "questions": questions,
        "_rubric_filename": path.name,
        "_rubric_display_name": path.stem.replace("_", " ").title()
    }

    return rubric


async def load_uploaded_rubric(rubric_file: UploadFile):
    suffix = Path(rubric_file.filename).suffix.lower()
    original_name = Path(rubric_file.filename).name
    temp_path = OUTPUT_DIR / f"uploaded_rubric_{original_name}"

    with open(temp_path, "wb") as f:
        f.write(await rubric_file.read())

    if suffix == ".json":
        rubric = load_rubric_from_json_path(temp_path)
        rubric["_rubric_filename"] = original_name
        rubric["_rubric_display_name"] = rubric.get("name") or rubric.get("title") or Path(original_name).stem.replace("_", " ").title()
        return rubric

    if suffix in [".xlsx", ".xls"]:
        rubric = load_rubric_from_excel(temp_path)
        rubric["_rubric_filename"] = original_name
        rubric["_rubric_display_name"] = Path(original_name).stem.replace("_", " ").title()
        return rubric

    raise ValueError("La rúbrica debe estar en formato .json, .xlsx o .xls.")


# ============================================================
# MOTOR CRB
# ============================================================

def fuzzy_contains(answer, target, threshold=78):
    answer_n = normalize_text(answer)
    target_n = normalize_text(target)
    if not answer_n or not target_n:
        return False, 0
    if target_n in answer_n:
        return True, 100
    score = fuzz.partial_ratio(answer_n, target_n)
    return score >= threshold, score


def score_accepted_answers(answer, question):
    accepted = question.get("accepted_answers", [])
    if not accepted:
        return 0, 0.0, "Sin respuestas aceptadas configuradas."

    best = 0
    for target in accepted:
        ok, score = fuzzy_contains(answer, target, threshold=80)
        best = max(best, score)
        if ok:
            return question["max_score"], min(score / 100, 1.0), f"Respuesta coincide con: {target}."
    return 0, best / 100, "No coincide suficientemente con las respuestas aceptadas."


def score_true_false(answer, question):
    a = normalize_text(answer)
    accepted = [normalize_text(x) for x in question.get("accepted_answers", [])]
    if a in accepted:
        return question["max_score"], 1.0, "Respuesta cerrada correcta."
    if accepted and a[:1] == accepted[0][:1]:
        return question["max_score"], 0.9, "Respuesta cerrada correcta por inicial."
    return 0, 0.9 if a else 0.2, "Respuesta cerrada incorrecta o vacía."


def score_criteria(answer, question):
    criteria = question.get("criteria", [])
    if not criteria:
        return score_accepted_answers(answer, question)

    total = 0.0
    matched = []
    missing = []
    confidence_scores = []

    for criterion in criteria:
        weight = float(criterion.get("weight", 1.0))
        variants = []
        concept = criterion.get("concept", "")
        if concept:
            variants.append(concept)
        variants.extend(criterion.get("semantic_variants", []))
        variants.extend(criterion.get("accepted_values", []))

        criterion_best = 0
        criterion_hit = False

        for v in variants:
            ok, score = fuzzy_contains(answer, v, threshold=68)
            criterion_best = max(criterion_best, score)
            if ok:
                criterion_hit = True

        confidence_scores.append(criterion_best / 100 if criterion_best else 0)

        if criterion_hit:
            total += weight
            matched.append(concept)
        else:
            missing.append(concept)

    max_score = float(question.get("max_score", total))
    total = min(total, max_score)
    confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    feedback = []
    if matched:
        feedback.append("Criterios detectados: " + "; ".join([m for m in matched if m]))
    if missing:
        feedback.append("Criterios no detectados: " + "; ".join([m for m in missing if m]))

    return round(total, 2), round(confidence, 2), " | ".join(feedback)


def score_enumeration(answer, question):
    max_score = float(question.get("max_score", 0))
    required = question.get("constraints", {}).get("required_number_of_items")
    accepted = question.get("accepted_concepts", question.get("accepted_answers", []))

    hits = []
    for concept in accepted:
        ok, _ = fuzzy_contains(answer, concept, threshold=68)
        if ok:
            hits.append(concept)

    if required is None:
        required = len(accepted) if accepted else 1

    counted = min(len(set(hits)), required)
    score = max_score * (counted / required) if required else 0
    confidence = counted / required if required else 0

    return round(score, 2), round(confidence, 2), f"Elementos válidos detectados: {counted}/{required}. {', '.join(sorted(set(hits)))}"


def score_matching(answer, question):
    max_score = float(question.get("max_score", 0))
    pairs = question.get("pairs", [])
    total = 0.0
    found = []

    for pair in pairs:
        left = pair.get("prompt_value", "")
        right = pair.get("correct_match", "")
        weight = float(pair.get("weight", 1.0))
        ok_left, _ = fuzzy_contains(answer, left, threshold=65)
        ok_right, _ = fuzzy_contains(answer, right, threshold=65)

        if ok_left and ok_right:
            total += weight
            found.append(f"{left} -> {right}")

    confidence = total / max_score if max_score else 0
    return round(min(total, max_score), 2), round(confidence, 2), "Relaciones detectadas: " + "; ".join(found)


def score_answer(answer, question):
    item_type = question.get("item_type", "")

    if not str(answer).strip():
        return 0, 0.1, "Respuesta vacía.", "revisar"

    if item_type == "true_false":
        score, conf, fb = score_true_false(answer, question)
    elif item_type in ["completion", "short_exact_answer"]:
        score, conf, fb = score_accepted_answers(answer, question)
    elif item_type in ["enumeration", "enumeration_closed", "enumeration_conceptual", "enumeration_categorized"]:
        score, conf, fb = score_enumeration(answer, question)
    elif item_type == "classification_matching":
        score, conf, fb = score_matching(answer, question)
    else:
        score, conf, fb = score_criteria(answer, question)

    status = "aceptado" if conf >= 0.85 else ("revisar" if conf < 0.60 else "aceptado_con_cautela")
    return score, conf, fb, status


def performance_level(pct):
    try:
        pct = float(pct)
    except Exception:
        return "Sin información"
    if pct >= 80:
        return "Alto"
    if pct >= 60:
        return "Medio"
    return "Bajo"


def pedagogical_item_suggestion(classification, review_pct, avg_score_pct, avg_confidence):
    classification = str(classification or "")
    if "problemático" in classification:
        return "Revisar el enunciado, la rúbrica o los criterios de corrección; varios estudiantes podrían haber interpretado la pregunta de manera distinta a lo esperado."
    if review_pct >= 30:
        return "Conviene revisar manualmente una muestra de respuestas antes de cerrar la calificación."
    if avg_score_pct < 60:
        return "La pregunta parece difícil para el grupo; puede requerir retroalimentación o refuerzo de contenidos."
    if avg_confidence < 0.70:
        return "La respuesta esperada podría necesitar criterios más explícitos o ejemplos adicionales."
    return "El ítem muestra funcionamiento estable bajo los criterios actuales de Evalia."


# ============================================================
# VALIDACIÓN FLEXIBLE
# ============================================================

def validate_columns_flexible(df, rubric):
    """
    Valida con flexibilidad:
    - student_id puede ser id, alumno_id, rut, codigo...
    - nombre puede ser name, estudiante, alumno...
    - P1 puede calzar con Q1, pregunta1, item1...
    """
    missing = []

    if not find_column(df, "student_id", ["id", "alumno_id", "estudiante_id", "rut", "codigo", "código"]):
        missing.append("student_id/id")

    if not find_column(df, "nombre", ["name", "student_name", "alumno", "estudiante", "nombre_estudiante"]):
        missing.append("nombre/name")

    for q in rubric.get("questions", []):
        pid = q.get("id")
        if not find_item_column(df, pid):
            aliases = sorted(item_aliases(pid))
            missing.append(f"{pid} ({' / '.join(list(aliases)[:5])}...)")

    return missing


# ============================================================
# INSIGHTS
# ============================================================

def build_question_insights(question_stats, questions):
    question_map = {q.get("id"): q for q in questions}
    insights_rows = []
    problematic_questions = []

    for pid, stats in question_stats.items():
        total = stats["total"] or 1
        max_score = float(question_map.get(pid, {}).get("max_score", 1)) or 1

        avg_score_raw = sum(stats["scores"]) / total if stats["scores"] else 0
        avg_score_pct = (avg_score_raw / max_score) * 100 if max_score else 0
        avg_confidence = sum(stats["confidences"]) / total if stats["confidences"] else 0

        accepted_pct = (stats["accepted"] / total) * 100
        caution_pct = (stats["caution"] / total) * 100
        review_pct = (stats["review"] / total) * 100

        if avg_score_pct >= 80 and review_pct < 20:
            classification = "funcionamiento alto"
        elif review_pct >= 35 or accepted_pct <= 50 or avg_confidence < 0.60:
            classification = "ítem potencialmente problemático"
            problematic_questions.append(pid)
        elif caution_pct >= 30:
            classification = "ítem con cautela interpretativa"
        else:
            classification = "funcionamiento medio/estable"

        insights_rows.append({
            "pregunta": pid,
            "tipo_item": question_map.get(pid, {}).get("item_type", ""),
            "puntaje_maximo": max_score,
            "promedio_puntaje": round(avg_score_raw, 2),
            "promedio_porcentaje": round(avg_score_pct, 1),
            "nivel_item": performance_level(avg_score_pct),
            "confianza_promedio": round(avg_confidence, 2),
            "aceptacion_pct": round(accepted_pct, 1),
            "cautela_pct": round(caution_pct, 1),
            "revision_pct": round(review_pct, 1),
            "clasificacion_evalia": classification,
            "sugerencia_docente": pedagogical_item_suggestion(classification, review_pct, avg_score_pct, avg_confidence)
        })

    return insights_rows, problematic_questions


def build_type_insights(insights_rows):
    if not insights_rows:
        return []

    df = pd.DataFrame(insights_rows)
    grouped = (
        df.groupby("tipo_item", dropna=False)
        .agg(
            preguntas=("pregunta", "count"),
            promedio_porcentaje=("promedio_porcentaje", "mean"),
            confianza_promedio=("confianza_promedio", "mean"),
            revision_pct=("revision_pct", "mean"),
            aceptacion_pct=("aceptacion_pct", "mean")
        )
        .reset_index()
    )

    rows = []
    for _, row in grouped.iterrows():
        rows.append({
            "tipo_item": row["tipo_item"] if row["tipo_item"] else "sin_tipo",
            "numero_preguntas": int(row["preguntas"]),
            "promedio_porcentaje": round(float(row["promedio_porcentaje"]), 1),
            "confianza_promedio": round(float(row["confianza_promedio"]), 2),
            "revision_promedio_pct": round(float(row["revision_pct"]), 1),
            "aceptacion_promedio_pct": round(float(row["aceptacion_pct"]), 1)
        })
    return rows


def build_interpretation(insights_rows, problematic_questions, total_students, rubric_name):
    if not insights_rows:
        return "No fue posible generar interpretación porque no se registraron métricas por pregunta."

    df = pd.DataFrame(insights_rows)
    avg_eval = float(df["promedio_porcentaje"].mean()) if "promedio_porcentaje" in df else 0
    avg_review = float(df["revision_pct"].mean()) if "revision_pct" in df else 0
    avg_conf = float(df["confianza_promedio"].mean()) if "confianza_promedio" in df else 0

    if avg_eval >= 80:
        level = "alto"
    elif avg_eval >= 60:
        level = "medio"
    else:
        level = "bajo"

    if problematic_questions:
        problem_text = (
            "Se identifican posibles focos de revisión en las preguntas "
            + ", ".join(problematic_questions)
            + ", debido a baja aceptación, baja confianza o alta proporción de respuestas marcadas para revisión."
        )
    else:
        problem_text = "No se detectan preguntas críticamente problemáticas bajo los criterios actuales de Evalia."

    return (
        f"La evaluación procesada con la rúbrica '{rubric_name}' incluyó {total_students} estudiante(s). "
        f"El desempeño global estimado por pregunta se ubica en un nivel {level}, "
        f"con un promedio general de {avg_eval:.1f}% del puntaje esperado. "
        f"La confianza promedio del sistema fue {avg_conf:.2f} y la tasa promedio de revisión manual fue {avg_review:.1f}%. "
        f"{problem_text} "
        "Estos resultados constituyen una primera capa de inteligencia evaluativa: permiten orientar la revisión docente, "
        "detectar ítems que podrían requerir ajuste y mejorar progresivamente la calidad de la evaluación."
    )



# ============================================================
# REPORTE EXCEL PREMIUM / DOCENTE
# ============================================================

def build_teacher_report_rows(score_rows, insights_rows, interpretation, problematic_questions, rubric_name, total_students, total_questions, auto_rate, caution_rate, review_rate):
    avg_pct = 0.0
    if score_rows:
        vals = [float(r.get("porcentaje", 0) or 0) for r in score_rows]
        avg_pct = sum(vals) / len(vals)

    if insights_rows:
        sorted_items = sorted(insights_rows, key=lambda x: float(x.get("promedio_porcentaje", 0) or 0))
        hardest = ", ".join([str(x.get("pregunta")) for x in sorted_items[:3]])
        best = ", ".join([str(x.get("pregunta")) for x in sorted_items[-3:][::-1]])
    else:
        hardest = "Sin información"
        best = "Sin información"

    return [
        {"seccion": "Síntesis", "indicador": "Rúbrica aplicada", "valor": rubric_name},
        {"seccion": "Síntesis", "indicador": "Estudiantes procesados", "valor": total_students},
        {"seccion": "Síntesis", "indicador": "Preguntas evaluadas", "valor": total_questions},
        {"seccion": "Resultados", "indicador": "Promedio general estimado", "valor": f"{avg_pct:.1f}%"},
        {"seccion": "Resultados", "indicador": "Nivel global", "valor": performance_level(avg_pct)},
        {"seccion": "Trazabilidad", "indicador": "Aceptación automática", "valor": f"{auto_rate}%"},
        {"seccion": "Trazabilidad", "indicador": "Aceptado con cautela", "valor": f"{caution_rate}%"},
        {"seccion": "Trazabilidad", "indicador": "Requiere revisión manual", "valor": f"{review_rate}%"},
        {"seccion": "Ítems", "indicador": "Preguntas con mayor dificultad", "valor": hardest},
        {"seccion": "Ítems", "indicador": "Preguntas con mejor funcionamiento", "valor": best},
        {"seccion": "Ítems", "indicador": "Preguntas potencialmente problemáticas", "valor": ", ".join(problematic_questions) if problematic_questions else "Sin preguntas críticas"},
        {"seccion": "Interpretación", "indicador": "Lectura automática", "valor": interpretation},
        {"seccion": "Sugerencia", "indicador": "Uso docente recomendado", "valor": "Usar este reporte como primera capa de revisión: confirmar manualmente los casos marcados como revisar y ajustar preguntas o criterios si un ítem aparece como problemático."},
    ]


def format_workbook(writer):
    wb = writer.book
    header_fill = PatternFill("solid", fgColor="111827")
    header_font = Font(color="FFFFFF", bold=True)
    soft_fill = PatternFill("solid", fgColor="F3F4F6")
    green_fill = PatternFill("solid", fgColor="DCFCE7")
    yellow_fill = PatternFill("solid", fgColor="FEF9C3")
    red_fill = PatternFill("solid", fgColor="FEE2E2")
    line = Side(style="thin", color="E5E7EB")
    border = Border(left=line, right=line, top=line, bottom=line)

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.sheet_view.showGridLines = False

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = border
                if cell.row % 2 == 0:
                    cell.fill = soft_fill

        headers = {cell.value: idx + 1 for idx, cell in enumerate(ws[1])}

        def color_by_column(col_name):
            col_idx = headers.get(col_name)
            if not col_idx:
                return
            for r in range(2, ws.max_row + 1):
                value = ws.cell(r, col_idx).value
                txt = str(value).lower() if value is not None else ""
                fill = None
                if any(x in txt for x in ["alto", "aceptado", "funcionamiento alto", "estable"]):
                    fill = green_fill
                if any(x in txt for x in ["medio", "cautela"]):
                    fill = yellow_fill
                if any(x in txt for x in ["bajo", "revisar", "problemático", "problemat"]):
                    fill = red_fill
                if fill:
                    ws.cell(r, col_idx).fill = fill

        for cn in ["nivel_desempeno", "nivel_item", "status", "clasificacion_evalia"]:
            color_by_column(cn)

        for col in range(1, ws.max_column + 1):
            letter = get_column_letter(col)
            max_len = 0
            for cell in ws[letter]:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, min(len(value), 70))
            ws.column_dimensions[letter].width = max(13, min(max_len + 2, 58))

        ws.auto_filter.ref = ws.dimensions

    if "REPORTE_DOCENTE" in wb.sheetnames:
        wb.move_sheet(wb["REPORTE_DOCENTE"], offset=-len(wb.sheetnames))
        ws = wb["REPORTE_DOCENTE"]
        ws.freeze_panes = "A2"
        for row in range(2, ws.max_row + 1):
            ws.cell(row, 1).font = Font(bold=True)
            ws.cell(row, 2).font = Font(bold=True)

# ============================================================
# CSS/UI
# ============================================================

def base_css():
    return """
    <style>
      :root {
        --bg: #f5f6f8; --card: #ffffff; --ink: #15171a; --muted: #667085;
        --line: #e5e7eb; --accent: #111827; --soft: #f9fafb; --ok: #067647;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0; min-height: 100vh;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        background: radial-gradient(circle at top left, rgba(17,24,39,.08), transparent 28%), var(--bg);
        color: var(--ink);
      }
      .page { width: 100%; min-height: 100vh; display: flex; justify-content: center; padding: 42px 20px; }
      .shell { width: 100%; max-width: 920px; }
      .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 26px; }
      .brand { display: flex; align-items: center; gap: 12px; }
      .logo {
        width: 42px; height: 42px; border-radius: 14px; background: #111827; color: white;
        display: grid; place-items: center; font-weight: 800; letter-spacing: -.03em;
      }
      .brand-title { font-size: 22px; font-weight: 800; letter-spacing: -.04em; }
      .brand-subtitle { color: var(--muted); font-size: 13px; margin-top: 2px; }
      .badge {
        padding: 8px 12px; background: white; border: 1px solid var(--line);
        border-radius: 999px; color: var(--muted); font-size: 13px;
      }
      .hero, .result-card {
        background: rgba(255,255,255,.88); backdrop-filter: blur(12px); border: 1px solid var(--line);
        border-radius: 28px; box-shadow: 0 20px 60px rgba(17,24,39,.08); overflow: hidden;
      }
      .hero-inner, .result-card { padding: 34px; }
      h1 { margin: 0; font-size: 42px; line-height: 1.04; letter-spacing: -.06em; }
      .lead { margin: 14px 0 26px; color: var(--muted); font-size: 17px; line-height: 1.55; max-width: 760px; }
      .panel { background: var(--soft); border: 1px solid var(--line); border-radius: 22px; padding: 22px; }
      .field-label { display: block; font-weight: 700; margin-bottom: 8px; font-size: 14px; }
      select, input.file-visible {
        width: 100%; padding: 14px; border: 1px solid var(--line); border-radius: 14px;
        background: white; font-size: 15px; color: var(--ink); margin-bottom: 18px;
      }
      .dropzone {
        position: relative; border: 2px dashed #cbd5e1; background: white; border-radius: 20px;
        padding: 28px 20px; text-align: center; transition: .18s ease; cursor: pointer; margin-bottom: 18px;
      }
      .dropzone:hover { border-color: #111827; transform: translateY(-1px); box-shadow: 0 12px 28px rgba(17,24,39,.08); }
      .dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
      .drop-icon { font-size: 30px; margin-bottom: 8px; }
      .drop-title { font-weight: 800; font-size: 17px; }
      .drop-subtitle { color: var(--muted); margin-top: 6px; font-size: 14px; }
      .file-name { margin-top: 10px; font-size: 14px; color: var(--ok); font-weight: 700; min-height: 18px; }
      .actions { display: flex; align-items: center; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
      button, .button {
        border: none; border-radius: 14px; padding: 13px 19px; font-size: 15px; font-weight: 800;
        background: var(--accent); color: white; cursor: pointer; text-decoration: none;
        display: inline-flex; align-items: center; gap: 8px; transition: .18s ease;
      }
      button:hover, .button:hover { transform: translateY(-1px); box-shadow: 0 14px 24px rgba(17,24,39,.18); }
      .hint { color: var(--muted); font-size: 13px; }
      .features { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; padding: 18px 34px 30px; }
      .feature { background: white; border: 1px solid var(--line); border-radius: 18px; padding: 16px; color: var(--muted); font-size: 13px; }
      .feature strong { display: block; color: var(--ink); font-size: 14px; margin-bottom: 5px; }
      .loader { display: none; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; margin-top: 14px; }
      .spinner {
        width: 18px; height: 18px; border: 3px solid #e5e7eb; border-top-color: #111827;
        border-radius: 50%; animation: spin 1s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
      .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0; }
      .metric { border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: var(--soft); }
      .metric-value { font-size: 24px; font-weight: 900; }
      .metric-label { color: var(--muted); font-size: 13px; margin-top: 4px; }
      .error {
        color: #b42318; background: #fff4f2; border: 1px solid #fecdca;
        border-radius: 18px; padding: 18px;
      }
      code { background: #eef2f7; padding: 2px 6px; border-radius: 6px; }
      @media (max-width: 760px) {
        h1 { font-size: 32px; }
        .features, .metric-grid { grid-template-columns: 1fr; }
        .topbar { align-items: flex-start; gap: 14px; flex-direction: column; }
      }
    </style>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    rubrics = get_available_rubrics()

    if rubrics:
        rubric_options = "\n".join(
            f'<option value="{escape(r["filename"])}">{escape(r["name"])} · {escape(r["filename"])}</option>'
            for r in rubrics
        )
    else:
        rubric_options = '<option value="">Sin rúbricas predefinidas</option>'

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Evalia · Inteligencia Evaluativa Automatizada</title>
      {base_css()}
    </head>
    <body>
      <div class="page">
        <main class="shell">
          <div class="topbar">
            <div class="brand">
              <div class="logo">E</div>
              <div>
                <div class="brand-title">Evalia</div>
                <div class="brand-subtitle">Inteligencia Evaluativa Automatizada</div>
              </div>
            </div>
            <div class="badge">CRB Engine · v2.3 docente</div>
          </div>

          <section class="hero">
            <div class="hero-inner">
              <h1>Evalúa respuestas. Detecta patrones. Mejora evaluaciones.</h1>
              <p class="lead">
                Evalia permite usar rúbricas predefinidas o subir una rúbrica simple en Excel.
                Acepta columnas P1, Q1, pregunta1 o item1 para acercarse al uso real docente.
              </p>

              <form action="/upload" enctype="multipart/form-data" method="post" id="uploadForm">
                <div class="panel">

                  <label class="field-label" for="rubric_selector">Rúbrica predefinida</label>
                  <select name="rubric_selector" id="rubric_selector">
                    {rubric_options}
                  </select>

                  <label class="field-label">O sube una rúbrica propia en Excel/JSON</label>
                  <div class="dropzone">
                    <input name="rubric_file" id="rubricFileInput" type="file" accept=".xlsx,.xls,.json">
                    <div class="drop-icon">🧩</div>
                    <div class="drop-title">Sube rúbrica simple</div>
                    <div class="drop-subtitle">Excel con columnas: pregunta, tipo, max_score, respuestas, criterios</div>
                    <div class="file-name" id="rubricFileName"></div>
                  </div>

                  <label class="field-label">Archivo de respuestas</label>
                  <div class="dropzone">
                    <input name="file" id="fileInput" type="file" accept=".xlsx,.xls" required>
                    <div class="drop-icon">📄</div>
                    <div class="drop-title">Arrastra tu Excel aquí</div>
                    <div class="drop-subtitle">Formatos .xlsx / .xls · acepta P1, Q1, pregunta1 o item1</div>
                    <div class="file-name" id="fileName"></div>
                  </div>

                  <div class="actions">
                    <button type="submit" id="submitBtn">Evaluar respuestas</button>
                    <span class="hint">Si subes rúbrica propia, Evalia la usará por sobre la predefinida.</span>
                  </div>

                  <div class="loader" id="loader">
                    <div class="spinner"></div>
                    <span>Analizando respuestas · aplicando rúbrica · generando insights...</span>
                  </div>
                </div>
              </form>
            </div>

            <div class="features">
              <div class="feature"><strong>Rúbrica Excel</strong>Pensada para profesores.</div>
              <div class="feature"><strong>Formato flexible</strong>P1, Q1, pregunta1 o item1.</div>
              <div class="feature"><strong>Detección de patrones</strong>Identifica ítems críticos.</div>
              <div class="feature"><strong>Reporte explicable</strong>Scores, confianza, feedback e insights.</div>
            </div>
          </section>
        </main>
      </div>

      <script>
        function bindFile(inputId, labelId) {{
          const input = document.getElementById(inputId);
          const label = document.getElementById(labelId);
          if (input) {{
            input.addEventListener("change", () => {{
              if (input.files.length > 0) {{
                label.textContent = "Archivo seleccionado: " + input.files[0].name;
              }}
            }});
          }}
        }}

        bindFile("fileInput", "fileName");
        bindFile("rubricFileInput", "rubricFileName");

        const form = document.getElementById("uploadForm");
        const loader = document.getElementById("loader");
        const submitBtn = document.getElementById("submitBtn");

        if (form) {{
          form.addEventListener("submit", () => {{
            if (loader) loader.style.display = "flex";
            if (submitBtn) {{
              submitBtn.disabled = true;
              submitBtn.textContent = "Procesando...";
            }}
          }});
        }}
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    rubric_selector: str = Form(""),
    rubric_file: Optional[UploadFile] = File(None)
):
    try:
        if rubric_file is not None and rubric_file.filename:
            selected_rubric = await load_uploaded_rubric(rubric_file)
        else:
            selected_rubric = load_selected_rubric(rubric_selector)
    except Exception as e:
        return HTMLResponse(
            f"""
            <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Error · Evalia</title>{base_css()}</head>
            <body><div class="page"><main class="shell"><div class="result-card">
              <div class="error"><strong>Error al cargar la rúbrica.</strong><br>{escape(str(e))}</div>
              <br><a class="button" href="/">Volver</a>
            </div></main></div></body></html>
            """
        )

    input_path = OUTPUT_DIR / file.filename
    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        return HTMLResponse(
            f"""
            <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Error · Evalia</title>{base_css()}</head>
            <body><div class="page"><main class="shell"><div class="result-card">
              <div class="error"><strong>No se pudo leer el Excel.</strong><br>{escape(str(e))}</div>
              <br><a class="button" href="/">Volver</a>
            </div></main></div></body></html>
            """
        )

    missing = validate_columns_flexible(df, selected_rubric)

    if missing:
        return HTMLResponse(
            f"""
            <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Error de formato · Evalia</title>{base_css()}</head>
            <body><div class="page"><main class="shell"><div class="result-card">
              <h1>Error de formato</h1>
              <div class="error"><strong>Faltan columnas requeridas o equivalentes:</strong><br>{escape(", ".join(missing))}</div>
              <p class="lead">Evalia acepta equivalencias como <code>P1/Q1/pregunta1/item1</code>, pero no encontró columnas suficientes.</p>
              <a class="button" href="/">Volver</a>
            </div></main></div></body></html>
            """
        )

    questions = selected_rubric.get("questions", [])
    score_rows = []
    conf_rows = []
    feedback_rows = []

    accepted_count = 0
    caution_count = 0
    review_count = 0
    total_answers = 0

    question_stats = {}
    for p in questions:
        pid = p.get("id")
        question_stats[pid] = {
            "scores": [],
            "confidences": [],
            "accepted": 0,
            "review": 0,
            "caution": 0,
            "total": 0
        }

    for _, row in df.iterrows():
        sid = get_student_id(row, df)
        nombre = get_student_name(row, df)
        score_row = {"student_id": sid, "nombre": nombre}
        conf_row = {"student_id": sid, "nombre": nombre}
        total = 0.0

        for p in questions:
            pid = p["id"]
            col = find_item_column(df, pid)
            answer = row.get(col, "") if col else ""

            score, conf, fb, status = score_answer(answer, p)

            score_row[f"{pid}_score"] = score
            conf_row[f"{pid}_confidence"] = conf
            total += score
            total_answers += 1

            question_stats[pid]["scores"].append(score)
            question_stats[pid]["confidences"].append(conf)
            question_stats[pid]["total"] += 1

            if status == "aceptado":
                accepted_count += 1
                question_stats[pid]["accepted"] += 1
            elif status == "aceptado_con_cautela":
                caution_count += 1
                question_stats[pid]["caution"] += 1
            else:
                review_count += 1
                question_stats[pid]["review"] += 1

            feedback_rows.append({
                "student_id": sid,
                "nombre": nombre,
                "rubric": selected_rubric.get("_rubric_display_name", selected_rubric.get("name", "")),
                "pregunta_id": pid,
                "columna_detectada": col,
                "prompt": p.get("prompt", ""),
                "answer": answer,
                "score": score,
                "max_score": p.get("max_score", ""),
                "confidence": conf,
                "status": status,
                "feedback": fb
            })

        total_score = float(selected_rubric.get("total_score", 0)) or sum(float(p.get("max_score", 0)) for p in questions) or 1
        score_row["total"] = round(total, 2)
        score_row["porcentaje"] = round((total / total_score) * 100, 2)
        score_row["nivel_desempeno"] = performance_level(score_row["porcentaje"])
        score_rows.append(score_row)
        conf_rows.append(conf_row)

    rubric_name = selected_rubric.get("_rubric_display_name", selected_rubric.get("name", "Rúbrica"))
    insights_rows, problematic_questions = build_question_insights(question_stats, questions)
    type_insights_rows = build_type_insights(insights_rows)
    interpretation = build_interpretation(insights_rows, problematic_questions, len(df), rubric_name)

    output_name = f"evalia_resultados_{Path(file.filename).stem}_{Path(selected_rubric.get('_rubric_filename', 'rubrica')).stem}.xlsx"
    output_path = OUTPUT_DIR / output_name

    auto_rate = round((accepted_count / total_answers) * 100, 1) if total_answers else 0
    review_rate = round((review_count / total_answers) * 100, 1) if total_answers else 0
    caution_rate = round((caution_count / total_answers) * 100, 1) if total_answers else 0

    summary_rows = [{
        "rubric": rubric_name,
        "rubric_file": selected_rubric.get("_rubric_filename", ""),
        "students_processed": len(df),
        "questions_evaluated": len(questions),
        "total_answers": total_answers,
        "accepted": accepted_count,
        "accepted_with_caution": caution_count,
        "review_required": review_count,
        "accepted_pct": auto_rate,
        "caution_pct": caution_rate,
        "review_pct": review_rate
    }]

    teacher_report_rows = build_teacher_report_rows(
        score_rows=score_rows,
        insights_rows=insights_rows,
        interpretation=interpretation,
        problematic_questions=problematic_questions,
        rubric_name=rubric_name,
        total_students=len(df),
        total_questions=len(questions),
        auto_rate=auto_rate,
        caution_rate=caution_rate,
        review_rate=review_rate
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(teacher_report_rows).to_excel(writer, sheet_name="REPORTE_DOCENTE", index=False)
        pd.DataFrame(score_rows).to_excel(writer, sheet_name="Puntajes", index=False)
        pd.DataFrame(conf_rows).to_excel(writer, sheet_name="Confianza", index=False)
        pd.DataFrame(feedback_rows).to_excel(writer, sheet_name="Feedback", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Resumen_Tecnico", index=False)
        pd.DataFrame(insights_rows).to_excel(writer, sheet_name="Analisis_Items", index=False)
        pd.DataFrame(type_insights_rows).to_excel(writer, sheet_name="Analisis_Tipos", index=False)
        pd.DataFrame([{
            "rubric": rubric_name,
            "resumen_interpretativo": interpretation,
            "preguntas_potencialmente_problematicas": ", ".join(problematic_questions) if problematic_questions else "Sin preguntas críticas"
        }]).to_excel(writer, sheet_name="Interpretacion", index=False)

        format_workbook(writer)

    problematic_display = ", ".join(problematic_questions) if problematic_questions else "Sin preguntas críticas"

    return HTMLResponse(
        f"""
        <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Resultados · Evalia</title>{base_css()}</head>
        <body><div class="page"><main class="shell">
          <div class="topbar">
            <div class="brand"><div class="logo">E</div><div><div class="brand-title">Evalia</div><div class="brand-subtitle">Reporte generado</div></div></div>
            <div class="badge">CRB Engine · v2.3 docente</div>
          </div>
          <section class="result-card">
            <h1>Procesamiento completado</h1>
            <p class="lead">Evalia aplicó la rúbrica <strong>{escape(rubric_name)}</strong> y generó un reporte Excel explicable.</p>
            <div class="metric-grid">
              <div class="metric"><div class="metric-value">{len(df)}</div><div class="metric-label">estudiante(s)</div></div>
              <div class="metric"><div class="metric-value">{len(questions)}</div><div class="metric-label">preguntas evaluadas</div></div>
              <div class="metric"><div class="metric-value">{auto_rate}%</div><div class="metric-label">aceptación automática</div></div>
              <div class="metric"><div class="metric-value">{review_rate}%</div><div class="metric-label">requiere revisión</div></div>
            </div>
            <p class="lead"><strong>Insight inicial:</strong> {escape(problematic_display)}</p>
            <div class="actions">
              <a class="button" href="/download/{output_name}">Descargar reporte Excel</a>
              <a class="button" style="background:#475467;" href="/">Evaluar otro archivo</a>
            </div>
          </section>
        </main></div></body></html>
        """
    )


@app.get("/download/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / filename
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )
