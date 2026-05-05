from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import pandas as pd
import json
import re
import unicodedata
from rapidfuzz import fuzz

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
RUBRIC_PATH = BASE_DIR / "rubric_psicolinguistica_2026.json"

app = FastAPI(title="Evalia CRB", version="1.0")


def load_rubric():
    with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^a-z0-9ñ\s/.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

    if isinstance(accepted, dict):
        total_hits = 0
        total_required = 0
        details = []

        for cat, vals in accepted.items():
            cat_hits = 0
            for v in vals:
                ok, _ = fuzzy_contains(answer, v, threshold=68)
                if ok:
                    cat_hits += 1

            needed = question.get("constraints", {}).get("min_per_category", 1)
            counted = min(cat_hits, needed)
            total_hits += counted
            total_required += needed
            details.append(f"{cat}: {counted}/{needed}")

        score = max_score * (total_hits / total_required) if total_required else 0
        conf = min(1.0, total_hits / total_required) if total_required else 0

        return round(score, 2), round(conf, 2), "Enumeración categorizada: " + "; ".join(details)

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


def validate_columns(df, rubric):
    required = rubric.get("input_format", {}).get("required_columns", [])
    return [c for c in required if c not in df.columns]


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <title>Evalia CRB</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 40px;
          background: #f7f7f8;
          color: #222;
        }
        .card {
          background: white;
          padding: 28px;
          border-radius: 16px;
          max-width: 760px;
          box-shadow: 0 4px 18px rgba(0,0,0,.08);
        }
        h1 { margin-top: 0; }
        .muted { color: #666; }
        input, button { font-size: 16px; margin-top: 12px; }
        button {
          padding: 10px 18px;
          border: none;
          border-radius: 10px;
          cursor: pointer;
          background: #222;
          color: white;
        }
        code {
          background:#eee;
          padding:2px 6px;
          border-radius:6px;
        }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Evalia CRB</h1>
        <p class="muted">MVP 1 · Corrección automatizada de respuestas breves desde Excel</p>

        <form action="/upload" enctype="multipart/form-data" method="post">
          <p>Sube un Excel con columnas <code>student_id</code>, <code>nombre</code>, <code>Q1</code> ... <code>Q34</code>.</p>
          <input name="file" type="file" accept=".xlsx,.xls" required>
          <br>
          <button type="submit">Procesar certamen</button>
        </form>
      </div>
    </body>
    </html>
    """)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    rubric = load_rubric()
    input_path = OUTPUT_DIR / file.filename

    with open(input_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_excel(input_path)
    missing = validate_columns(df, rubric)

    if missing:
        return HTMLResponse(
            f"<h2>Error de formato</h2><p>Faltan columnas: {', '.join(missing)}</p>"
            "<p>El Excel debe incluir: student_id, nombre, Q1...Q34.</p>"
            "<p><a href='/'>Volver</a></p>"
        )

    questions = rubric.get("questions", [])
    score_rows = []
    conf_rows = []
    feedback_rows = []

    for _, row in df.iterrows():
        sid = row.get("student_id")
        nombre = row.get("nombre")
        score_row = {"student_id": sid, "nombre": nombre}
        conf_row = {"student_id": sid, "nombre": nombre}
        total = 0.0

        for q in questions:
            qid = q["id"]
            answer = row.get(qid, "")
            score, conf, fb, status = score_answer(answer, q)

            score_row[f"{qid}_score"] = score
            conf_row[f"{qid}_confidence"] = conf
            total += score

            feedback_rows.append({
                "student_id": sid,
                "nombre": nombre,
                "question_id": qid,
                "prompt": q.get("prompt", ""),
                "answer": answer,
                "score": score,
                "max_score": q.get("max_score", ""),
                "confidence": conf,
                "status": status,
                "feedback": fb
            })

        score_row["total"] = round(total, 2)
        score_row["porcentaje"] = round((total / float(rubric.get("total_score", 81))) * 100, 2)
        score_rows.append(score_row)
        conf_rows.append(conf_row)

    output_name = f"evalia_resultados_{Path(file.filename).stem}.xlsx"
    output_path = OUTPUT_DIR / output_name

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(score_rows).to_excel(writer, sheet_name="scores", index=False)
        pd.DataFrame(conf_rows).to_excel(writer, sheet_name="confidence", index=False)
        pd.DataFrame(feedback_rows).to_excel(writer, sheet_name="feedback", index=False)

    return HTMLResponse(
        f"""
        <h2>Procesamiento completado</h2>
        <p>Archivo generado correctamente.</p>
        <p><a href="/download/{output_name}">Descargar resultados Excel</a></p>
        <p><a href="/">Volver</a></p>
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

