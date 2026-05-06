from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import pandas as pd
import json
import re
import unicodedata
from rapidfuzz import fuzz
from html import escape

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Evalia v2.0: rúbricas dinámicas
RUBRICS_DIR = BASE_DIR / "rubrics"
RUBRICS_DIR.mkdir(exist_ok=True)

# Compatibilidad: si todavía existe la rúbrica antigua en raíz,
# Evalia puede usarla como respaldo.
LEGACY_RUBRIC_PATH = BASE_DIR / "rubric_psicolinguistica_2026.json"

app = FastAPI(title="Evalia CRB", version="2.0")


def get_available_rubrics():
    """
    Busca todas las rúbricas JSON disponibles en /rubrics.
    Si no hay rúbricas en /rubrics, intenta usar la rúbrica antigua
    ubicada en la raíz del proyecto.
    """
    rubrics = []

    for path in sorted(RUBRICS_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rubric_name = data.get("name") or data.get("title") or path.stem.replace("_", " ").title()
            rubrics.append({"filename": path.name, "name": rubric_name, "path": path})
        except Exception:
            continue

    if not rubrics and LEGACY_RUBRIC_PATH.exists():
        try:
            with open(LEGACY_RUBRIC_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            rubric_name = data.get("name") or data.get("title") or "Psicolingüística 2026"
            rubrics.append({"filename": LEGACY_RUBRIC_PATH.name, "name": rubric_name, "path": LEGACY_RUBRIC_PATH})
        except Exception:
            pass

    return rubrics


def safe_rubric_path(selected_filename):
    """Evita path traversal: solo permite cargar rúbricas registradas."""
    for r in get_available_rubrics():
        if r["filename"] == selected_filename:
            return r["path"]
    return None


def load_rubric(selected_filename=None):
    available = get_available_rubrics()
    if not available:
        raise FileNotFoundError(
            "No se encontraron rúbricas JSON. Crea la carpeta /rubrics y agrega al menos una rúbrica .json."
        )

    if selected_filename:
        path = safe_rubric_path(selected_filename)
        if path is None:
            raise FileNotFoundError("La rúbrica seleccionada no existe o no está permitida.")
    else:
        path = available[0]["path"]

    with open(path, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    rubric["_rubric_filename"] = path.name
    rubric["_rubric_display_name"] = rubric.get("name") or rubric.get("title") or path.stem.replace("_", " ").title()
    return rubric


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


def base_css():
    return """
    <style>
      :root { --bg:#f5f6f8; --card:#ffffff; --ink:#15171a; --muted:#667085; --line:#e5e7eb; --accent:#111827; --soft:#f9fafb; --ok:#067647; }
      * { box-sizing: border-box; }
      body { margin:0; min-height:100vh; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; background: radial-gradient(circle at top left, rgba(17,24,39,.08), transparent 28%), var(--bg); color:var(--ink); }
      .page { width:100%; min-height:100vh; display:flex; justify-content:center; padding:42px 20px; }
      .shell { width:100%; max-width:920px; }
      .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:26px; }
      .brand { display:flex; align-items:center; gap:12px; }
      .logo { width:42px; height:42px; border-radius:14px; background:#111827; color:white; display:grid; place-items:center; font-weight:800; letter-spacing:-.03em; }
      .brand-title { font-size:22px; font-weight:800; letter-spacing:-.04em; }
      .brand-subtitle { color:var(--muted); font-size:13px; margin-top:2px; }
      .badge { padding:8px 12px; background:white; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:13px; }
      .hero { background:rgba(255,255,255,.86); backdrop-filter:blur(12px); border:1px solid var(--line); border-radius:28px; box-shadow:0 20px 60px rgba(17,24,39,.08); overflow:hidden; }
      .hero-inner { padding:34px; }
      h1 { margin:0; font-size:42px; line-height:1.04; letter-spacing:-.06em; }
      .lead { margin:14px 0 26px; color:var(--muted); font-size:17px; line-height:1.55; max-width:720px; }
      .panel { background:var(--soft); border:1px solid var(--line); border-radius:22px; padding:22px; }
      .field-label { display:block; font-weight:700; margin-bottom:8px; font-size:14px; }
      select { width:100%; padding:14px; border:1px solid var(--line); border-radius:14px; background:white; font-size:15px; color:var(--ink); margin-bottom:18px; }
      .dropzone { position:relative; border:2px dashed #cbd5e1; background:white; border-radius:20px; padding:30px 22px; text-align:center; transition:.18s ease; cursor:pointer; }
      .dropzone:hover { border-color:#111827; transform:translateY(-1px); box-shadow:0 12px 28px rgba(17,24,39,.08); }
      .dropzone input { position:absolute; inset:0; opacity:0; cursor:pointer; }
      .drop-icon { font-size:34px; margin-bottom:10px; }
      .drop-title { font-weight:800; font-size:18px; }
      .drop-subtitle { color:var(--muted); margin-top:6px; font-size:14px; }
      .file-name { margin-top:12px; font-size:14px; color:var(--ok); font-weight:700; min-height:18px; }
      .actions { display:flex; align-items:center; gap:14px; margin-top:18px; flex-wrap:wrap; }
      button, .button { border:none; border-radius:14px; padding:13px 19px; font-size:15px; font-weight:800; background:var(--accent); color:white; cursor:pointer; text-decoration:none; display:inline-flex; align-items:center; gap:8px; transition:.18s ease; }
      button:hover, .button:hover { transform:translateY(-1px); box-shadow:0 14px 24px rgba(17,24,39,.18); }
      .hint { color:var(--muted); font-size:13px; }
      .features { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; padding:18px 34px 30px; }
      .feature { background:white; border:1px solid var(--line); border-radius:18px; padding:16px; color:var(--muted); font-size:13px; }
      .feature strong { display:block; color:var(--ink); font-size:14px; margin-bottom:5px; }
      .loader { display:none; align-items:center; gap:10px; color:var(--muted); font-size:14px; margin-top:14px; }
      .spinner { width:18px; height:18px; border:3px solid #e5e7eb; border-top-color:#111827; border-radius:50%; animation:spin 1s linear infinite; }
      @keyframes spin { to { transform: rotate(360deg); } }
      .result-card { background:white; border:1px solid var(--line); border-radius:28px; box-shadow:0 20px 60px rgba(17,24,39,.08); padding:34px; }
      .result-title { font-size:30px; font-weight:900; letter-spacing:-.04em; margin:0 0 10px; }
      .metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:24px 0; }
      .metric { border:1px solid var(--line); border-radius:18px; padding:16px; background:var(--soft); }
      .metric-value { font-size:24px; font-weight:900; }
      .metric-label { color:var(--muted); font-size:13px; margin-top:4px; }
      .error { color:#b42318; background:#fff4f2; border:1px solid #fecdca; border-radius:18px; padding:18px; }
      code { background:#eef2f7; padding:2px 6px; border-radius:6px; }
      @media (max-width:760px) { h1{font-size:32px;} .features,.metric-grid{grid-template-columns:1fr;} .topbar{align-items:flex-start; gap:14px; flex-direction:column;} }
    </style>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    rubrics = get_available_rubrics()

    if not rubrics:
        rubric_options = ""
        disabled = "disabled"
        rubric_warning = """
        <div class="error" style="margin-bottom:18px;">
          <strong>No hay rúbricas disponibles.</strong><br>
          Crea una carpeta <code>rubrics</code> y agrega al menos un archivo <code>.json</code>.
        </div>
        """
    else:
        rubric_options = "\n".join(
            f'<option value="{escape(r["filename"])}">{escape(r["name"])} · {escape(r["filename"])}</option>'
            for r in rubrics
        )
        disabled = ""
        rubric_warning = ""

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Evalia · Evaluación Automatizada Inteligente</title>
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
                <div class="brand-subtitle">Evaluación Automatizada Inteligente</div>
              </div>
            </div>
            <div class="badge">CRB Engine · v2.0 multi-rúbrica</div>
          </div>

          <section class="hero">
            <div class="hero-inner">
              <h1>Corrige respuestas breves con rúbricas configurables.</h1>
              <p class="lead">
                Evalia aplica un motor Criteria/Rubric Based para evaluar respuestas desde Excel,
                calcular puntajes, estimar confianza y generar retroalimentación por ítem.
              </p>

              {rubric_warning}

              <form action="/upload" enctype="multipart/form-data" method="post" id="uploadForm">
                <div class="panel">
                  <label class="field-label" for="rubric">Selecciona rúbrica de evaluación</label>
                  <select name="rubric" id="rubric" {disabled}>
                    {rubric_options}
                  </select>

                  <label class="field-label">Archivo de respuestas</label>
                  <div class="dropzone" id="dropzone">
                    <input name="file" id="fileInput" type="file" accept=".xlsx,.xls" required {disabled}>
                    <div class="drop-icon">📄</div>
                    <div class="drop-title">Arrastra tu Excel aquí</div>
                    <div class="drop-subtitle">o haz clic para seleccionar archivo · formatos .xlsx / .xls</div>
                    <div class="file-name" id="fileName"></div>
                  </div>

                  <div class="actions">
                    <button type="submit" id="submitBtn" {disabled}>Evaluar respuestas</button>
                    <span class="hint">El Excel debe contener las columnas definidas por la rúbrica seleccionada.</span>
                  </div>

                  <div class="loader" id="loader">
                    <div class="spinner"></div>
                    <span>Analizando respuestas · aplicando rúbrica · generando reporte...</span>
                  </div>
                </div>
              </form>
            </div>

            <div class="features">
              <div class="feature"><strong>Multi-rúbrica</strong>Selecciona distintas evaluaciones JSON.</div>
              <div class="feature"><strong>Scoring automático</strong>Puntajes por ítem y total.</div>
              <div class="feature"><strong>Confianza</strong>Marca respuestas seguras y dudosas.</div>
              <div class="feature"><strong>Excel final</strong>Scores, confidence y feedback.</div>
            </div>
          </section>
        </main>
      </div>

      <script>
        const fileInput = document.getElementById("fileInput");
        const fileName = document.getElementById("fileName");
        const form = document.getElementById("uploadForm");
        const loader = document.getElementById("loader");
        const submitBtn = document.getElementById("submitBtn");

        if (fileInput) {{
          fileInput.addEventListener("change", () => {{
            if (fileInput.files.length > 0) {{
              fileName.textContent = "Archivo seleccionado: " + fileInput.files[0].name;
            }}
          }});
        }}

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
    """)


@app.post("/upload")
async def upload(file: UploadFile = File(...), rubric: str = Form(...)):
    try:
        selected_rubric = load_rubric(rubric)
    except Exception as e:
        return HTMLResponse(f"""
        <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Error · Evalia</title>{base_css()}</head>
        <body><div class="page"><main class="shell"><div class="result-card">
        <div class="error"><strong>Error al cargar la rúbrica.</strong><br>{escape(str(e))}</div>
        <br><a class="button" href="/">Volver</a></div></main></div></body></html>
        """)

    input_path = OUTPUT_DIR / file.filename
    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        return HTMLResponse(f"""
        <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Error · Evalia</title>{base_css()}</head>
        <body><div class="page"><main class="shell"><div class="result-card">
        <div class="error"><strong>No se pudo leer el Excel.</strong><br>{escape(str(e))}</div>
        <br><a class="button" href="/">Volver</a></div></main></div></body></html>
        """)

    missing = validate_columns(df, selected_rubric)
    if missing:
        expected = selected_rubric.get("input_format", {}).get("required_columns", [])
        return HTMLResponse(f"""
        <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Error de formato · Evalia</title>{base_css()}</head>
        <body><div class="page"><main class="shell"><div class="result-card">
        <h1 class="result-title">Error de formato</h1>
        <div class="error"><strong>Faltan columnas requeridas:</strong><br>{escape(", ".join(missing))}</div>
        <p class="lead">La rúbrica <strong>{escape(selected_rubric.get("_rubric_display_name", ""))}</strong> espera las siguientes columnas:<br><code>{escape(", ".join(expected))}</code></p>
        <a class="button" href="/">Volver</a></div></main></div></body></html>
        """)

    questions = selected_rubric.get("questions", [])
    score_rows = []
    conf_rows = []
    feedback_rows = []

    accepted_count = 0
    caution_count = 0
    review_count = 0
    total_answers = 0

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
            total_answers += 1

            if status == "aceptado":
                accepted_count += 1
            elif status == "aceptado_con_cautela":
                caution_count += 1
            else:
                review_count += 1

            feedback_rows.append({
                "student_id": sid,
                "nombre": nombre,
                "rubric": selected_rubric.get("_rubric_display_name", ""),
                "question_id": qid,
                "prompt": q.get("prompt", ""),
                "answer": answer,
                "score": score,
                "max_score": q.get("max_score", ""),
                "confidence": conf,
                "status": status,
                "feedback": fb
            })

        total_score = float(selected_rubric.get("total_score", 0)) or sum(float(q.get("max_score", 0)) for q in questions) or 1
        score_row["total"] = round(total, 2)
        score_row["porcentaje"] = round((total / total_score) * 100, 2)
        score_rows.append(score_row)
        conf_rows.append(conf_row)

    output_name = f"evalia_resultados_{Path(file.filename).stem}_{Path(selected_rubric.get('_rubric_filename', 'rubrica')).stem}.xlsx"
    output_path = OUTPUT_DIR / output_name

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(score_rows).to_excel(writer, sheet_name="scores", index=False)
        pd.DataFrame(conf_rows).to_excel(writer, sheet_name="confidence", index=False)
        pd.DataFrame(feedback_rows).to_excel(writer, sheet_name="feedback", index=False)
        pd.DataFrame([{
            "rubric": selected_rubric.get("_rubric_display_name", ""),
            "rubric_file": selected_rubric.get("_rubric_filename", ""),
            "students_processed": len(df),
            "questions_evaluated": len(questions),
            "total_answers": total_answers,
            "accepted": accepted_count,
            "accepted_with_caution": caution_count,
            "review_required": review_count
        }]).to_excel(writer, sheet_name="summary", index=False)

    auto_rate = round((accepted_count / total_answers) * 100, 1) if total_answers else 0
    review_rate = round((review_count / total_answers) * 100, 1) if total_answers else 0

    return HTMLResponse(f"""
    <!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Resultados · Evalia</title>{base_css()}</head>
    <body><div class="page"><main class="shell">
      <div class="topbar"><div class="brand"><div class="logo">E</div><div><div class="brand-title">Evalia</div><div class="brand-subtitle">Reporte generado</div></div></div><div class="badge">CRB Engine · v2.0</div></div>
      <section class="result-card">
        <h1 class="result-title">Procesamiento completado</h1>
        <p class="lead">Evalia aplicó la rúbrica <strong>{escape(selected_rubric.get("_rubric_display_name", ""))}</strong> y generó un reporte Excel con puntajes, confianza, feedback y resumen.</p>
        <div class="metric-grid">
          <div class="metric"><div class="metric-value">{len(df)}</div><div class="metric-label">estudiante(s)</div></div>
          <div class="metric"><div class="metric-value">{len(questions)}</div><div class="metric-label">preguntas evaluadas</div></div>
          <div class="metric"><div class="metric-value">{auto_rate}%</div><div class="metric-label">aceptación automática</div></div>
          <div class="metric"><div class="metric-value">{review_rate}%</div><div class="metric-label">requiere revisión</div></div>
        </div>
        <div class="actions"><a class="button" href="/download/{output_name}">Descargar reporte Excel</a><a class="button" style="background:#475467;" href="/">Evaluar otro archivo</a></div>
      </section>
    </main></div></body></html>
    """)


@app.get("/download/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / filename
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )
