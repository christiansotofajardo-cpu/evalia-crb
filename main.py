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

app = FastAPI(title="Evalia CRB", version="1.5")


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


def base_html(content: str, title: str = "Evalia CRB"):
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{title}</title>
      <style>
        :root {{
          --bg: #f5f6f8;
          --card: #ffffff;
          --text: #171717;
          --muted: #666f7a;
          --line: #d9dde3;
          --primary: #111827;
          --primary-hover: #000000;
          --soft: #f0f2f5;
          --success: #137333;
          --warning: #9a6700;
        }}

        * {{ box-sizing: border-box; }}

        body {{
          margin: 0;
          font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
          background:
            radial-gradient(circle at top left, rgba(30, 64, 175, 0.08), transparent 28%),
            var(--bg);
          color: var(--text);
        }}

        .page {{
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 36px 20px;
        }}

        .shell {{
          width: 100%;
          max-width: 920px;
        }}

        .brand {{
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: 18px;
        }}

        .logo {{
          width: 44px;
          height: 44px;
          border-radius: 14px;
          background: var(--primary);
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 800;
          letter-spacing: -0.04em;
        }}

        .brand h1 {{
          margin: 0;
          font-size: 34px;
          letter-spacing: -0.04em;
        }}

        .brand p {{
          margin: 4px 0 0;
          color: var(--muted);
          font-size: 15px;
        }}

        .card {{
          background: var(--card);
          border: 1px solid rgba(0,0,0,0.04);
          border-radius: 26px;
          padding: 34px;
          box-shadow: 0 24px 70px rgba(17, 24, 39, 0.10);
        }}

        .eyebrow {{
          display: inline-flex;
          align-items: center;
          gap: 8px;
          color: var(--muted);
          background: var(--soft);
          padding: 7px 11px;
          border-radius: 999px;
          font-size: 13px;
          margin-bottom: 18px;
        }}

        h2 {{
          margin: 0 0 10px;
          font-size: 28px;
          letter-spacing: -0.03em;
        }}

        .lead {{
          margin: 0 0 24px;
          color: var(--muted);
          line-height: 1.55;
          max-width: 760px;
        }}

        .dropzone {{
          border: 2px dashed var(--line);
          background: #fbfbfc;
          border-radius: 22px;
          padding: 34px;
          text-align: center;
          transition: all .2s ease;
          cursor: pointer;
        }}

        .dropzone:hover, .dropzone.dragover {{
          border-color: var(--primary);
          background: #f8fafc;
          transform: translateY(-1px);
        }}

        .drop-icon {{
          width: 54px;
          height: 54px;
          margin: 0 auto 14px;
          border-radius: 18px;
          background: var(--soft);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 26px;
        }}

        .drop-title {{
          font-weight: 750;
          font-size: 19px;
          margin-bottom: 6px;
        }}

        .drop-subtitle {{
          color: var(--muted);
          font-size: 14px;
        }}

        input[type="file"] {{ display: none; }}

        .file-name {{
          margin-top: 14px;
          color: var(--primary);
          font-size: 14px;
          font-weight: 650;
        }}

        .actions {{
          display: flex;
          gap: 12px;
          align-items: center;
          justify-content: space-between;
          margin-top: 22px;
          flex-wrap: wrap;
        }}

        .btn {{
          appearance: none;
          border: none;
          border-radius: 14px;
          padding: 13px 22px;
          font-size: 15px;
          font-weight: 750;
          cursor: pointer;
          text-decoration: none;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }}

        .btn-primary {{
          background: var(--primary);
          color: white;
        }}

        .btn-primary:hover {{ background: var(--primary-hover); }}

        .btn-secondary {{
          background: var(--soft);
          color: var(--primary);
        }}

        .hint {{
          color: var(--muted);
          font-size: 13px;
        }}

        .features {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
          margin-top: 24px;
        }}

        .feature {{
          background: #fbfbfc;
          border: 1px solid #edf0f3;
          border-radius: 16px;
          padding: 14px;
          color: #374151;
          font-size: 13px;
          line-height: 1.35;
        }}

        .loader {{
          display: none;
          margin-top: 22px;
          padding: 18px;
          background: #fbfbfc;
          border-radius: 18px;
          color: var(--muted);
        }}

        .bar {{
          height: 8px;
          background: var(--soft);
          border-radius: 999px;
          overflow: hidden;
          margin-top: 12px;
        }}

        .bar span {{
          display: block;
          height: 100%;
          width: 45%;
          background: var(--primary);
          border-radius: 999px;
          animation: move 1.3s infinite ease-in-out;
        }}

        @keyframes move {{
          0% {{ transform: translateX(-120%); }}
          100% {{ transform: translateX(240%); }}
        }}

        .stats {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 12px;
          margin: 24px 0;
        }}

        .stat {{
          background: #fbfbfc;
          border: 1px solid #edf0f3;
          border-radius: 18px;
          padding: 18px;
        }}

        .stat-number {{
          font-size: 26px;
          font-weight: 850;
          letter-spacing: -0.04em;
        }}

        .stat-label {{
          color: var(--muted);
          font-size: 13px;
          margin-top: 4px;
        }}

        .success {{ color: var(--success); }}
        .warning {{ color: var(--warning); }}

        .error-box {{
          background: #fff8f6;
          border: 1px solid #ffd7ce;
          color: #7a271a;
          border-radius: 18px;
          padding: 18px;
          margin-top: 18px;
          line-height: 1.5;
        }}

        code {{
          background: var(--soft);
          padding: 2px 6px;
          border-radius: 7px;
          font-size: 13px;
        }}

        @media (max-width: 760px) {{
          .card {{ padding: 24px; }}
          .features, .stats {{ grid-template-columns: 1fr 1fr; }}
          .brand h1 {{ font-size: 30px; }}
        }}
      </style>
    </head>
    <body>
      <main class="page">
        <section class="shell">
          {content}
        </section>
      </main>
    </body>
    </html>
    """)


@app.get("/", response_class=HTMLResponse)
def home():
    content = """
      <div class="brand">
        <div class="logo">Ev</div>
        <div>
          <h1>Evalia</h1>
          <p>Evaluación automatizada inteligente basada en rúbricas</p>
        </div>
      </div>

      <div class="card">
        <div class="eyebrow">MVP 1.5 · Motor CRB operativo</div>
        <h2>Corrige respuestas breves desde Excel</h2>
        <p class="lead">
          Sube un archivo Excel con columnas <code>student_id</code>, <code>nombre</code>, <code>Q1</code> ... <code>Q34</code>.
          Evalia aplicará la rúbrica configurada, calculará puntajes, confianza, estado de revisión y feedback por ítem.
        </p>

        <form id="uploadForm" action="/upload" enctype="multipart/form-data" method="post">
          <label class="dropzone" id="dropzone" for="fileInput">
            <div class="drop-icon">📄</div>
            <div class="drop-title">Arrastra tu Excel aquí</div>
            <div class="drop-subtitle">o haz clic para seleccionar archivo · .xlsx / .xls</div>
            <div class="file-name" id="fileName">Ningún archivo seleccionado</div>
          </label>

          <input id="fileInput" name="file" type="file" accept=".xlsx,.xls" required>

          <div class="actions">
            <button class="btn btn-primary" type="submit">Evaluar respuestas</button>
            <span class="hint">Rúbrica activa: <code>rubric_psicolinguistica_2026.json</code></span>
          </div>

          <div class="loader" id="loader">
            <strong>Procesando evaluación...</strong><br>
            Analizando respuestas, aplicando criterios y generando reporte Excel.
            <div class="bar"><span></span></div>
          </div>
        </form>

        <div class="features">
          <div class="feature">✓ Corrección automática por rúbrica</div>
          <div class="feature">✓ Puntaje y confianza por ítem</div>
          <div class="feature">✓ Feedback explicativo</div>
          <div class="feature">✓ Exportación Excel</div>
        </div>
      </div>

      <script>
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const dropzone = document.getElementById('dropzone');
        const form = document.getElementById('uploadForm');
        const loader = document.getElementById('loader');

        fileInput.addEventListener('change', () => {
          fileName.textContent = fileInput.files.length ? fileInput.files[0].name : 'Ningún archivo seleccionado';
        });

        ['dragenter', 'dragover'].forEach(eventName => {
          dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
          });
        });

        ['dragleave', 'drop'].forEach(eventName => {
          dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
          });
        });

        dropzone.addEventListener('drop', (e) => {
          const files = e.dataTransfer.files;
          if (files.length) {
            fileInput.files = files;
            fileName.textContent = files[0].name;
          }
        });

        form.addEventListener('submit', () => {
          loader.style.display = 'block';
        });
      </script>
    """
    return base_html(content)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    rubric = load_rubric()
    input_path = OUTPUT_DIR / file.filename

    with open(input_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_excel(input_path)
    missing = validate_columns(df, rubric)

    if missing:
        content = f"""
          <div class="brand">
            <div class="logo">Ev</div>
            <div>
              <h1>Evalia</h1>
              <p>Evaluación automatizada inteligente basada en rúbricas</p>
            </div>
          </div>
          <div class="card">
            <div class="eyebrow">Error de formato</div>
            <h2>No se pudo procesar el archivo</h2>
            <div class="error-box">
              <strong>Faltan columnas requeridas:</strong><br>
              {', '.join(missing)}
              <br><br>
              El Excel debe incluir <code>student_id</code>, <code>nombre</code>, <code>Q1</code> ... <code>Q34</code>.
            </div>
            <div class="actions">
              <a class="btn btn-secondary" href="/">Volver</a>
            </div>
          </div>
        """
        return base_html(content, title="Evalia · Error")

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

    scores_df = pd.DataFrame(score_rows)
    conf_df = pd.DataFrame(conf_rows)
    feedback_df = pd.DataFrame(feedback_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        scores_df.to_excel(writer, sheet_name="scores", index=False)
        conf_df.to_excel(writer, sheet_name="confidence", index=False)
        feedback_df.to_excel(writer, sheet_name="feedback", index=False)

    n_students = len(scores_df)
    n_questions = len(questions)
    total_items = len(feedback_df)
    accepted = int((feedback_df["status"] == "aceptado").sum()) if total_items else 0
    caution = int((feedback_df["status"] == "aceptado_con_cautela").sum()) if total_items else 0
    review = int((feedback_df["status"] == "revisar").sum()) if total_items else 0
    auto_rate = round(((accepted + caution) / total_items) * 100, 1) if total_items else 0
    review_rate = round((review / total_items) * 100, 1) if total_items else 0
    avg_score = round(scores_df["porcentaje"].mean(), 1) if "porcentaje" in scores_df else 0

    content = f"""
      <div class="brand">
        <div class="logo">Ev</div>
        <div>
          <h1>Evalia</h1>
          <p>Evaluación automatizada inteligente basada en rúbricas</p>
        </div>
      </div>

      <div class="card">
        <div class="eyebrow success">✓ Procesamiento completado</div>
        <h2>Reporte generado correctamente</h2>
        <p class="lead">
          Evalia evaluó el archivo <strong>{file.filename}</strong> y generó un reporte Excel con puntajes,
          confianza y feedback detallado por ítem.
        </p>

        <div class="stats">
          <div class="stat">
            <div class="stat-number">{n_students}</div>
            <div class="stat-label">estudiante(s)</div>
          </div>
          <div class="stat">
            <div class="stat-number">{n_questions}</div>
            <div class="stat-label">preguntas</div>
          </div>
          <div class="stat">
            <div class="stat-number">{auto_rate}%</div>
            <div class="stat-label">aceptación automática</div>
          </div>
          <div class="stat">
            <div class="stat-number warning">{review_rate}%</div>
            <div class="stat-label">requiere revisión</div>
          </div>
        </div>

        <p class="lead">
          Puntaje promedio estimado del archivo: <strong>{avg_score}%</strong>.
        </p>

        <div class="actions">
          <a class="btn btn-primary" href="/download/{output_name}">Descargar reporte Excel</a>
          <a class="btn btn-secondary" href="/">Procesar otro archivo</a>
        </div>
      </div>
    """
    return base_html(content, title="Evalia · Resultados")


@app.get("/download/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / filename
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )
