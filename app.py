"""
Gradio web app for GLiNER IT Job Skills NER.

Loads the fine-tuned GLiNER model once at startup, then exposes a browser UI
where users can paste resume text or upload a file (.txt / .pdf) and get
structured skill and job-title extraction with confidence scores.

Deploy on Hugging Face Spaces or run locally:
    python app.py
"""

import gradio as gr

from src.resume_parser import ResumeParser, DEMO_TEXT, LABELS, ZEROSHOT_LABELS

# ── Load model once at startup ───────────────────────────────────────────────

parser = ResumeParser()

# ── Formatting ───────────────────────────────────────────────────────────────


def format_results(result: dict) -> str:
    """Format the structured parse result as Markdown for the Gradio UI."""
    lines: list[str] = []

    # ── Technical Skills ─────────────────────────────────────────────────
    tech = result.get("TECHNICAL_SKILL", {})
    if isinstance(tech, dict) and tech:
        lines.append("## Technical Skills\n")
        lines.append("| Category | Skill |")
        lines.append("|----------|-------|")
        for cat in sorted(tech):
            for skill in tech[cat]:
                lines.append(f"| {cat} | {skill} |")
        lines.append("")

    # ── Other labels (JOB_TITLE, etc.) ───────────────────────────────────
    for label in LABELS:
        if label == "TECHNICAL_SKILL":
            continue
        items = result.get(label, [])
        if not items:
            continue
        display_label = label.replace("_", " ").title()
        zs_note = " _(zero-shot)_" if label in ZEROSHOT_LABELS else ""
        lines.append(f"## {display_label}{zs_note}\n")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    if not lines:
        return "_No entities found. Try pasting a longer resume text._"

    return "\n".join(lines)


# ── Prediction entry point ───────────────────────────────────────────────────


def predict(text: str, file) -> str:
    """Handle text input and/or file upload, return formatted Markdown."""
    # If a file was uploaded, extract text from it
    if file is not None:
        file_path = file.name if hasattr(file, "name") else str(file)
        if file_path.lower().endswith(".pdf"):
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(file_path)
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
            except ImportError:
                return "**Error**: PyMuPDF is required for PDF uploads. Install with `pip install PyMuPDF`."
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    if not text or not text.strip():
        return "_Please provide resume text or upload a file._"

    result = parser.parse(text)
    return format_results(result)


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="GLiNER IT Job Skills NER") as app:
    gr.Markdown(
        "# GLiNER IT Job Skills NER\n"
        "Extract **technical skills** (grouped by category) and **job titles** from resume text.\n\n"
        "Model: [`dineshsivaji/gliner-it-job-skills-ner`]"
        "(https://huggingface.co/dineshsivaji/gliner-it-job-skills-ner)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Resume Text",
                placeholder="Paste resume text here...",
                lines=14,
            )
            file_input = gr.File(
                label="Or upload a file (.txt, .pdf)",
                file_types=[".txt", ".pdf"],
            )
            run_btn = gr.Button("Extract Skills", variant="primary")
        with gr.Column(scale=1):
            output = gr.Markdown(label="Results")

    run_btn.click(fn=predict, inputs=[text_input, file_input], outputs=output)

    gr.Examples(
        examples=[[DEMO_TEXT.strip(), None]],
        inputs=[text_input, file_input],
        label="Try this example",
    )

if __name__ == "__main__":
    app.launch()
