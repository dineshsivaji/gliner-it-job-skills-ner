## gliner-it-job-skills-ner

Fine-tuned [GLiNER](https://github.com/urchade/gliner) project for extracting **IT job skills and roles** from resumes/CVs.

This repo contains:
- **Data relabelling** from many fine-grained IT labels into a compact schema
- **Training script** to fine-tune `urchade/gliner_medium-v2.1` on synthetic resume data
- **Inference/CLI parser** that runs the fine-tuned model and enriches skills with a taxonomy
- **Taxonomy mapper** that groups raw skills into categories like `PROGRAMMING_LANGUAGE`, `CLOUD`, `DEVOPS`, etc.

---

## Project layout

- `src/relabel_dataset.py` — collapse many IT labels into:
  - `TECHNICAL_SKILL`
  - `JOB_TITLE`
- `src/train_gliner_resume.py` — fine-tunes **GLiNER medium** on the relabelled dataset
- `src/taxonomy_mapper.py` — maps each `TECHNICAL_SKILL` span to a fine-grained category
- `src/resume_parser.py` — CLI tool to parse resumes using the fine-tuned model + taxonomy
- `pyproject.toml` — project metadata and dependencies (for `uv`)

---

## Environment (uv-based)

This project is configured as a **uv** Python project via `pyproject.toml`.

### 1. Install uv

See the official docs for the latest install instructions: `https://docs.astral.sh/uv/`

Common one-liner:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

From the project root:

```bash
uv sync
```

This will:
- Create a virtual environment managed by uv
- Install runtime deps: `gliner`, `huggingface_hub`, etc.

### 3. Run scripts with uv

Use `uv run` to ensure the correct environment is used:

```bash
# Parse a resume (demo resume if no args)
uv run python -m src.resume_parser

# Parse custom text
uv run python -m src.resume_parser --text "Your resume text here..."

# Parse a resume from file
uv run python -m src.resume_parser --file path/to/resume.txt
```

---

## Data pipeline

### 1. Re-label the synthetic dataset

The original synthetic dataset contains many IT-related labels (`PYTHON_ECOSYSTEM`, `CLOUD`, `DEVOPS`, etc.) that are **collapsed** into a compact schema for training:

- All tech-related labels → `TECHNICAL_SKILL`
- Job-related labels → `JOB_TITLE`
- Sparse labels (e.g. `SOFT_SKILL`, `EDUCATION`, `CERTIFICATION`, `YEARS_OF_EXPERIENCE`) are **dropped** and handled zero-shot at inference if needed.

Run (paths are set up for Kaggle by default):

```bash
uv run python -m src.relabel_dataset \
  --input /kaggle/input/datasets/dineshsivaji/gliner-synth-it-skills/synthetic_gliner_dataset.json \
  --output /kaggle/working/synthetic_gliner_relabelled.json
```

This prints label distributions before/after and saves the relabelled dataset.

### 2. Train GLiNER medium

`train_gliner_resume.py` fine-tunes `urchade/gliner_medium-v2.1` using:

- Labels: `TECHNICAL_SKILL`, `JOB_TITLE`
- Focal loss
- Support for **single- or multi-GPU** (via `torchrun` on Kaggle)

On Kaggle (single GPU):

```bash
uv run python -m src.train_gliner_resume
```

The script:
- Loads the relabelled dataset (`/kaggle/working/synthetic_gliner_relabelled.json`)
- Splits into train/test
- Trains for a target number of steps
- Saves the model locally and (optionally) uploads it to the configured Hugging Face repo

> Note: The Hugging Face upload section expects `HF_TOKEN` to be configured as a Kaggle secret.

---

## Inference: resume parser

`src/resume_parser.py` implements a **2‑stage pipeline**:

1. **GLiNER model** (`dineshsivaji/gliner-it-job-skills-ner`)
   - Predicts entities for labels:
     - Trained: `TECHNICAL_SKILL`, `JOB_TITLE`
     - Optional zero‑shot: `SOFT_SKILL`, `CERTIFICATION`, `EDUCATION`, `YEARS_OF_EXPERIENCE`
   - Uses chunking to stay under the model’s max context length.
2. **TaxonomyMapper**
   - Maps each `TECHNICAL_SKILL` span into one of:
     - `PROGRAMMING_LANGUAGE`, `PYTHON_ECOSYSTEM`, `JS_ECOSYSTEM`, `JAVA_ECOSYSTEM`
     - `CLOUD`, `DEVOPS`, `DATABASE`, `AI_ML`, `MOBILE`, `IT_TOOLS`, etc.
   - Falls back to `UNCATEGORIZED` when a skill is unknown.

Example usage:

```bash
uv run python -m src.resume_parser
```

If no input is provided, it runs on a **demo resume** baked into the script and prints a structured summary of:

- Technical skills by category
- Job titles
- (Optional) zero‑shot fields like certifications and education

To parse your own text:

```bash
uv run python -m src.resume_parser --text "Paste your resume text here..."
```

Or from a file:

```bash
uv run python -m src.resume_parser --file /path/to/resume.txt
```

---

## Taxonomy mapping

`src/taxonomy_mapper.py` contains the mapping logic that turns raw skill strings into categories.

Key APIs:

- `TaxonomyMapper.map("React")  -> "JS_ECOSYSTEM"`
- `TaxonomyMapper.map("AWS")    -> "CLOUD"`
- `TaxonomyMapper.enrich(entities)`:
  - Takes GLiNER entities and adds a `"category"` field to `TECHNICAL_SKILL` items.

You can run a small smoke test directly:

```bash
uv run python -m src.taxonomy_mapper
```

This prints example mappings and a grouped summary by category.

---

## Notes / assumptions

- The project targets **Python 3.10+**.
- Training code assumes a **Kaggle** environment with GPU and secrets configured.
- The inference script is designed to run locally (e.g. on macOS) with the model downloaded from Hugging Face.

