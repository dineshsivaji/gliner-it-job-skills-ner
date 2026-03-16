# GLiNER IT Job Skills NER -- Project Summary

**Project:** `gliner-it-job-skills-ner`
**Author:** Dinesh Sivaji
**License:** Apache 2.0
**Python:** >= 3.10
**Package Manager:** uv
**Published Model:** `dineshsivaji/gliner-it-job-skills-ner` (Hugging Face Hub)

---

## What This Project Does

This project fine-tunes a GLiNER (Generalist and Lightweight Named Entity Recognition) model to extract **IT job skills** and **job titles** from resumes, CVs, and job descriptions. It then enriches the extracted skills with fine-grained categories using a rule-based taxonomy mapper.

**Input:** Raw resume text (via `--text` string, `--file` path, or built-in demo)
**Output:** Structured JSON with skills grouped by category + job titles

Example output:

```json
{
  "TECHNICAL_SKILL": {
    "PROGRAMMING_LANGUAGE": ["Python", "Java", "C#"],
    "CLOUD": ["AWS", "Azure"],
    "DEVOPS": ["Docker", "Kubernetes", "Terraform"],
    "PYTHON_ECOSYSTEM": ["Django", "FastAPI"],
    "DATABASE": ["PostgreSQL", "Redis"]
  },
  "JOB_TITLE": ["Software Engineer", "Technology Consultant"]
}
```

---

## Architecture: Two-Stage Pipeline

```
Resume Text
    |
    v
+-------------------------------+
|  Stage 1: GLiNER NER Model    |
|  (fine-tuned medium, 170M)    |
|  Labels: TECHNICAL_SKILL,     |
|          JOB_TITLE             |
|  Chunking for long documents  |
+-------------------------------+
    |
    v
+-------------------------------+
|  Stage 2: Taxonomy Mapper     |
|  Rule-based categorization    |
|  209 skill -> 12 categories   |
|  3-tier matching:             |
|    exact > word-boundary >    |
|    fuzzy (optional)           |
+-------------------------------+
    |
    v
Structured JSON Output
```

**Why two stages?**
- The neural model handles linguistic complexity (abbreviations, misspellings, context)
- The rule-based mapper is fast, interpretable, and extensible without retraining
- Adding a new skill category only requires editing `taxonomy_mapper.py`

---

## Project Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/relabel_dataset.py` | 266 | Collapses 17 fine-grained labels into 2 coarse labels for training |
| `src/skill_synth_data_gen.py` | 140 | Generates 50K synthetic training examples from templates |
| `src/train_gliner_resume.py` | 188 | Fine-tunes GLiNER medium with focal loss on Kaggle GPUs |
| `src/taxonomy_mapper.py` | 462 | Maps TECHNICAL_SKILL spans to 12 fine-grained categories |
| `src/resume_parser.py` | 254 | End-to-end inference pipeline with chunking and deduplication |
| `eval_model.py` | 265 | Evaluates model with P/R/F1/F2 metrics and threshold sweep |
| `validate_data.py` | 194 | Audits training data for annotation quality issues |
| `pyproject.toml` | 24 | Project metadata and dependencies |
| **Total** | **1,803** | |

---

## Training Data

| Dataset | Examples | File Size |
|---------|----------|-----------|
| Original (`synthetic_gliner_dataset.json`) | 50,218 | 37.6 MB |
| Relabelled (`synthetic_gliner_relabelled.json`) | 37,583 | 36.3 MB |

**Original label distribution (21 labels):**

| Label | Spans |
|-------|-------|
| PROGRAMMING_LANGUAGE | 112,417 |
| CLOUD | 34,300 |
| JAVA_ECOSYSTEM | 34,170 |
| JS_ECOSYSTEM | 33,205 |
| MOBILE | 30,340 |
| DATABASE | 29,355 |
| PYTHON_ECOSYSTEM | 27,162 |
| DEVOPS | 26,944 |
| AI_ML | 26,091 |
| JOB_POSITION | 20,393 |
| 11 sparse labels | < 100 each |

**After relabelling (2 labels):**

| Label | Spans |
|-------|-------|
| TECHNICAL_SKILL | 354,132 |
| JOB_TITLE | 20,408 |

**Data split:** 90/10 train/test (seed=42)

---

## Model Training

| Setting | Value |
|---------|-------|
| Base model | `urchade/gliner_medium-v2.1` (~170M params) |
| Labels | TECHNICAL_SKILL, JOB_TITLE |
| Loss function | Focal loss (alpha=0.75, gamma=2) |
| Learning rate | 5e-6 (model), 1e-5 (heads) |
| Batch size | 8 per device |
| Epochs | 3 |
| Precision | FP16 |
| Platform | Kaggle T4 x2 |
| Scheduler | Linear with 10% warmup |
| Checkpointing | Best model by eval_loss, top 3 saved |

---

## Evaluation Results

**After data quality improvements (cleaned punctuation + expanded blocklist):**

| Label | Precision | Recall | F1 | F2 | Support |
|-------|-----------|--------|----|----|---------|
| JOB_TITLE | 1.0000 | 0.8553 | 0.9220 | 0.8808 | 304 |
| TECHNICAL_SKILL | 0.9966 | 0.8391 | 0.9111 | 0.8665 | 4,146 |
| **OVERALL (micro)** | **0.9968** | **0.8402** | **0.9118** | **0.8675** | **4,450** |

**Before data quality fixes (for reference):**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| JOB_TITLE | 0.8274 | 0.6151 | 0.7057 |
| TECHNICAL_SKILL | 0.2008 | 0.1710 | 0.1847 |
| OVERALL | 0.2368 | 0.2000 | 0.2169 |

---

## Taxonomy Coverage

**209 skills across 12 categories:**

| Category | Skills | Examples |
|----------|--------|----------|
| PROGRAMMING_LANGUAGE | 25 | Python, Java, Go, Rust, TypeScript |
| DEVOPS | 25 | Docker, Kubernetes, Terraform, Jenkins, Git |
| AI_ML | 24 | TensorFlow, PyTorch, LangChain, HuggingFace |
| CLOUD | 21 | AWS, Azure, GCP, Lambda, S3, Heroku |
| JS_ECOSYSTEM | 19 | React, Angular, Next.js, Node.js, Express |
| PYTHON_ECOSYSTEM | 18 | Django, FastAPI, Pandas, NumPy, Celery |
| DATABASE | 18 | PostgreSQL, MongoDB, Redis, Snowflake, SQL |
| IT_TOOLS | 18 | Jira, Figma, Postman, Tableau, VS Code |
| BIG_DATA | 14 | Spark, Kafka, Airflow, Hadoop, Flink |
| JAVA_ECOSYSTEM | 11 | Spring Boot, Hibernate, Maven, Gradle |
| DISTRIBUTED_SYSTEMS | 10 | Microservices, gRPC, RabbitMQ, Istio |
| MOBILE | 6 | React Native, Flutter, Android, iOS |

Unmatched skills fall to **UNCATEGORIZED**.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Parse a resume (uses demo text if no args)
uv run python -m src.resume_parser

# Parse custom text
uv run python -m src.resume_parser --text "Expert in Python, AWS, React..."

# Parse from file
uv run python -m src.resume_parser --file resume.txt

# Evaluate model
uv run python eval_model.py
uv run python eval_model.py --sweep     # threshold sweep

# Validate training data
uv run python validate_data.py

# Re-run data relabelling
uv run python -m src.relabel_dataset

# Test taxonomy mapper
uv run python -m src.taxonomy_mapper
```

---

## Dependencies

```toml
[dependencies]
gliner >= 0.2.25           # GLiNER NER model framework
huggingface_hub >= 1.7.0   # Model download/upload
accelerate >= 1.1.0        # Multi-GPU training support

[optional]
rapidfuzz >= 3.9.0         # Fuzzy skill matching
```

---

## Key Design Decisions

1. **Coarse training, fine inference:** Train on 2 labels, restore granularity via taxonomy mapper at inference
2. **Synthetic data:** No privacy concerns, consistent annotations, scalable
3. **Focal loss:** Handles class imbalance (17:1 TECHNICAL_SKILL to JOB_TITLE ratio)
4. **FP16 training:** T4 Tensor Cores provide major FP16 speedup over P100
5. **Chunking:** Splits long resumes to stay under model's 384-token context limit
6. **Word-boundary matching:** Prevents false positives ("java" won't match "somaiya")
7. **Data quality pipeline:** Punctuation stripping + blocklist filtering before training
