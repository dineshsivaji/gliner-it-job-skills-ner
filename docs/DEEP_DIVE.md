# GLiNER IT Job Skills NER -- Deep Dive

A comprehensive technical reference for every component, algorithm, design decision, and data flow in this project.

---

## Table of Contents

1. [GLiNER: How It Works](#1-gliner-how-it-works)
2. [Data Pipeline: From Raw to Relabelled](#2-data-pipeline-from-raw-to-relabelled)
3. [Synthetic Data Generation](#3-synthetic-data-generation)
4. [Relabelling Pipeline](#4-relabelling-pipeline)
5. [Model Training](#5-model-training)
6. [Taxonomy Mapper: Design and Algorithms](#6-taxonomy-mapper-design-and-algorithms)
7. [Resume Parser: Inference Pipeline](#7-resume-parser-inference-pipeline)
8. [Evaluation System](#8-evaluation-system)
9. [Data Validation](#9-data-validation)
10. [Data Quality Crisis and Resolution](#10-data-quality-crisis-and-resolution)
11. [Configuration Reference](#11-configuration-reference)
12. [Data Format Specification](#12-data-format-specification)
13. [Deployment and Infrastructure](#13-deployment-and-infrastructure)
14. [Limitations and Future Work](#14-limitations-and-future-work)

---

## 1. GLiNER: How It Works

### 1.1 What is GLiNER?

GLiNER (Generalist and Lightweight Named Entity Recognition) is a span-based NER model that can recognize arbitrary entity types at inference time without retraining. Unlike traditional NER models that have a fixed set of output labels, GLiNER takes entity type descriptions as input alongside the text.

### 1.2 Architecture

```
Input Text:  "Expert in Python and AWS"
Entity Types: ["TECHNICAL_SKILL", "JOB_TITLE"]

                    ┌─────────────────────┐
                    │  Bidirectional       │
  Text tokens  ──>  │  Transformer Encoder │  ──> Token embeddings
  Type tokens  ──>  │  (shared encoder)    │  ──> Type embeddings
                    └─────────────────────┘
                              │
                              v
                    ┌─────────────────────┐
                    │  Span Representation │
                    │  Layer               │
                    │  (all possible spans)│
                    └─────────────────────┘
                              │
                              v
                    ┌─────────────────────┐
                    │  Biaffine Classifier │
                    │  span_repr x type_repr│
                    │  -> match score       │
                    └─────────────────────┘
                              │
                              v
                    Entities with scores
```

**Key components:**

1. **Shared Encoder:** Both text tokens and entity type tokens pass through the same bidirectional transformer. This enables zero-shot generalization -- the model learns the relationship between text spans and type descriptions.

2. **Span Representation:** For every possible (start, end) token pair, the model computes a span representation. For a text of length N, there are O(N^2) possible spans.

3. **Biaffine Classifier:** Computes a score for each (span, entity_type) pair using biaffine attention:
   ```
   score(span, type) = span_repr^T * W * type_repr + bias
   ```
   A span is predicted as an entity if the score exceeds the threshold.

### 1.3 Model Variants Used

| Model | Role | Parameters |
|-------|------|------------|
| `urchade/gliner_medium-v2.1` | Base model for fine-tuning | ~170M |
| `dineshsivaji/gliner-it-job-skills-ner` | Fine-tuned output | ~170M |

The medium variant balances accuracy and inference speed. The base model already has general NER capability; fine-tuning specializes it for IT resume domain.

### 1.4 Zero-Shot vs Fine-Tuned

GLiNER supports both modes:

- **Zero-shot:** Pass any entity type string at inference (e.g., "CERTIFICATION"). The base model handles this using its learned type embeddings.
- **Fine-tuned:** Train on specific labels with domain data. Much higher accuracy for those labels.

This project uses fine-tuning for TECHNICAL_SKILL and JOB_TITLE (high frequency, domain-specific), while relying on zero-shot for optional labels like SOFT_SKILL and EDUCATION.

---

## 2. Data Pipeline: From Raw to Relabelled

### 2.1 Pipeline Overview

```
                    skill_synth_data_gen.py
                    + external JSONL data
                            │
                            v
            synthetic_gliner_dataset.json
            (50,218 examples, 21 labels)
                            │
                            v
                    relabel_dataset.py
                    ├── Label consolidation (17 -> 1, 2 -> 1)
                    ├── Punctuation stripping
                    ├── Blocklist filtering
                    └── Short span removal
                            │
                            v
            synthetic_gliner_relabelled.json
            (37,583 examples, 2 labels)
                            │
                            v
                    validate_data.py
                    (audit for remaining issues)
                            │
                            v
                    train_gliner_resume.py
                    (90/10 split, seed=42)
```

### 2.2 Data Statistics

**Original dataset:**

```
Total examples  : 50,218
Total spans     : 374,584
Unique labels   : 21
Top labels      :
  PROGRAMMING_LANGUAGE : 112,417 (30.0%)
  CLOUD                :  34,300 ( 9.2%)
  JAVA_ECOSYSTEM       :  34,170 ( 9.1%)
  JS_ECOSYSTEM         :  33,205 ( 8.9%)
  MOBILE               :  30,340 ( 8.1%)
  DATABASE              :  29,355 ( 7.8%)
  PYTHON_ECOSYSTEM     :  27,162 ( 7.3%)
  DEVOPS               :  26,944 ( 7.2%)
  AI_ML                :  26,091 ( 7.0%)
  JOB_POSITION         :  20,393 ( 5.4%)
  11 sparse labels     :    < 100 each
```

**After relabelling:**

```
Total examples  : 37,583 (25.2% dropped)
Total spans     : 374,540
Labels          : 2
  TECHNICAL_SKILL : 354,132 (94.5%)
  JOB_TITLE       :  20,408 ( 5.5%)
```

**Per-example statistics:**

```
Token lengths   : min=3, max=118, avg=15
Spans per example: min=1, max=24, avg=10.0
```

---

## 3. Synthetic Data Generation

**File:** `src/skill_synth_data_gen.py` (140 lines)

### 3.1 Generation Strategy

The generator creates training examples by filling sentence templates with random skills and job positions, then computing token-level NER annotations.

**Components:**

| Component | Count | Purpose |
|-----------|-------|---------|
| Skill categories | 9 | PROGRAMMING_LANGUAGE through CLOUD |
| Total skill values | 86 | Individual skills across categories |
| Job positions | 10 | "Software Engineer", "SRE", etc. |
| Templates | 5 | Sentence patterns with `{skills}` and `{position}` placeholders |
| Target samples | 50,000 | Total generated examples |
| Negative ratio | 15% | Examples with zero entities |

### 3.2 Template System

```python
TEMPLATES = [
    "Expertise in {skills} is required for this role.",
    "Hands-on experience with {skills} required.",
    "Looking for a {position} with skills in {skills}.",
    "The ideal candidate has worked with {skills}.",
    "Required: {skills}."
]
```

For each positive example:
1. Select 2-5 random skills from random categories
2. Select a random job position
3. Select a random template
4. Fill the template and tokenize (whitespace split)
5. Use sliding-window matching to find entity spans at the token level

### 3.3 Entity Detection Algorithm

```python
# For each skill value, slide a window across tokens:
for i in range(len(tokens) - val_len + 1):
    # Strip punctuation for matching
    sub_section = " ".join([t.strip(",.;") for t in tokens[i:i + val_len]])
    if sub_section.lower() == val.lower():
        entities.append([i, i + val_len - 1, category])
```

This tokenization-aware approach ensures span indices align with the whitespace-tokenized text that GLiNER expects.

### 3.4 External Data Integration

The generator also appends data from an external `it_training_data.jsonl` file, converting field names:

```python
data["tokenized_text"] = data.pop("text")
data["ner"] = data.pop("spans")
```

### 3.5 Negative Examples

15% of examples are negative (no entities):

```python
text = "The candidate should have strong communication skills and a growth mindset."
dataset.append({"tokenized_text": text.split(), "ner": []})
```

Negatives are critical for training -- they teach the model that not every sentence contains an entity. Without negatives, the model becomes overly aggressive in tagging.

---

## 4. Relabelling Pipeline

**File:** `src/relabel_dataset.py` (266 lines)

### 4.1 Label Consolidation

The core idea: train on coarse labels for better generalization, restore granularity at inference via taxonomy mapping.

**Mapping (24 rules):**

```
17 source labels  -->  TECHNICAL_SKILL
 2 source labels  -->  JOB_TITLE
 5 source labels  -->  Dropped (None)
```

Source labels mapped to TECHNICAL_SKILL:
- PYTHON_ECOSYSTEM, JS_ECOSYSTEM, JAVA_ECOSYSTEM, FRAMEWORK
- CLOUD, DEVOPS, DATABASE, AI_ML, MOBILE
- IT TOOLS, IT LANGUAGES, IT LIBRARIES, IT SKILLS, IT TECHNOLOGIES
- PROGRAMMING_LANGUAGE, DISTRIBUTED_SYSTEMS, BIG_DATA

Source labels mapped to JOB_TITLE:
- JOB_POSITION, JOB POSITION

Source labels dropped:
- SOFT SKILLS, CERTIFICATION, EDUCATION, YEARS OF EXPERIENCE, JOB TYPE

**Rationale for dropping sparse labels:**
These labels had < 100 examples in the original dataset. Training on them would cause overfitting and hurt performance on the well-supported labels. The base GLiNER model can still recognize them zero-shot at inference time.

### 4.2 Span Blocklist (76 entries)

A curated set of words that frequently get mislabelled as TECHNICAL_SKILL in synthetic data. Organized into four groups:

**Section headers (11):**
```
role, responsibilities, qualifications, experience, requirements,
description, summary, overview, about, duties, skills
```

**Adjectives/filler (17):**
```
required, preferred, desired, minimum, must, nice, bonus,
ability, strong, excellent, good, effective, high,
proficient, familiar, knowledge, understanding
```

**Generic action verbs (20):**
```
designing, developing, implementing, building,
maintaining, managing, leading, working,
creating, supporting, delivering, ensuring,
collaborating, contributing, utilizing, leveraging,
optimizing, analyzing, testing, deploying
```

**Generic nouns (28):**
```
data, solutions, systems, applications, tools, services,
platform, platforms, technologies, environment, infrastructure,
software, code, projects, team, business, quality, performance,
security, scalability, integration, development, engineering,
architecture, drive, deliver, work, closely
```

### 4.3 Punctuation Stripping

**The problem:** The synthetic data generator creates tokens with trailing punctuation attached (e.g., `"Python,"` as a single token). When the model learns to predict entities, it learns to include the punctuation as part of the entity span. At inference, the model's tokenizer may split differently, causing mismatches.

**The fix:** A two-level cleaning approach.

**Level 1: Token-level stripping** (`_strip_span_punct`)

```python
_STRIP_CHARS = " ,.;:!?\"'`()[]{}/-–—"

def _strip_span_punct(tokens, start, end):
    boundary = set(_STRIP_CHARS) - {" "}
    # Remove pure-punctuation tokens from right
    while end >= start and all(c in boundary for c in tokens[end]):
        end -= 1
    # Remove pure-punctuation tokens from left
    while start <= end and all(c in boundary for c in tokens[start]):
        start += 1
    if start > end:
        return None  # span was all punctuation
    return start, end
```

This handles cases where punctuation is a separate token: `["Python", ","]` -> `["Python"]`.

**Level 2: Text-level stripping** (`_span_text`)

```python
def _span_text(tokens, start, end):
    return " ".join(tokens[start: end + 1]).lower().strip(_STRIP_CHARS)
```

This handles cases where punctuation is attached: `["Python,"]` -> `"python"` (after strip).

### 4.4 Processing Pipeline per Example

For each example in the dataset:

```
1. For each span [start, end, label]:
   a. Validate: start >= 0, end < len(tokens), start <= end
   b. Map label: LABEL_MAP[label] -> new_label (or None to drop)
   c. Strip punctuation: _strip_span_punct(tokens, start, end)
   d. Blocklist check: skip if _span_text matches SPAN_BLOCKLIST
   e. Length check: skip if normalized text < 2 characters
   f. Keep: append [start, end, new_label]

2. If no spans remain, drop the entire example
```

### 4.5 Data Quality Audit

After relabelling, the script prints a quality audit:

```
── Data quality audit ──
  Spans with trailing punct : 0/374540
  Avg skills/example        : 9.4
  Max skills in one example : 24
  Examples with >15 skills  : 523
```

---

## 5. Model Training

**File:** `src/train_gliner_resume.py` (188 lines)

### 5.1 Training Configuration

```python
trainer = model.train_model(
    train_dataset     = train_dataset,        # 90% of data (~33,824 examples)
    eval_dataset      = test_dataset,         # 10% of data (~3,759 examples)
    output_dir        = "checkpoints",
    learning_rate     = 5e-6,                 # Main model parameters
    weight_decay      = 0.01,                 # L2 regularization
    others_lr         = 1e-5,                 # Classification head parameters
    others_weight_decay = 0.01,
    lr_scheduler_type = "linear",             # Linear decay after warmup
    warmup_ratio      = 0.1,                  # 10% of training for warmup
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    focal_loss_alpha  = 0.75,                 # Positive class weight
    focal_loss_gamma  = 2,                    # Hard example focus
    num_train_epochs  = 3,
    eval_strategy     = "steps",
    eval_steps        = 200,                  # Evaluate every 200 steps
    save_steps        = 200,                  # Save checkpoint every 200 steps
    save_total_limit  = 3,                    # Keep top 3 checkpoints
    load_best_model_at_end = True,            # Restore best checkpoint
    metric_for_best_model  = "eval_loss",     # Select by lowest eval loss
    greater_is_better      = False,
    dataloader_num_workers = 0,
    use_cpu           = False,                # GPU if available
    fp16              = True,                 # Half-precision for Tensor Cores
    report_to         = "none",              # No W&B/TensorBoard
)
```

### 5.2 Focal Loss

Standard cross-entropy treats all misclassifications equally. Focal loss addresses two problems:

1. **Class imbalance:** Most spans are negative (not entities). Alpha weights the positive class higher.
2. **Easy negatives:** Most negative spans are trivially easy (e.g., "the", "and"). Gamma reduces loss for high-confidence predictions, forcing the model to focus on hard examples.

**Formula:**

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
  p_t   = model's predicted probability for the correct class
  alpha = 0.75 for positive class (entities), 0.25 for negative
  gamma = 2    (hard example focusing parameter)
```

**Effect of gamma:**

| Predicted Confidence | gamma=0 (CE) | gamma=2 (FL) | Reduction |
|---------------------|---------------|---------------|-----------|
| 0.9 (easy correct)  | 0.105         | 0.001         | 99%       |
| 0.5 (uncertain)     | 0.693         | 0.173         | 75%       |
| 0.1 (hard mistake)  | 2.303         | 1.864         | 19%       |

At gamma=2, easy examples contribute almost nothing to the loss, and the model concentrates its learning capacity on genuinely ambiguous cases.

### 5.3 Dual Learning Rates

The model uses two learning rates:

- **`learning_rate = 5e-6`** for the transformer backbone (slow, preserves pre-trained knowledge)
- **`others_lr = 1e-5`** for the span classifier and type embeddings (faster, these need more adaptation)

This is standard practice in transfer learning: pre-trained layers should change slowly to avoid catastrophic forgetting, while task-specific layers need faster learning.

### 5.4 Training Schedule

```
Step 0         warmup_ratio*total_steps        total_steps
  |───── Warmup (linear) ─────|───── Decay (linear) ─────|
  0                          LR_max                       0
```

The linear scheduler increases the learning rate from 0 to LR_max during the first 10% of training (warmup), then linearly decreases it to 0. Warmup prevents large gradient updates in the first few steps when the model hasn't adapted to the new data distribution.

### 5.5 Multi-GPU Support

```python
num_gpus = torch.cuda.device_count()
effective_batch = args.batch_size * max(1, num_gpus)
```

With `torchrun --nproc_per_node=2`, each GPU processes `batch_size=8` examples, giving an effective batch of 16. The Hugging Face Trainer handles gradient synchronization via PyTorch DDP (DistributedDataParallel).

### 5.6 HuggingFace Upload

After training, the model is uploaded to the Hugging Face Hub:

```python
# Token resolution order:
1. HF_TOKEN environment variable
2. Kaggle secrets (UserSecretsClient)
3. Skip upload if neither available
```

The upload uses `HfApi.upload_folder()` which uploads all model files (config, weights, tokenizer) to the configured repo.

### 5.7 FP16 Training

Enabled when CUDA GPUs are available. Uses Tensor Cores on T4/A100/V100 GPUs for significant training speedup with negligible accuracy loss. The Trainer handles mixed-precision automatically via `torch.cuda.amp`.

---

## 6. Taxonomy Mapper: Design and Algorithms

**File:** `src/taxonomy_mapper.py` (462 lines)

### 6.1 Taxonomy Structure

209 skill entries mapped to 12 categories:

```
PROGRAMMING_LANGUAGE (25): python, java, c#, c++, go, golang, rust,
    ruby, php, swift, kotlin, scala, r, matlab, typescript, javascript,
    bash, shell, powershell, perl, dart, elixir, haskell, lua, groovy

PYTHON_ECOSYSTEM (18): django, flask, fastapi, sqlalchemy, celery,
    pydantic, numpy, pandas, scipy, matplotlib, seaborn, pytest,
    poetry, uvicorn, gunicorn, alembic, aiohttp, asyncio

JS_ECOSYSTEM (19): react, vue, angular, next.js, nextjs, nuxt, svelte,
    node.js, nodejs, express, nest.js, nestjs, webpack, vite, jest,
    graphql, redux, tailwind, tailwindcss

JAVA_ECOSYSTEM (11): spring, spring boot, spring mvc, hibernate,
    maven, gradle, junit, mockito, quarkus, micronaut, struts

AI_ML (24): tensorflow, pytorch, keras, scikit-learn, sklearn,
    xgboost, lightgbm, hugging face, huggingface, transformers,
    langchain, openai, llm, bert, gpt, rag, mlflow, wandb,
    computer vision, nlp, machine learning, deep learning,
    neural network, gliner

CLOUD (21): aws, azure, gcp, google cloud, s3, ec2, lambda,
    cloudformation, pulumi, cloud functions, app service,
    azure functions, cosmosdb, azure ad, keyvault, firebase,
    cloudflare, vercel, netlify, heroku, digitalocean

DEVOPS (25): docker, kubernetes, k8s, helm, terraform, jenkins,
    github actions, gitlab ci, azure devops, circleci, ansible, chef,
    puppet, prometheus, grafana, elk, elasticsearch, logstash, kibana,
    datadog, nginx, linux, unix, ci/cd, git

DATABASE (18): postgresql, postgres, mysql, sqlite, mongodb, redis,
    cassandra, dynamodb, oracle, sql server, mssql, neo4j, influxdb,
    snowflake, bigquery, redshift, sql, nosql

MOBILE (6): react native, flutter, android, ios, xcode, expo

IT_TOOLS (18): jira, confluence, figma, postman, swagger, vs code,
    intellij, visual studio, github, gitlab, bitbucket, notion,
    slack, sonarqube, splunk, tableau, power bi, excel

BIG_DATA (14): apache spark, spark, hadoop, flink, kafka, airflow,
    clickhouse, hive, presto, dbt, etl, data pipeline,
    data engineering, data warehouse

DISTRIBUTED_SYSTEMS (10): distributed systems, distributed computing,
    microservices, grpc, message queues, rabbitmq, service mesh,
    istio, load balancing, event driven
```

### 6.2 Three-Tier Matching Algorithm

The `map()` method uses a priority-ordered matching strategy:

```
Input: "FastAPI"

Tier 1: Exact Match
  normalize("FastAPI") = "fastapi"
  "fastapi" in taxonomy? YES -> "PYTHON_ECOSYSTEM"
  DONE

Input: "AWS Lambda"

Tier 1: Exact Match
  normalize("AWS Lambda") = "aws lambda"
  "aws lambda" in taxonomy? NO

Tier 2: Word-Boundary Match
  For each key in taxonomy:
    len <= 2 (e.g., "r", "go"): exact match only -> skip
    len > 2: use regex \baws\b on input
      "aws" matches! -> candidate (len=3)
    \blambda\b on input
      "lambda" matches! -> candidate (len=6)

  Pick longest candidate: "lambda" (len=6) > "aws" (len=3)
  Return taxonomy["lambda"] = "CLOUD"
  DONE

Input: "React Native development"

Tier 1: Exact Match
  normalize = "react native development"
  "react native development" in taxonomy? NO

Tier 2: Word-Boundary Match
  For each key in taxonomy:
    len <= 2 (e.g., "r", "go"): exact match only -> skip
    len > 2: use regex \breact native\b on input
      "react native" matches! -> candidate
    "react" also matches -> candidate

  Pick longest candidate: "react native" (len=12) > "react" (len=5)
  Return taxonomy["react native"] = "MOBILE"
  DONE

Input: "Rakt" (misspelling)

Tier 1: Exact match? NO
Tier 2: Word-boundary match? NO
Tier 3: Fuzzy Match (if enabled)
  rapidfuzz.extractOne("rakt", all_keys, score_cutoff=80)
  Best match: "react" score=80.0 -> "JS_ECOSYSTEM"
  DONE (if score >= threshold)

Otherwise: "UNCATEGORIZED"
```

### 6.3 Word-Boundary Regex

The mapper pre-compiles regex patterns for every key longer than 2 characters:

```python
self._patterns = {
    k: re.compile(r'\b' + re.escape(k) + r'\b')
    for k in self._keys if len(k) > 2
}
```

**Why word boundaries matter:**

Without `\b` anchors:
- "java" would match inside "javascript" -> wrong
- "spring" would match inside "offspring" -> wrong
- "react" would match inside "react native" (both should be candidates, but the longer match wins)

Short keys (1-2 chars like "r", "go", "c#") are restricted to exact match only — they skip the word-boundary tier entirely. This means "r" can't match inside "master" and "go" can't match inside "mongoose". Note that "golang" has its own separate taxonomy entry, so `mapper.map("golang")` returns PROGRAMMING_LANGUAGE via exact match regardless.

### 6.4 Longest-Match Priority

When multiple taxonomy keys match via word-boundary regex, the mapper picks the **longest match**:

```python
best_key = max(candidates, key=len)
```

Example: Input "spring boot application"
- "spring" matches (len=6) -> JAVA_ECOSYSTEM
- "spring boot" matches (len=11) -> JAVA_ECOSYSTEM
- Winner: "spring boot" (more specific)

### 6.5 Normalization

```python
@staticmethod
def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
```

Simple lowercase + whitespace collapse. Does not strip punctuation here (unlike relabel_dataset) because the mapper handles user-facing input that's already fairly clean.

### 6.6 Enrich and Summary APIs

**`enrich(entities)`**: Takes raw GLiNER output, adds `"category"` field to TECHNICAL_SKILL entities:

```python
# Input:
[{"text": "React", "label": "TECHNICAL_SKILL", "score": 0.91}]

# Output:
[{"text": "React", "label": "TECHNICAL_SKILL", "category": "JS_ECOSYSTEM", "score": 0.91}]
```

**`summary(entities)`**: Groups skills by category for display:

```python
{
    "JS_ECOSYSTEM": ["React"],
    "CLOUD": ["AWS", "Azure"],
    "JOB_TITLE": ["Software Engineer"]
}
```

---

## 7. Resume Parser: Inference Pipeline

**File:** `src/resume_parser.py` (254 lines)

### 7.1 Configuration

```python
MODEL_ID   = "dineshsivaji/gliner-it-job-skills-ner"
THRESHOLD  = 0.75          # Score threshold for entity extraction
CHUNK_MAX  = 1200          # Max characters per chunk (~300 tokens)
```

### 7.2 Text Chunking Algorithm

GLiNER has a maximum context length of 384 tokens. Resumes often exceed this. The chunker splits text while preserving semantic boundaries.

```
Input: Full resume text (may be 1000+ tokens)

Step 1: Split on double newlines (paragraph boundaries)
        "Education\n\nSkills\n\nExperience" -> 3 paragraphs

Step 2: For each paragraph > CHUNK_MAX:
        Split on sentence boundaries: (?<=[.\n])\s+

Step 3: For each sentence > CHUNK_MAX:
        Split at word boundaries (find last space before limit)

Step 4: Filter empty chunks
```

**Priority order:** paragraph > sentence > word boundary. This keeps related skills together (e.g., "Skills" section stays as one chunk).

### 7.3 Entity Extraction

```python
for chunk in chunks:
    # Run trained labels at normal threshold
    raw_entities.extend(
        self.model.predict_entities(chunk, TRAINED_LABELS, threshold=self.threshold)
    )
    # Run zero-shot labels at higher threshold
    if ZEROSHOT_LABELS:
        raw_entities.extend(
            self.model.predict_entities(chunk, ZEROSHOT_LABELS, threshold=ZEROSHOT_THRESHOLD)
        )
```

**Dual threshold strategy:**
- Trained labels (TECHNICAL_SKILL, JOB_TITLE): 0.75 threshold
- Zero-shot labels (SOFT_SKILL, etc.): 0.75 threshold (currently same as trained; can be raised independently to reduce false positives on unseen labels)

### 7.4 Deduplication

When text is chunked, the same entity may appear in multiple chunks. The parser deduplicates by (text, label) and keeps the highest confidence score:

```python
seen: dict[tuple[str, str], float] = {}
for e in raw_entities:
    key = (e["text"].strip().lower(), e["label"])
    if key not in seen or e["score"] > seen[key]:
        seen[key] = e["score"]
```

### 7.5 Post-Processing

After deduplication and taxonomy enrichment:

1. **Section prefix removal:** Strips patterns like "Languages:", "Skills:" from extracted text
   ```python
   cleaned = re.sub(r"^(languages?|skills?)\s*:\s*", "", text, flags=re.IGNORECASE)
   ```

2. **Comma splitting:** Breaks multi-skill spans into atomic skills
   ```python
   parts = re.split(r"[,\u2022]", cleaned)  # comma or bullet point
   ```

3. **Per-skill categorization:** Each atomic skill gets its own taxonomy lookup
   ```python
   for part in parts:
       skill = part.strip(" -–—").strip()
       cat = self.mapper.map(skill)
       result["TECHNICAL_SKILL"].setdefault(cat, [])
   ```

### 7.6 Output Structure

```python
{
    "TECHNICAL_SKILL": {           # dict of category -> [skills]
        "PROGRAMMING_LANGUAGE": ["Python", "Java"],
        "CLOUD": ["AWS", "Azure"],
        ...
    },
    "JOB_TITLE": ["Software Engineer", "Tech Lead"],  # flat list
    "SOFT_SKILL": [],              # empty if zero-shot disabled
    "CERTIFICATION": [],
    "EDUCATION": [],
    "YEARS_OF_EXPERIENCE": [],
}
```

### 7.7 Demo Text

The parser includes a built-in demo resume for testing:

```
Deep M. Mehta | Full Stack Software Developer
...
Languages: C#, Java, Python, Ruby, JavaScript, SQL
Cloud: Azure, AWS, Firebase
DevOps: Azure-DevOps, GitHub Actions, Terraform, Ansible, Kubernetes
...
```

Run with no arguments to parse this demo:
```bash
uv run python -m src.resume_parser
```

---

## 8. Evaluation System

**File:** `eval_model.py` (265 lines)

### 8.1 Evaluation Method: Exact Span Match

A predicted span is a **True Positive** only if BOTH the normalized text AND the label exactly match a gold span. Partial matches (e.g., predicting "React" when gold is "React Native") count as both a False Positive AND a False Negative.

### 8.2 Span Normalization

Both gold and predicted spans pass through the same normalization:

```python
def _normalize_span(text: str) -> str:
    text = text.lower().strip()
    text = text.strip(" ,.;:!?\"'`()[]{}/-–—")   # same chars as relabel_dataset
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

This is critical -- using a different normalization in evaluation vs training was the root cause of the initial poor F1 scores (0.18 for TECHNICAL_SKILL).

### 8.3 Metrics

**Precision:** Of all predicted spans, how many are correct.

```
P = TP / (TP + FP)
```

**Recall:** Of all gold spans, how many were found.

```
R = TP / (TP + FN)
```

**F1 (balanced):** Harmonic mean of precision and recall.

```
F1 = 2 * P * R / (P + R)
```

**F2 (recall-weighted):** Penalizes missed entities more than false positives.

```
F2 = 5 * P * R / (4 * P + R)
```

F2 is useful for resume parsing where missing a skill (FN) is typically worse than including a false one (FP) -- a missed skill means the candidate might not be matched to a relevant job.

**Micro-averaging:** TP/FP/FN are pooled across all labels before computing metrics. This weights each label proportionally to its frequency (TECHNICAL_SKILL dominates due to 94.5% share).

### 8.4 Threshold Sweep

```bash
uv run python eval_model.py --sweep
```

Evaluates at thresholds [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] to show the precision/recall tradeoff curve.

Lower threshold -> higher recall (fewer missed entities) but lower precision (more false positives).

### 8.5 Test Split Consistency

The evaluation uses the exact same split as training:

```python
random.seed(42)          # Same seed as train_gliner_resume.py
random.shuffle(data)
split = int(len(data) * (1 - 0.1))  # Same 10% test split
test_data = data[split:]
```

This ensures no data leakage between training and evaluation.

---

## 9. Data Validation

**File:** `validate_data.py` (194 lines)

### 9.1 Checks Performed

| Check | What It Catches |
|-------|----------------|
| `trailing_punct` | Spans ending with `,` `.` `;` etc. |
| `leading_punct` | Spans starting with punctuation |
| `blocklisted` | TECHNICAL_SKILL spans matching generic words |
| `short_span` | Spans < 2 characters after normalization |
| `invalid_indices` | Out-of-bounds start/end indices |
| `high_density_examples` | Examples with > 15 TECHNICAL_SKILL spans |

### 9.2 Consistency

The validator imports normalization constants directly from `relabel_dataset.py`:

```python
from src.relabel_dataset import SPAN_BLOCKLIST, _STRIP_CHARS
BOUNDARY_PUNCT = set(_STRIP_CHARS) - {" "}
```

This ensures validation checks exactly match what the training pipeline does.

### 9.3 Fix Mode

```bash
uv run python validate_data.py --fix
```

Re-runs the entire relabelling pipeline with updated filters and validates the result.

---

## 10. Data Quality Crisis and Resolution

### 10.1 The Problem

Initial evaluation showed catastrophic TECHNICAL_SKILL performance:

```
TECHNICAL_SKILL: P=0.20, R=0.17, F1=0.18
```

### 10.2 Root Causes Identified

**Cause 1: Trailing punctuation in training spans**

The synthetic data generator produced tokens with attached punctuation:

```
Gold span: "Groovy,"   (comma included)
Predicted:  "Groovy"   (no comma)
Result:     FP + FN    (exact match fails)
```

A large proportion of TECHNICAL_SKILL spans had trailing punctuation.

**Cause 2: Generic words labelled as TECHNICAL_SKILL**

The original blocklist (32 entries) missed many common words:

```
"data"          -> TECHNICAL_SKILL  (should be filtered)
"developing"    -> TECHNICAL_SKILL  (should be filtered)
"solutions"     -> TECHNICAL_SKILL  (should be filtered)
"implementing"  -> TECHNICAL_SKILL  (should be filtered)
```

The model learned to aggressively tag generic words, destroying precision.

**Cause 3: Evaluation normalization mismatch**

The evaluation script used `str.lower().strip()` while training data had punctuation-attached tokens. Even correct predictions were counted as mismatches.

### 10.3 Fixes Applied

| Fix | Change | Impact |
|-----|--------|--------|
| Punctuation stripping | Added `_strip_span_punct()` + `_span_text()` | Removed all trailing/leading punct from spans |
| Blocklist expansion | 32 -> 76 entries | Removed generic verbs, nouns, section headers |
| Minimum span length | Skip spans < 2 chars | Removed noise annotations |
| Bounds validation | Check start/end indices | Prevent crashes on malformed data |
| Eval normalization | Use same `_STRIP_CHARS` as training | Fair comparison between gold and predicted |
| Empty span filtering | Skip empty strings after normalization | Clean metric computation |

### 10.4 Results

```
Before:  TECHNICAL_SKILL F1 = 0.1847
After:   TECHNICAL_SKILL F1 = 0.9111  (+393%)

Before:  Overall F1 = 0.2169
After:   Overall F1 = 0.9118  (+320%)
```

The data quality fix was the single most impactful change. No hyperparameters or model architecture were modified.

---

## 11. Configuration Reference

### 11.1 relabel_dataset.py

| Constant | Type | Value | Purpose |
|----------|------|-------|---------|
| `LABEL_MAP` | dict | 24 entries | Maps source labels to TECHNICAL_SKILL, JOB_TITLE, or None |
| `SPAN_BLOCKLIST` | set | 76 entries | Words to filter from TECHNICAL_SKILL spans |
| `VALID_LABELS` | set | 2 entries | Expected output labels |
| `_STRIP_CHARS` | str | 20 chars | Punctuation to strip from span boundaries |

### 11.2 train_gliner_resume.py

| Parameter | Default | CLI Flag | Env Var |
|-----------|---------|----------|---------|
| Training data path | `src/training_data/synthetic_gliner_relabelled.json` | `--data` | `TRAIN_PATH` |
| Output directory | `models/gliner-it-job-skills-ner` | `--output` | `OUTPUT_DIR` |
| Base model | `urchade/gliner_medium-v2.1` | `--base-model` | - |
| HF repo | `dineshsivaji/gliner-it-job-skills-ner` | `--hf-repo` | - |
| Epochs | 3 | `--epochs` | - |
| Batch size | 8 | `--batch-size` | - |
| Learning rate | 5e-6 | `--lr` | - |
| Others LR | 1e-5 | `--others-lr` | - |
| Warmup ratio | 0.1 | `--warmup-ratio` | - |
| Eval steps | 200 | `--eval-steps` | - |
| Seed | 42 | `--seed` | - |
| Skip upload | False | `--no-upload` | - |
| HF token | - | - | `HF_TOKEN` |

### 11.3 resume_parser.py

| Constant | Value | Purpose |
|----------|-------|---------|
| `MODEL_ID` | `dineshsivaji/gliner-it-job-skills-ner` | HuggingFace model to load |
| `THRESHOLD` | 0.75 | Score threshold for trained labels |
| `CHUNK_MAX` | 1200 | Max characters per text chunk |
| `ZEROSHOT_THRESHOLD` | 0.75 | Score threshold for zero-shot labels |

### 11.4 eval_model.py

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_MODEL` | `dineshsivaji/gliner-it-job-skills-ner` | Model to evaluate |
| `DEFAULT_THRESHOLD` | 0.5 | Default evaluation threshold |
| `TEST_SPLIT` | 0.1 | Must match training split |
| `RANDOM_SEED` | 42 | Must match training seed |
| `MAX_EVAL_SAMPLES` | 500 | Cap for fast evaluation |
| `SWEEP_THRESHOLDS` | [0.3 - 0.9] | Thresholds for P/R curve |

---

## 12. Data Format Specification

### 12.1 GLiNER Training Format

```json
{
  "tokenized_text": ["Expert", "in", "Python", "and", "AWS"],
  "ner": [
    [2, 2, "TECHNICAL_SKILL"],
    [4, 4, "TECHNICAL_SKILL"]
  ]
}
```

**Fields:**
- `tokenized_text`: List of string tokens (whitespace-split)
- `ner`: List of `[start_index, end_index, label]` triples
  - Indices are 0-based, inclusive on both ends
  - `start_index` and `end_index` refer to positions in `tokenized_text`

### 12.2 GLiNER Inference Output

```python
model.predict_entities(text, labels, threshold=0.5)
# Returns:
[
    {"text": "Python", "label": "TECHNICAL_SKILL", "score": 0.95, "start": 10, "end": 16},
    {"text": "AWS",    "label": "TECHNICAL_SKILL", "score": 0.88, "start": 21, "end": 24},
]
```

**Fields:**
- `text`: The extracted entity text (substring of input)
- `label`: The predicted entity type
- `score`: Confidence score (0 to 1)
- `start`/`end`: Character offsets in the input string

### 12.3 Resume Parser Output

```json
{
  "TECHNICAL_SKILL": {
    "PROGRAMMING_LANGUAGE": ["Python", "Java"],
    "CLOUD": ["AWS"],
    "UNCATEGORIZED": ["some-unknown-tool"]
  },
  "JOB_TITLE": ["Software Engineer"],
  "SOFT_SKILL": [],
  "CERTIFICATION": [],
  "EDUCATION": [],
  "YEARS_OF_EXPERIENCE": []
}
```

---

## 13. Deployment and Infrastructure

### 13.1 Training Environment

```
Platform : Kaggle Notebooks
GPU      : NVIDIA T4 x2 (16GB VRAM each)
Runtime  : Python 3.10+
Package  : uv (Astral's package manager)
Secrets  : HF_TOKEN via Kaggle Secrets
```

**Why Kaggle:** Free GPU access, reproducible environments, integrated with Hugging Face ecosystem.

**Why T4 x2 over P100:**
- T4 has Tensor Cores (65 TFLOPS FP16 vs 18.7 TFLOPS on P100)
- Two T4s double effective batch size
- Combined T4 x2 FP16 throughput is ~7x that of a single P100

### 13.2 Inference Environment

```
Platform : Any machine with Python 3.10+
GPU      : Optional (CPU is acceptable for single resumes)
Model    : Downloaded from Hugging Face Hub on first run (~780MB)
RAM      : 4GB minimum
```

### 13.3 Model Distribution

```
Training (Kaggle)
    |
    v
model.save_pretrained("models/gliner-it-job-skills-ner")
    |
    v
HfApi.upload_folder() -> huggingface.co/dineshsivaji/gliner-it-job-skills-ner
    |
    v
GLiNER.from_pretrained("dineshsivaji/gliner-it-job-skills-ner")  (any machine)
```

### 13.4 Dependency Tree

```
gliner >= 0.2.25
  ├── torch
  ├── transformers
  └── tokenizers
huggingface_hub >= 1.7.0
  └── requests
accelerate >= 1.1.0
  └── torch
rapidfuzz >= 3.9.0 (optional, dev only)
```

---

## 14. Limitations and Future Work

### 14.1 Current Limitations

**Data:**
- Training data is entirely synthetic (template-based + external JSONL)
- Only English language support
- Skill taxonomy is manually curated (209 entries) -- new skills require manual addition

**Model:**
- 384 token context limit requires chunking for long documents
- Exact span matching can't handle abbreviations (model must learn them)
- No coreference resolution ("it" referring to a previously mentioned skill)

**Inference:**
- Text-only input (no PDF/DOCX parsing)
- No REST API (CLI only)
- No batch processing mode
- Model download (~780MB) on first run

**Evaluation:**
- Evaluated only on synthetic test data (same distribution as training)
- No real-world resume evaluation benchmark
- Exact span match is strict (partial matches count as errors)

### 14.2 Potential Improvements

**Short-term:**
- Lower inference threshold from 0.75 to 0.65 to improve recall
- Enable zero-shot labels (SOFT_SKILL, CERTIFICATION, EDUCATION)
- Add PDF/DOCX text extraction layer
- Wrap in a REST API (FastAPI)

**Medium-term:**
- Collect and annotate real resume data for evaluation
- Add active learning loop (flag low-confidence predictions for review)
- Export to ONNX for 2-5x inference speedup
- Add batch processing for multiple resumes
- Integrate with W&B for experiment tracking

**Long-term:**
- Train on multilingual resumes (use `gliner_multi-v2.1` base)
- Add skill relationship graph (co-occurrence, hierarchy)
- Contextual disambiguation ("React" vs "React Native")
- Temporal skill extraction ("5 years of Python experience")
- Confidence calibration (Platt scaling or temperature scaling)

### 14.3 Scaling Considerations

| Scale | Approach |
|-------|----------|
| 1-10 resumes | CLI, single process |
| 10-1000 resumes | Batch processing with GPU |
| 1000+ resumes | ONNX export + async worker pool |
| Production | REST API behind load balancer, model cached in memory |
