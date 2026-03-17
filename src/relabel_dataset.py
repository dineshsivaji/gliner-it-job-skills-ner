"""
relabel_dataset.py
------------------
Collapses fine-grained IT skill labels in synthetic_gliner_dataset.json
into 2 well-supported labels for GLiNER fine-tuning.

Sparse labels (SOFT_SKILL, CERTIFICATION, EDUCATION, YEARS_OF_EXPERIENCE)
are dropped — the base model handles these zero-shot at inference time.

Fine-grained  →  Coarse
──────────────────────────────────────────────────────────
PYTHON_ECOSYSTEM, JS_ECOSYSTEM, JAVA_ECOSYSTEM,
FRAMEWORK, CLOUD, DEVOPS, DATABASE, AI_ML,
MOBILE, IT TOOLS, IT LANGUAGES, IT LIBRARIES,
IT SKILLS, IT TECHNOLOGIES, PROGRAMMING_LANGUAGE  →  TECHNICAL_SKILL

JOB_POSITION, JOB POSITION                        →  JOB_TITLE

All others                                         →  (dropped)
"""

import json
import argparse
from collections import Counter
from pathlib import Path

# ── Label mapping ────────────────────────────────────────────────────────────

LABEL_MAP = {
    # All tech → TECHNICAL_SKILL
    "PYTHON_ECOSYSTEM":     "TECHNICAL_SKILL",
    "JS_ECOSYSTEM":         "TECHNICAL_SKILL",
    "JAVA_ECOSYSTEM":       "TECHNICAL_SKILL",
    "FRAMEWORK":            "TECHNICAL_SKILL",
    "CLOUD":                "TECHNICAL_SKILL",
    "DEVOPS":               "TECHNICAL_SKILL",
    "DATABASE":             "TECHNICAL_SKILL",
    "AI_ML":                "TECHNICAL_SKILL",
    "MOBILE":               "TECHNICAL_SKILL",
    "IT TOOLS":             "TECHNICAL_SKILL",
    "IT LANGUAGES":         "TECHNICAL_SKILL",
    "IT LIBRARIES":         "TECHNICAL_SKILL",
    "IT SKILLS":            "TECHNICAL_SKILL",
    "IT TECHNOLOGIES":      "TECHNICAL_SKILL",
    "PROGRAMMING_LANGUAGE": "TECHNICAL_SKILL",
    "DISTRIBUTED_SYSTEMS":  "TECHNICAL_SKILL",
    "BIG_DATA":             "TECHNICAL_SKILL",

    # Job title variants → JOB_TITLE
    "JOB_TITLE":            "JOB_TITLE",
    "JOB_POSITION":         "JOB_TITLE",
    "JOB POSITION":         "JOB_TITLE",

    # Drop all sparse labels — zero-shot handles these at inference
    "SOFT SKILLS":          None,
    "CERTIFICATION":        None,
    "EDUCATION":            None,
    "YEARS OF EXPERIENCE":  None,
    "JOB TYPE":             None,
}

# ── Span blocklist ───────────────────────────────────────────────────────────
# Section-header tokens and filler words that get mislabelled as TECHNICAL_SKILL.
# Checked after lowercasing + stripping trailing punctuation.

SPAN_BLOCKLIST = {
    # Section headers
    "role", "responsibilities", "qualifications",
    "experience", "requirements", "description",
    "summary", "overview", "about", "duties", "skills",
    # Adjectives / filler
    "required", "preferred", "desired", "minimum", "must", "nice", "bonus",
    "ability", "strong", "excellent", "good", "effective", "high",
    "proficient", "familiar", "knowledge", "understanding",
    # Generic action verbs
    "designing", "developing", "implementing", "building",
    "maintaining", "managing", "leading", "working",
    "creating", "supporting", "delivering", "ensuring",
    "collaborating", "contributing", "utilizing", "leveraging",
    "optimizing", "analyzing", "testing", "deploying",
    # Generic nouns that are not actual skills
    "data", "solutions", "systems", "applications",
    "tools", "services", "platform", "platforms",
    "technologies", "environment", "infrastructure",
    "software", "code", "projects", "team", "business",
    "quality", "performance", "security", "scalability",
    "integration", "development", "engineering", "architecture",
    "drive", "deliver", "work", "closely",
}

VALID_LABELS = {
    "TECHNICAL_SKILL",
    "JOB_TITLE",
}


# ── Punctuation that should never be part of a span boundary ─────────────────

_STRIP_CHARS = " ,.;:!?\"'`()[]{}/-–—"


# ── Normalise examples with raw-string text + char-offset spans ──────────────

def _normalise_example(example: dict) -> dict:
    """
    If tokenized_text is a plain string (char-level) with dict-style spans,
    convert to the standard format: token list + token-index spans.
    """
    text = example["tokenized_text"]
    if isinstance(text, list):
        return example  # already tokenised

    tokens = text.split()
    # Build a mapping from character offset → token index
    char_to_tok = {}
    pos = 0
    for idx, tok in enumerate(tokens):
        for j in range(len(tok)):
            char_to_tok[pos + j] = idx
        pos += len(tok) + 1  # +1 for the space

    new_ner = []
    for span in example["ner"]:
        if isinstance(span, dict):
            start_char, end_char, label = span["start"], span["end"], span["label"]
        else:
            start_char, end_char, label = span
        start_tok = char_to_tok.get(start_char)
        end_tok = char_to_tok.get(end_char)
        if start_tok is not None and end_tok is not None:
            new_ner.append([start_tok, end_tok, label])

    return {"tokenized_text": tokens, "ner": new_ner}


# ── Core re-labelling ────────────────────────────────────────────────────────

def _strip_span_punct(tokens: list[str], start: int, end: int) -> tuple[int, int] | None:
    """
    Trim leading/trailing tokens that are pure punctuation.
    Returns the adjusted (start, end) or None if the span collapses.
    """
    boundary = set(_STRIP_CHARS) - {" "}
    # Shrink from the right while the token is only punctuation
    while end >= start and all(c in boundary for c in tokens[end]):
        end -= 1
    # Shrink from the left while the token is only punctuation
    while start <= end and all(c in boundary for c in tokens[start]):
        start += 1
    if start > end:
        return None
    return start, end


def _span_text(tokens: list[str], start: int, end: int) -> str:
    """Join tokens and strip boundary punctuation for comparison."""
    return " ".join(tokens[start: end + 1]).lower().strip(_STRIP_CHARS)


def _is_blocklisted(tokens: list[str], start: int, end: int) -> bool:
    """Check if a span's text matches the blocklist (section headers / filler)."""
    return _span_text(tokens, start, end) in SPAN_BLOCKLIST


def relabel_example(example: dict) -> dict | None:
    """
    Re-label a single training example.
    Returns None if the example ends up with zero NER spans (e.g. all dropped).
    """
    example = _normalise_example(example)
    tokens = example["tokenized_text"]
    new_ner = []
    for span in example["ner"]:
        start, end, label = span

        # ── Validate indices ──
        if start < 0 or end >= len(tokens) or start > end:
            continue

        new_label = LABEL_MAP.get(label)
        if new_label is None:
            continue  # drop this span (sparse label)

        # ── Trim punctuation from span boundaries ──
        trimmed = _strip_span_punct(tokens, start, end)
        if trimmed is None:
            continue  # span was only punctuation
        start, end = trimmed

        # ── Drop noisy section-header / filler spans ──
        if new_label == "TECHNICAL_SKILL" and _is_blocklisted(tokens, start, end):
            continue

        # ── Drop single-char spans (likely annotation noise) ──
        span_text = _span_text(tokens, start, end)
        if len(span_text) < 2:
            continue

        new_ner.append([start, end, new_label])

    if not new_ner:
        return None  # skip examples that become empty

    return {**example, "ner": new_ner}


def relabel_dataset(input_path: str, output_path: str) -> None:
    print(f"Loading dataset from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    print(f"  Total examples loaded : {len(data)}")

    # ── Before stats ──
    before_counts: Counter = Counter()
    for ex in data:
        for span in ex["ner"]:
            label = span["label"] if isinstance(span, dict) else span[2]
            before_counts[label] += 1

    print("\n── Label distribution BEFORE re-labelling ──")
    for label, count in before_counts.most_common():
        print(f"  {label:30s} {count:>6}")

    # ── Re-label ──
    relabelled = []
    dropped_examples = 0
    for ex in data:
        result = relabel_example(ex)
        if result is None:
            dropped_examples += 1
        else:
            relabelled.append(result)

    # ── After stats ──
    after_counts: Counter = Counter()
    for ex in relabelled:
        for _, _, label in ex["ner"]:
            after_counts[label] += 1

    print(f"\n── Label distribution AFTER re-labelling ──")
    for label, count in after_counts.most_common():
        print(f"  {label:30s} {count:>6}")

    print(f"\n── Summary ──")
    print(f"  Examples before  : {len(data)}")
    print(f"  Examples dropped : {dropped_examples}  (all spans mapped to None — sparse labels dropped)")
    print(f"  Examples after   : {len(relabelled)}")

    # ── Warn on sparse labels ──
    print("\n── Sparsity check (warn if < 100 examples) ──")
    for label in VALID_LABELS:
        count = after_counts.get(label, 0)
        flag = "⚠️  SPARSE" if count < 100 else "✅"
        print(f"  {label:30s} {count:>6}  {flag}")

    # ── Data quality audit ──
    punct_spans = 0
    total_spans = 0
    skill_counts = []
    for ex in relabelled:
        n_skills = 0
        for s, e, label in ex["ner"]:
            total_spans += 1
            span = " ".join(ex["tokenized_text"][s: e + 1])
            if span[-1:] in ",.;:!?" :
                punct_spans += 1
            if label == "TECHNICAL_SKILL":
                n_skills += 1
        skill_counts.append(n_skills)

    avg_skills = sum(skill_counts) / len(skill_counts) if skill_counts else 0
    max_skills = max(skill_counts) if skill_counts else 0
    over_15 = sum(1 for c in skill_counts if c > 15)

    print(f"\n── Data quality audit ──")
    print(f"  Spans with trailing punct : {punct_spans}/{total_spans}")
    print(f"  Avg skills/example        : {avg_skills:.1f}")
    print(f"  Max skills in one example : {max_skills}")
    print(f"  Examples with >15 skills  : {over_15}")

    # ── Save ──
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(relabelled, f, indent=2)
    print(f"\n✅ Saved re-labelled dataset to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-label GLiNER training dataset")
    parser.add_argument(
        "--input",
        default="src/training_data/synthetic_gliner_dataset.json",
        help="Path to original dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="src/training_data/synthetic_gliner_relabelled.json",
        help="Path to write re-labelled dataset JSON",
    )
    args = parser.parse_args()
    relabel_dataset(args.input, args.output)
