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
    "role", "required", "responsibilities", "qualifications",
    "preferred", "experience", "requirements", "description",
    "summary", "overview", "about", "duties", "skills",
    "desired", "minimum", "must", "nice", "bonus",
    "designing", "developing", "implementing", "building",
    "maintaining", "managing", "leading", "working",
    "ability", "strong", "excellent", "good",
}

VALID_LABELS = {
    "TECHNICAL_SKILL",
    "JOB_TITLE",
}


# ── Core re-labelling ────────────────────────────────────────────────────────

def _is_blocklisted(tokens: list[str], start: int, end: int) -> bool:
    """Check if a span's text matches the blocklist (section headers / filler)."""
    span_text = " ".join(tokens[start: end + 1]).lower().strip(" ,.:;-–—")
    return span_text in SPAN_BLOCKLIST


def relabel_example(example: dict) -> dict | None:
    """
    Re-label a single training example.
    Returns None if the example ends up with zero NER spans (e.g. all dropped).
    """
    tokens = example["tokenized_text"]
    new_ner = []
    for span in example["ner"]:
        start, end, label = span
        new_label = LABEL_MAP.get(label)
        if new_label is None:
            continue  # drop this span (sparse label)
        if new_label == "TECHNICAL_SKILL" and _is_blocklisted(tokens, start, end):
            continue  # drop noisy section-header spans
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
        for _, _, label in ex["ner"]:
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
