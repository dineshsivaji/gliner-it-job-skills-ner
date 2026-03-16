"""
validate_data.py
----------------
Audits a relabelled GLiNER dataset for common annotation problems:
  - Spans with trailing/leading punctuation
  - Spans that match the blocklist (generic words)
  - Very short spans (<2 chars)
  - Examples with abnormally high skill density

Run BEFORE training to catch data quality issues early.

Usage:
    python validate_data.py
    python validate_data.py --data path/to/relabelled.json
    python validate_data.py --fix --output path/to/cleaned.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.relabel_dataset import SPAN_BLOCKLIST, _STRIP_CHARS

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DATA = "src/training_data/synthetic_gliner_relabelled.json"
MAX_SKILLS_PER_EXAMPLE = 15
BOUNDARY_PUNCT = set(_STRIP_CHARS) - {" "}


# ── Checks ───────────────────────────────────────────────────────────────────

def _normalize(tokens, start, end):
    """Same normalization as relabel_dataset._span_text()."""
    return " ".join(tokens[start: end + 1]).lower().strip(_STRIP_CHARS)


def check_trailing_punct(tokens, start, end):
    span = " ".join(tokens[start: end + 1])
    if span and span[-1] in BOUNDARY_PUNCT:
        return span
    return None


def check_leading_punct(tokens, start, end):
    span = " ".join(tokens[start: end + 1])
    if span and span[0] in BOUNDARY_PUNCT:
        return span
    return None


def check_blocklisted(tokens, start, end):
    span_text = _normalize(tokens, start, end)
    if span_text in SPAN_BLOCKLIST:
        return span_text
    return None


def check_short_span(tokens, start, end):
    span_text = _normalize(tokens, start, end)
    if len(span_text) < 2:
        return span_text
    return None


# ── Main audit ───────────────────────────────────────────────────────────────

def audit(data: list[dict]) -> dict:
    issues = {
        "trailing_punct": [],
        "leading_punct": [],
        "blocklisted": [],
        "short_span": [],
        "invalid_indices": [],
        "high_density_examples": [],
    }

    label_counts = Counter()
    skill_counts = []

    for i, ex in enumerate(data):
        tokens = ex.get("tokenized_text", [])
        n_skills = 0

        for start, end, label in ex.get("ner", []):
            # Validate indices
            if start < 0 or end >= len(tokens) or start > end:
                issues["invalid_indices"].append((i, f"[{start}:{end}] for {len(tokens)} tokens", label))
                continue

            label_counts[label] += 1
            if label == "TECHNICAL_SKILL":
                n_skills += 1

            span = check_trailing_punct(tokens, start, end)
            if span:
                issues["trailing_punct"].append((i, span, label))

            span = check_leading_punct(tokens, start, end)
            if span:
                issues["leading_punct"].append((i, span, label))

            span = check_blocklisted(tokens, start, end)
            if span and label == "TECHNICAL_SKILL":
                issues["blocklisted"].append((i, span, label))

            span = check_short_span(tokens, start, end)
            if span:
                issues["short_span"].append((i, span, label))

        skill_counts.append(n_skills)
        if n_skills > MAX_SKILLS_PER_EXAMPLE:
            issues["high_density_examples"].append((i, n_skills))

    return {
        "issues": issues,
        "label_counts": label_counts,
        "skill_counts": skill_counts,
    }


def print_report(result: dict, total: int) -> None:
    issues = result["issues"]
    label_counts = result["label_counts"]
    skill_counts = result["skill_counts"]

    print(f"\n{'=' * 60}")
    print(f"Data Validation Report  ({total} examples)")
    print(f"{'=' * 60}")

    print(f"\n-- Label distribution --")
    for label, count in label_counts.most_common():
        print(f"  {label:25s}: {count:>7}")

    avg_skills = sum(skill_counts) / len(skill_counts) if skill_counts else 0
    max_skills = max(skill_counts) if skill_counts else 0
    print(f"\n-- Skill density --")
    print(f"  Avg skills/example    : {avg_skills:.1f}")
    print(f"  Max skills in example : {max_skills}")

    for key, items in issues.items():
        count = len(items)
        flag = "PASS" if count == 0 else "FAIL"
        print(f"\n-- {key} ({flag}: {count} issues) --")
        for entry in items[:5]:
            if key == "high_density_examples":
                print(f"  Example {entry[0]}: {entry[1]} skills")
            else:
                print(f"  Example {entry[0]}: '{entry[1]}' [{entry[2]}]")
        if count > 5:
            print(f"  ... and {count - 5} more")

    total_issues = sum(len(v) for v in issues.values())
    print(f"\n{'=' * 60}")
    print(f"Total issues found: {total_issues}")
    print(f"{'=' * 60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GLiNER training dataset")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to relabelled dataset JSON")
    parser.add_argument("--fix", action="store_true",
                        help="Also write a cleaned version (re-run relabelling)")
    parser.add_argument("--output", default=None,
                        help="Output path for --fix (default: overwrite input)")
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    with open(args.data) as f:
        data = json.load(f)

    result = audit(data)
    print_report(result, len(data))

    if args.fix:
        from src.relabel_dataset import relabel_dataset

        raw_path = "src/training_data/synthetic_gliner_dataset.json"
        out_path = args.output or args.data
        print(f"\n-- Re-running relabelling with updated filters --")
        print(f"  Input  : {raw_path}")
        print(f"  Output : {out_path}")
        relabel_dataset(raw_path, out_path)

        # Validate again
        print(f"\n-- Validating cleaned dataset --")
        with open(out_path) as f:
            cleaned = json.load(f)
        result2 = audit(cleaned)
        print_report(result2, len(cleaned))
