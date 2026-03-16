"""
evaluate_model.py
-----------------
Evaluates a fine-tuned GLiNER model on the test split of the relabelled dataset.
Reports per-label and overall:
  - Precision
  - Recall
  - F1  (harmonic mean of precision + recall — balanced)
  - F2  (weights recall 2x over precision — useful when missing a skill
          is worse than a false positive)

NER evaluation uses EXACT SPAN MATCH:
  A predicted span is a True Positive only if both the text AND the label
  match a gold span exactly. Partial matches count as FP + FN.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model dineshsivaji/gliner-resume-ner-medium
    python evaluate_model.py --data /path/to/relabelled.json --threshold 0.5
"""

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

from gliner import GLiNER

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "dineshsivaji/gliner-it-job-skills-ner"
DEFAULT_DATA = "src/training_data/synthetic_gliner_relabelled.json"
DEFAULT_THRESHOLD = 0.5  # lower than inference (0.65) to get full precision/recall curve
LABELS = ["TECHNICAL_SKILL", "JOB_TITLE"]
TEST_SPLIT = 0.1  # must match train_gliner_resume.py split
RANDOM_SEED = 42
MAX_EVAL_SAMPLES = 500  # cap to keep evaluation fast; set None for full test set
SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_test_split(data_path: str) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)
    random.seed(RANDOM_SEED)
    random.shuffle(data)
    split = int(len(data) * (1 - TEST_SPLIT))
    test_data = data[split:]
    print(f"Loaded {len(test_data)} test examples from {data_path}")
    return test_data


def _normalize_span(text: str) -> str:
    """
    Normalize a span for fair comparison.
    Uses the same strip chars as relabel_dataset._span_text() to ensure
    training/evaluation consistency.
    """
    text = text.lower().strip()
    text = text.strip(" ,.;:!?\"'`()[]{}/-–—")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens_to_text(tokens: list[str], start: int, end: int) -> str:
    """Reconstruct span text from tokenized_text using start/end indices."""
    raw = " ".join(tokens[start: end + 1])
    return _normalize_span(raw)


def get_gold_spans(example: dict) -> set[tuple[str, str]]:
    """
    Returns a set of (span_text, label) tuples from the gold NER annotations.
    Only includes labels we're evaluating.
    """
    tokens = example["tokenized_text"]
    spans = set()
    for start, end, label in example["ner"]:
        if label in LABELS:
            text = tokens_to_text(tokens, start, end)
            if text:
                spans.add((text, label))
    return spans


def get_pred_spans(
        model: GLiNER,
        example: dict,
        threshold: float,
) -> set[tuple[str, str]]:
    """
    Runs the model on the reconstructed sentence and returns predicted spans.
    """
    tokens = example["tokenized_text"]
    sentence = " ".join(tokens)
    entities = model.predict_entities(sentence, LABELS, threshold=threshold)
    spans = set()
    for e in entities:
        normalized = _normalize_span(e["text"])
        if normalized:
            spans.add((normalized, e["label"]))
    return spans


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class LabelMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        return self._fbeta(beta=1)

    @property
    def f2(self) -> float:
        """
        F2 weights recall twice as much as precision.
        Use when missing a real skill (FN) is more costly than a false alarm (FP).
        Formula: (1 + 2²) * P * R / (2² * P + R) = 5PR / (4P + R)
        """
        return self._fbeta(beta=2)

    def _fbeta(self, beta: float) -> float:
        p, r = self.precision, self.recall
        b2 = beta ** 2
        return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0

    def support(self) -> int:
        return self.tp + self.fn  # total gold spans


def compute_overall(metrics: dict[str, LabelMetrics]) -> LabelMetrics:
    """Micro-average across all labels (pools TP/FP/FN)."""
    overall = LabelMetrics()
    for m in metrics.values():
        overall.tp += m.tp
        overall.fp += m.fp
        overall.fn += m.fn
    return overall


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
        model: GLiNER,
        test_data: list[dict],
        threshold: float,
        max_samples: int | None,
) -> dict[str, LabelMetrics]:
    if max_samples:
        test_data = test_data[:max_samples]

    metrics: dict[str, LabelMetrics] = defaultdict(LabelMetrics)
    total = len(test_data)

    for i, example in enumerate(test_data):
        if (i + 1) % 100 == 0:
            print(f"  Evaluating {i + 1}/{total}...")

        gold = get_gold_spans(example)
        preds = get_pred_spans(model, example, threshold)

        for span in preds:
            label = span[1]
            if span in gold:
                metrics[label].tp += 1
            else:
                metrics[label].fp += 1

        for span in gold:
            label = span[1]
            if span not in preds:
                metrics[label].fn += 1

    return dict(metrics)


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(metrics: dict[str, LabelMetrics], threshold: float) -> None:
    overall = compute_overall(metrics)

    col_w = 24
    print(f"\n{'═' * 70}")
    print(f"GLiNER Evaluation Report  (threshold={threshold})")
    print(f"{'═' * 70}")
    print(
        f"{'Label':{col_w}} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'F2':>8} {'Support':>8}"
    )
    print(f"{'─' * 70}")

    for label in sorted(metrics):
        m = metrics[label]
        print(
            f"{label:{col_w}} {m.precision:>10.4f} {m.recall:>8.4f} "
            f"{m.f1:>8.4f} {m.f2:>8.4f} {m.support():>8}"
        )

    print(f"{'─' * 70}")
    print(
        f"{'OVERALL (micro)':24} {overall.precision:>10.4f} {overall.recall:>8.4f} "
        f"{overall.f1:>8.4f} {overall.f2:>8.4f} {overall.support():>8}"
    )
    print(f"{'═' * 70}")
    print()
    print("Metric guide:")
    print("  Precision : of all predicted spans, how many are correct")
    print("  Recall    : of all gold spans, how many were found")
    print("  F1        : balanced harmonic mean of precision + recall")
    print("  F2        : recall-weighted — penalises missed skills more than false alarms")
    print(f"{'═' * 70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max", type=int, default=MAX_EVAL_SAMPLES,
                        help="Max test samples to evaluate (None = all)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run evaluation at multiple thresholds for P/R analysis")
    args = parser.parse_args()

    print(f"Loading model : {args.model}")
    model = GLiNER.from_pretrained(args.model)

    test_data = load_test_split(args.data)

    if args.sweep:
        print(f"\n{'═' * 80}")
        print(f"Threshold sweep on {min(args.max or len(test_data), len(test_data))} samples")
        print(f"{'═' * 80}")
        print(f"{'Threshold':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'F2':>8}")
        print(f"{'─' * 50}")
        for t in SWEEP_THRESHOLDS:
            m = evaluate(model, test_data, t, args.max)
            overall = compute_overall(m)
            print(f"{t:<12.2f} {overall.precision:>10.4f} {overall.recall:>8.4f} "
                  f"{overall.f1:>8.4f} {overall.f2:>8.4f}")
        print(f"{'═' * 80}\n")
    else:
        print(f"\nRunning evaluation on {min(args.max or len(test_data), len(test_data))} samples...")
        metrics = evaluate(model, test_data, args.threshold, args.max)
        print_report(metrics, args.threshold)


if __name__ == "__main__":
    main()