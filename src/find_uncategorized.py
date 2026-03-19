"""
find_uncategorized.py
---------------------
Scans resume text files or a directory of resumes, runs the GLiNER pipeline,
and reports all UNCATEGORIZED skills sorted by frequency.

Use this to discover taxonomy gaps — skills the model extracts correctly but
the taxonomy mapper cannot categorise.

Usage:
    python -m src.find_uncategorized --file resume.txt
    python -m src.find_uncategorized --dir resumes/
    python -m src.find_uncategorized --text "Experienced with FastAPI, Tomcat, and RxJava"
"""

import argparse
import os
from collections import Counter
from pathlib import Path

from .resume_parser import ResumeParser


def collect_texts(args) -> list[str]:
    """Gather resume texts from CLI arguments."""
    texts: list[str] = []
    if args.text:
        texts.append(args.text)
    if args.file:
        texts.append(Path(args.file).read_text(encoding="utf-8", errors="ignore"))
    if args.dir:
        for root, _, files in os.walk(args.dir):
            for fname in files:
                if fname.endswith((".txt", ".md")):
                    fpath = os.path.join(root, fname)
                    texts.append(Path(fpath).read_text(encoding="utf-8", errors="ignore"))
    return texts


def main():
    ap = argparse.ArgumentParser(description="Find UNCATEGORIZED skills across resumes")
    ap.add_argument("--text", type=str, help="Single resume text string")
    ap.add_argument("--file", type=str, help="Path to a single resume text file")
    ap.add_argument("--dir", type=str, help="Directory of resume .txt/.md files")
    ap.add_argument("--top", type=int, default=50, help="Show top N uncategorized skills (default 50)")
    args = ap.parse_args()

    texts = collect_texts(args)
    if not texts:
        print("No input provided. Use --text, --file, or --dir.")
        return

    parser = ResumeParser()
    uncategorized: Counter = Counter()
    total_skills = 0

    for i, text in enumerate(texts, 1):
        result = parser.parse(text)
        tech = result.get("TECHNICAL_SKILL", {})
        if isinstance(tech, dict):
            for cat, skills in tech.items():
                total_skills += len(skills)
                if cat == "UNCATEGORIZED":
                    for skill in skills:
                        uncategorized[skill.lower()] += 1

    # Report
    print(f"\n{'=' * 60}")
    print(f"Taxonomy Gap Report")
    print(f"{'=' * 60}")
    print(f"  Resumes scanned        : {len(texts)}")
    print(f"  Total skills extracted  : {total_skills}")
    print(f"  Unique UNCATEGORIZED    : {len(uncategorized)}")
    print(f"  Total UNCATEGORIZED hits: {sum(uncategorized.values())}")

    if uncategorized:
        print(f"\n  Top {args.top} UNCATEGORIZED skills (by frequency):\n")
        print(f"  {'Skill':<40s} {'Count':>6}")
        print(f"  {'─' * 46}")
        for skill, count in uncategorized.most_common(args.top):
            print(f"  {skill:<40s} {count:>6}")
    else:
        print("\n  No UNCATEGORIZED skills found — taxonomy coverage is complete!")

    print(f"\n{'=' * 60}")
    print("Add frequent entries to RAW_TAXONOMY in src/taxonomy_mapper.py")


if __name__ == "__main__":
    main()
