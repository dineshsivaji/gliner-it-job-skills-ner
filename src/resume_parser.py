"""
resume_parser.py
----------------
2-stage resume parsing pipeline:
  Stage 1: GLiNER (fine-tuned medium) extracts spans with coarse labels
  Stage 2: TaxonomyMapper categorises TECHNICAL_SKILL spans into fine-grained categories

Usage:
    python resume_parser.py
    python resume_parser.py --text "your resume text here"
    python resume_parser.py --file resume.txt
"""

import argparse
import re
from gliner import GLiNER
from .taxonomy_mapper import TaxonomyMapper, ENTITY_BLOCKLIST, _JOB_TITLE_BLOCKLIST_PATTERNS

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID   = "dineshsivaji/gliner-it-job-skills-ner"
THRESHOLD  = 0.6
CHUNK_MAX  = 1200  # ~300 tokens, safely under model max 384
CHUNK_OVERLAP = 200  # ~50 tokens overlap to catch boundary-straddling entities

# Labels the model was fine-tuned on (high confidence)
TRAINED_LABELS = [
    "TECHNICAL_SKILL",
    "JOB_TITLE",
]

# Labels handled zero-shot by the base model (lower confidence, still useful)
ZEROSHOT_LABELS = [
    # "SOFT_SKILL",
    # "CERTIFICATION",
    # "EDUCATION",
    # "YEARS_OF_EXPERIENCE",
]

LABELS = TRAINED_LABELS + ZEROSHOT_LABELS

# Use a higher threshold for zero-shot labels to reduce false positives
ZEROSHOT_THRESHOLD = 0.75

# ── Normalisation ──────────────────────────────────────────────────────────

# Common suffix patterns that create duplicate skill entries
_SUFFIX_RE = re.compile(r"[.\-]?js$", re.IGNORECASE)
_STRIP_CHARS = " ,.;:!?\"'`()[]{}/-–—"

# Frameworks where ".js" is part of the canonical name — don't strip suffix
_JS_PRESERVE = frozenset({
    "three.js", "p5.js", "d3.js", "chart.js", "anime.js",
    "paper.js", "matter.js", "tone.js", "brain.js",
})


def normalise_skill(text: str) -> str:
    """
    Normalise a skill string for deduplication.

    Collapses variants like 'React.js', 'ReactJS', 'react' into the same key.
    Preserves known .js frameworks (three.js, p5.js, etc.) where the suffix
    is part of the canonical name.
    """
    text = text.lower().strip().strip(_STRIP_CHARS)
    text = re.sub(r"\s+", " ", text)
    # Don't strip suffix for known .js frameworks
    if text in _JS_PRESERVE:
        return text
    # Remove trailing .js / -js / js (but not the whole string)
    stripped = _SUFFIX_RE.sub("", text).strip()
    return stripped if stripped else text

# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = CHUNK_MAX, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks to stay under model max length.

    First splits on paragraph boundaries. Paragraphs that fit go through as-is.
    Paragraphs that exceed max_chars are split into overlapping windows at the
    nearest word boundary so entities straddling a split point are captured by
    at least one chunk.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Split into paragraphs first (natural semantic boundaries)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        # Further split long paragraphs on sentence/line boundaries
        sentences = [s.strip() for s in re.split(r"(?<=[.\n])\s+", para) if s.strip()]

        for sent in sentences:
            if len(sent) <= max_chars:
                chunks.append(sent)
            else:
                # Sliding window with overlap for very long blocks
                start = 0
                while start < len(sent):
                    end = min(start + max_chars, len(sent))
                    # Snap to word boundary
                    if end < len(sent):
                        last_space = sent.rfind(" ", start, end + 1)
                        if last_space > start:
                            end = last_space + 1
                    chunk = sent[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    # Step forward by (max_chars - overlap) instead of max_chars
                    step = max(end - start - overlap, 1)
                    start += step

    return [c for c in chunks if c]


# ── Parser ───────────────────────────────────────────────────────────────────

class ResumeParser:
    def __init__(self, model_id: str = MODEL_ID, threshold: float = THRESHOLD):
        print(f"Loading model: {model_id}")
        self.model     = GLiNER.from_pretrained(model_id)
        self.mapper    = TaxonomyMapper(fuzzy=False)
        self.threshold = threshold
        print("Ready.\n")

    def parse(self, text: str) -> dict:
        """
        Parse a resume text.

        TECHNICAL_SKILL and JOB_TITLE use fine-tuned threshold (0.6).
        SOFT_SKILL, CERTIFICATION, EDUCATION, YEARS_OF_EXPERIENCE use
        zero-shot threshold (0.75) to reduce false positives.

        Returns:
        {
            "TECHNICAL_SKILL": {
                "PYTHON_ECOSYSTEM":    ["Django", "FastAPI"],
                "CLOUD":               ["AWS", "Azure"],
                ...
            },
            "JOB_TITLE":           ["Software Engineer", "Tech Lead"],
            "SOFT_SKILL":          ["Leadership", "Communication"],   # zero-shot
            "CERTIFICATION":       ["AWS Solutions Architect"],        # zero-shot
            "EDUCATION":           ["M.S. Computer Science"],         # zero-shot
            "YEARS_OF_EXPERIENCE": ["5 years", "3+ years"],           # zero-shot
        }
        """
        chunks = chunk_text(text)
        raw_entities: list[dict] = []

        for chunk in chunks:
            # Run trained labels at normal threshold
            if TRAINED_LABELS:
                raw_entities.extend(
                    self.model.predict_entities(chunk, TRAINED_LABELS, threshold=self.threshold)
                )
            # Run zero-shot labels at higher threshold to reduce false positives
            if ZEROSHOT_LABELS:
                raw_entities.extend(
                    self.model.predict_entities(chunk, ZEROSHOT_LABELS, threshold=ZEROSHOT_THRESHOLD)
                )

        # Deduplicate by normalised (text, label), keep highest score + best display text
        seen: dict[tuple[str, str], dict] = {}
        for e in raw_entities:
            norm = normalise_skill(e["text"]) if e["label"] == "TECHNICAL_SKILL" else e["text"].strip().lower()
            key = (norm, e["label"])
            if key not in seen or e["score"] > seen[key]["score"]:
                seen[key] = {"text": e["text"].strip(), "label": e["label"], "score": e["score"]}

        deduped = list(seen.values())

        # Enrich TECHNICAL_SKILL with fine-grained category
        enriched = self.mapper.enrich(deduped)

        # Structure output
        result: dict = {label: [] for label in LABELS}

        for e in enriched:
            label = e["label"]
            text  = e["text"]

            if label == "TECHNICAL_SKILL":
                # Split long comma-separated lines (e.g. "Languages: C#, Java, Python, ...")
                if not isinstance(result["TECHNICAL_SKILL"], dict):
                    result["TECHNICAL_SKILL"] = {}

                # Remove common section prefixes like "Languages:" or "Skills:"
                cleaned = re.sub(r"^(languages?|skills?)\s*:\s*", "", text, flags=re.IGNORECASE)

                # Split on commas or bullet separators
                parts = re.split(r"[,\u2022]", cleaned)
                for part in parts:
                    skill = part.strip(" -–—").strip()
                    if not skill:
                        continue

                    # Re-check blocklist on each atomic skill after splitting
                    if skill.strip().lower() in ENTITY_BLOCKLIST:
                        continue

                    # Recompute category per atomic skill
                    cat = self.mapper.map(skill)
                    if cat == "UNCATEGORIZED":
                        continue
                    result["TECHNICAL_SKILL"].setdefault(cat, [])
                    if skill not in result["TECHNICAL_SKILL"][cat]:
                        result["TECHNICAL_SKILL"][cat].append(skill)
            else:
                if text not in result[label]:
                    result[label].append(text)

        return result

    def print_result(self, result: dict) -> None:
        print("═" * 60)
        print("RESUME PARSE RESULT")
        print("═" * 60)

        for label, value in result.items():
            if not value:
                continue

            zs_note = "  (zero-shot)" if label in ZEROSHOT_LABELS else ""

            if label == "TECHNICAL_SKILL" and isinstance(value, dict):
                print(f"\n TECHNICAL SKILLS")
                for cat, skills in sorted(value.items()):
                    print(f"   {cat:25s}: {', '.join(skills)}")
            else:
                items = value if isinstance(value, list) else [value]
                print(f"\n• {label}{zs_note}")
                for item in items:
                    print(f"   • {item}")

        print("\n" + "═" * 60)


# ── Demo text ─────────────────────────────────────────────────────────────────

DEMO_TEXT = """
Deep M. Mehta | Full Stack Software Developer
mail@deepmehta.co.in | github.com/deep-mm | Redmond, WA

EDUCATION
North Carolina State University — Master of Computer Science, GPA 4.0
K J Somaiya College of Engineering — B.Tech Computer Engineering, GPA 9.1

SKILLS
Languages: C#, Java, Python, Ruby, JavaScript, SQL
Web & Mobile: Angular, React, Android, React-Native, Ruby on Rails, Django
Cloud: Azure, AWS, Firebase
DevOps: Azure-DevOps, GitHub Actions, Terraform, Ansible, Kubernetes

WORK EXPERIENCE
Software Engineer, Microsoft, Redmond, WA — June 2024 - Present

Software Development Engineer Intern, Amazon — May 2023 - Aug 2023
• 3 months full stack development – React Native, Java, AWS
• Integrated AWS Textract OCR feature, reducing manual effort by 85%

Technology Consultant, Microsoft, Hyderabad — Jul 2019 – Jul 2022
• 3 years full stack: Angular, React, C#, Java, Azure
• Azure certifications: Solutions Architect, DevOps Engineer, Kubernetes (CKA)
• Awards: Best Consultant 2021, Spark Award 2021

PROJECTS
Generative AI GitHub Bot using LLMs to automate CI/CD workflows.
Azure password manager using Azure AD, KeyVault, VNet, APIM.
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume parser using GLiNER + taxonomy")
    parser.add_argument("--text", type=str, default=None, help="Resume text string")
    parser.add_argument("--file", type=str, default=None, help="Path to resume text file")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            resume_text = f.read()
    elif args.text:
        resume_text = args.text
    else:
        print("No input provided — running on demo resume.\n")
        resume_text = DEMO_TEXT

    resume_parser = ResumeParser()
    result = resume_parser.parse(resume_text)
    resume_parser.print_result(result)
