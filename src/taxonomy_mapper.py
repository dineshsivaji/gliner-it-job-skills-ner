"""
taxonomy_mapper.py
------------------
Stage 2 of the 2-stage resume parsing pipeline.

Stage 1: GLiNER extracts spans with coarse labels (TECHNICAL_SKILL etc.)
Stage 2: This module maps each extracted TECHNICAL_SKILL span to a
         fine-grained category using:
           1. Exact/case-insensitive lookup in a curated taxonomy dict
           2. Fuzzy fallback via rapidfuzz (optional)
           3. "UNCATEGORIZED" if no match found

Usage:
    from taxonomy_mapper import TaxonomyMapper

    mapper = TaxonomyMapper()
    category = mapper.map("FastAPI")        # → "PYTHON_ECOSYSTEM"
    category = mapper.map("React Native")  # → "MOBILE"
    category = mapper.map("AWS Lambda")    # → "CLOUD"

    # Batch — enrich GLiNER output directly
    entities = model.predict_entities(text, labels, threshold=0.65)
    enriched = mapper.enrich(entities)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

# ── Taxonomy ─────────────────────────────────────────────────────────────────
# Key   : lowercase skill name (exact or substring match)
# Value : fine-grained category

RAW_TAXONOMY: dict[str, str] = {

    # ── PROGRAMMING_LANGUAGE ─────────────────────────────────────────────────
    "python":           "PROGRAMMING_LANGUAGE",
    "java":             "PROGRAMMING_LANGUAGE",
    "c#":               "PROGRAMMING_LANGUAGE",
    "c++":              "PROGRAMMING_LANGUAGE",
    "go":               "PROGRAMMING_LANGUAGE",
    "golang":           "PROGRAMMING_LANGUAGE",
    "rust":             "PROGRAMMING_LANGUAGE",
    "ruby":             "PROGRAMMING_LANGUAGE",
    "php":              "PROGRAMMING_LANGUAGE",
    "swift":            "PROGRAMMING_LANGUAGE",
    "kotlin":           "PROGRAMMING_LANGUAGE",
    "scala":            "PROGRAMMING_LANGUAGE",
    "r":                "PROGRAMMING_LANGUAGE",
    "matlab":           "PROGRAMMING_LANGUAGE",
    "typescript":       "PROGRAMMING_LANGUAGE",
    "javascript":       "PROGRAMMING_LANGUAGE",
    "bash":             "PROGRAMMING_LANGUAGE",
    "shell":            "PROGRAMMING_LANGUAGE",
    "powershell":       "PROGRAMMING_LANGUAGE",
    "perl":             "PROGRAMMING_LANGUAGE",
    "dart":             "PROGRAMMING_LANGUAGE",
    "elixir":           "PROGRAMMING_LANGUAGE",
    "haskell":          "PROGRAMMING_LANGUAGE",
    "lua":              "PROGRAMMING_LANGUAGE",
    "groovy":           "PROGRAMMING_LANGUAGE",
    "c":                "PROGRAMMING_LANGUAGE",
    "objective-c":      "PROGRAMMING_LANGUAGE",
    "julia":            "PROGRAMMING_LANGUAGE",
    "zig":              "PROGRAMMING_LANGUAGE",

    # ── PYTHON_ECOSYSTEM ─────────────────────────────────────────────────────
    "django":           "PYTHON_ECOSYSTEM",
    "flask":            "PYTHON_ECOSYSTEM",
    "fastapi":          "PYTHON_ECOSYSTEM",
    "sqlalchemy":       "PYTHON_ECOSYSTEM",
    "celery":           "PYTHON_ECOSYSTEM",
    "pydantic":         "PYTHON_ECOSYSTEM",
    "numpy":            "PYTHON_ECOSYSTEM",
    "pandas":           "PYTHON_ECOSYSTEM",
    "scipy":            "PYTHON_ECOSYSTEM",
    "matplotlib":       "PYTHON_ECOSYSTEM",
    "seaborn":          "PYTHON_ECOSYSTEM",
    "pytest":           "PYTHON_ECOSYSTEM",
    "poetry":           "PYTHON_ECOSYSTEM",
    "uvicorn":          "PYTHON_ECOSYSTEM",
    "gunicorn":         "PYTHON_ECOSYSTEM",
    "alembic":          "PYTHON_ECOSYSTEM",
    "aiohttp":          "PYTHON_ECOSYSTEM",
    "asyncio":          "PYTHON_ECOSYSTEM",

    # ── JS_ECOSYSTEM ─────────────────────────────────────────────────────────
    "react":            "JS_ECOSYSTEM",
    "vue":              "JS_ECOSYSTEM",
    "angular":          "JS_ECOSYSTEM",
    "next.js":          "JS_ECOSYSTEM",
    "nextjs":           "JS_ECOSYSTEM",
    "nuxt":             "JS_ECOSYSTEM",
    "svelte":           "JS_ECOSYSTEM",
    "node.js":          "JS_ECOSYSTEM",
    "nodejs":           "JS_ECOSYSTEM",
    "express":          "JS_ECOSYSTEM",
    "nest.js":          "JS_ECOSYSTEM",
    "nestjs":           "JS_ECOSYSTEM",
    "webpack":          "JS_ECOSYSTEM",
    "vite":             "JS_ECOSYSTEM",
    "jest":             "JS_ECOSYSTEM",
    "graphql":          "JS_ECOSYSTEM",
    "redux":            "JS_ECOSYSTEM",
    "tailwind":         "JS_ECOSYSTEM",
    "tailwindcss":      "JS_ECOSYSTEM",
    "reactjs":          "JS_ECOSYSTEM",
    "angularjs":        "JS_ECOSYSTEM",
    "expressjs":        "JS_ECOSYSTEM",
    "jquery":           "JS_ECOSYSTEM",
    "socket.io":        "JS_ECOSYSTEM",
    "html":             "JS_ECOSYSTEM",
    "css":              "JS_ECOSYSTEM",
    "html/css":         "JS_ECOSYSTEM",

    # ── JAVA_ECOSYSTEM ───────────────────────────────────────────────────────
    "spring":           "JAVA_ECOSYSTEM",
    "spring boot":      "JAVA_ECOSYSTEM",
    "spring mvc":       "JAVA_ECOSYSTEM",
    "hibernate":        "JAVA_ECOSYSTEM",
    "maven":            "JAVA_ECOSYSTEM",
    "gradle":           "JAVA_ECOSYSTEM",
    "junit":            "JAVA_ECOSYSTEM",
    "mockito":          "JAVA_ECOSYSTEM",
    "quarkus":          "JAVA_ECOSYSTEM",
    "micronaut":        "JAVA_ECOSYSTEM",
    "struts":           "JAVA_ECOSYSTEM",
    "tomcat":           "JAVA_ECOSYSTEM",
    "jpa":              "JAVA_ECOSYSTEM",

    # ── AI_ML ────────────────────────────────────────────────────────────────
    "tensorflow":       "AI_ML",
    "pytorch":          "AI_ML",
    "keras":            "AI_ML",
    "scikit-learn":     "AI_ML",
    "sklearn":          "AI_ML",
    "xgboost":          "AI_ML",
    "lightgbm":         "AI_ML",
    "hugging face":     "AI_ML",
    "huggingface":      "AI_ML",
    "transformers":     "AI_ML",
    "langchain":        "AI_ML",
    "openai":           "AI_ML",
    "llm":              "AI_ML",
    "bert":             "AI_ML",
    "gpt":              "AI_ML",
    "rag":              "AI_ML",
    "mlflow":           "AI_ML",
    "wandb":            "AI_ML",
    "computer vision":  "AI_ML",
    "nlp":                        "AI_ML",
    "natural language processing": "AI_ML",
    "machine learning": "AI_ML",
    "deep learning":    "AI_ML",
    "neural network":   "AI_ML",
    "neural networks":  "AI_ML",
    "reinforcement learning": "AI_ML",
    "generative ai":    "AI_ML",
    "weights and biases": "AI_ML",
    "gliner":           "AI_ML",

    # ── CLOUD ────────────────────────────────────────────────────────────────
    "aws":              "CLOUD",
    "azure":            "CLOUD",
    "gcp":              "CLOUD",
    "google cloud":     "CLOUD",
    "s3":               "CLOUD",
    "ec2":              "CLOUD",
    "lambda":           "CLOUD",
    "cloudformation":   "CLOUD",
    "pulumi":           "CLOUD",
    "cloud functions":  "CLOUD",
    "app service":      "CLOUD",
    "azure functions":  "CLOUD",
    "cosmosdb":         "CLOUD",
    "azure ad":         "CLOUD",
    "keyvault":         "CLOUD",
    "firebase":         "CLOUD",
    "cloudflare":       "CLOUD",
    "vercel":           "CLOUD",
    "netlify":          "CLOUD",
    "heroku":           "CLOUD",
    "digitalocean":     "CLOUD",

    # ── DEVOPS ───────────────────────────────────────────────────────────────
    "docker":           "DEVOPS",
    "kubernetes":       "DEVOPS",
    "k8s":              "DEVOPS",
    "helm":             "DEVOPS",
    "terraform":        "DEVOPS",
    "jenkins":          "DEVOPS",
    "github actions":   "DEVOPS",
    "gitlab ci":        "DEVOPS",
    "azure devops":     "DEVOPS",
    "circleci":         "DEVOPS",
    "ansible":          "DEVOPS",
    "chef":             "DEVOPS",
    "puppet":           "DEVOPS",
    "prometheus":       "DEVOPS",
    "grafana":          "DEVOPS",
    "elk":              "DEVOPS",
    "elasticsearch":    "DEVOPS",
    "logstash":         "DEVOPS",
    "kibana":           "DEVOPS",
    "datadog":          "DEVOPS",
    "nginx":            "DEVOPS",
    "linux":            "DEVOPS",
    "unix":             "DEVOPS",
    "ci/cd":            "DEVOPS",
    "git":              "DEVOPS",
    "grunt":            "DEVOPS",

    # ── DATABASE ─────────────────────────────────────────────────────────────
    "postgresql":       "DATABASE",
    "postgres":         "DATABASE",
    "mysql":            "DATABASE",
    "sqlite":           "DATABASE",
    "mongodb":          "DATABASE",
    "redis":            "DATABASE",
    "cassandra":        "DATABASE",
    "dynamodb":         "DATABASE",
    "oracle":           "DATABASE",
    "sql server":       "DATABASE",
    "mssql":            "DATABASE",
    "neo4j":            "DATABASE",
    "influxdb":         "DATABASE",
    "snowflake":        "DATABASE",
    "bigquery":         "DATABASE",
    "redshift":         "DATABASE",
    "sql":              "DATABASE",
    "nosql":            "DATABASE",

    # ── MOBILE ───────────────────────────────────────────────────────────────
    "react native":     "MOBILE",
    "flutter":          "MOBILE",
    "android":          "MOBILE",
    "ios":              "MOBILE",
    "xcode":            "MOBILE",
    "expo":             "MOBILE",
    "exponentjs":       "MOBILE",

    # ── IT TOOLS ─────────────────────────────────────────────────────────────
    "jira":             "IT_TOOLS",
    "confluence":       "IT_TOOLS",
    "figma":            "IT_TOOLS",
    "postman":          "IT_TOOLS",
    "swagger":          "IT_TOOLS",
    "vs code":          "IT_TOOLS",
    "intellij":         "IT_TOOLS",
    "visual studio":    "IT_TOOLS",
    "github":           "IT_TOOLS",
    "gitlab":           "IT_TOOLS",
    "bitbucket":        "IT_TOOLS",
    "notion":           "IT_TOOLS",
    "slack":            "IT_TOOLS",
    "sonarqube":        "IT_TOOLS",
    "splunk":           "IT_TOOLS",
    "tableau":          "IT_TOOLS",
    "power bi":         "IT_TOOLS",
    "excel":            "IT_TOOLS",

    # ── BIG_DATA ─────────────────────────────────────────────────────────────
    "apache spark":     "BIG_DATA",
    "spark":            "BIG_DATA",
    "hadoop":           "BIG_DATA",
    "flink":            "BIG_DATA",
    "kafka":            "BIG_DATA",
    "airflow":          "BIG_DATA",
    "clickhouse":       "BIG_DATA",
    "hive":             "BIG_DATA",
    "presto":           "BIG_DATA",
    "dbt":              "BIG_DATA",
    "etl":              "BIG_DATA",
    "data pipeline":    "BIG_DATA",
    "data engineering": "BIG_DATA",
    "data warehouse":   "BIG_DATA",

    # ── DISTRIBUTED_SYSTEMS ──────────────────────────────────────────────────
    "distributed systems":    "DISTRIBUTED_SYSTEMS",
    "distributed computing":  "DISTRIBUTED_SYSTEMS",
    "microservices":          "DISTRIBUTED_SYSTEMS",
    "grpc":                   "DISTRIBUTED_SYSTEMS",
    "message queues":         "DISTRIBUTED_SYSTEMS",
    "rabbitmq":               "DISTRIBUTED_SYSTEMS",
    "service mesh":           "DISTRIBUTED_SYSTEMS",
    "istio":                  "DISTRIBUTED_SYSTEMS",
    "load balancing":         "DISTRIBUTED_SYSTEMS",
    "event driven":           "DISTRIBUTED_SYSTEMS",
    "rest":                   "DISTRIBUTED_SYSTEMS",
    "rest api":               "DISTRIBUTED_SYSTEMS",
    "consensus algorithms":   "DISTRIBUTED_SYSTEMS",
    "scalability":            "DISTRIBUTED_SYSTEMS",
    "fault tolerance":        "DISTRIBUTED_SYSTEMS",
}

# ── Entity blocklist ─────────────────────────────────────────────────────────
# False positives the model extracts with high confidence.
# Checked case-insensitively after stripping whitespace.

ENTITY_BLOCKLIST: set[str] = {
    # Generic / vague phrases (TECHNICAL_SKILL false positives)
    "application development",
    "web application design",
    "scripting and coding",
    "requirements gathering",
    "ui testing",
    "product testing and deployment",
    # UI components / project artifacts
    "file browser",
    "code editor",
    "chat window",
    # Academic fields / degree names
    "computer science",
    "management information systems",
    "information technology",
    "computer engineering",
    # Sentence fragments / noise
    "improve customer satisfaction",
    "integrated agent monitoring system",
    # Project names
    "picoshell",
}

# Patterns for JOB_TITLE false positives (e.g. "call center engineering team")
_JOB_TITLE_BLOCKLIST_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bteam$", re.IGNORECASE),
]

# ── Mapper class ──────────────────────────────────────────────────────────────

@dataclass
class TaxonomyMapper:
    """
    Maps a TECHNICAL_SKILL span text to a fine-grained category.

    Parameters
    ----------
    fuzzy : bool
        If True, fall back to rapidfuzz similarity when exact match fails.
        Requires `pip install rapidfuzz`.
    fuzzy_threshold : float
        Minimum similarity score (0–100) to accept a fuzzy match.
    extra_taxonomy : dict
        Additional {skill: category} entries to merge into the built-in taxonomy.
    """
    fuzzy: bool = False
    fuzzy_threshold: float = 80.0
    extra_taxonomy: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Build normalised lookup
        self._taxonomy: dict[str, str] = {
            k.lower(): v for k, v in {**RAW_TAXONOMY, **self.extra_taxonomy}.items()
        }
        self._keys = list(self._taxonomy.keys())

        # Pre-compile word-boundary patterns for all multi-char keys
        # e.g. "java" → re.compile(r'\bjava\b') so it won't match "somaiya"
        self._patterns: dict[str, re.Pattern] = {
            k: re.compile(r'\b' + re.escape(k) + r'\b')
            for k in self._keys
            if len(k) > 2
        }

        if self.fuzzy:
            try:
                from rapidfuzz import process as rf_process
                self._rf_process = rf_process
            except ImportError:
                raise ImportError(
                    "rapidfuzz is required for fuzzy matching: pip install rapidfuzz"
                )

    # ── public API ────────────────────────────────────────────────────────────

    def map(self, skill_text: str) -> str:
        """
        Returns the fine-grained category for a skill string.
        Falls back to 'UNCATEGORIZED' if nothing matches.
        """
        normalised = self._normalise(skill_text)

        # 1. Exact match
        if normalised in self._taxonomy:
            return self._taxonomy[normalised]

        # 2. Whole-word boundary match
        #    Wraps each key in \b anchors so "r" won't match inside "master",
        #    "java" won't match inside "somaiya", "go" won't match inside "golang".
        #    Short keys (≤2 chars) require exact match only — no substring at all.
        candidates = []
        for k in self._keys:
            if len(k) <= 2:
                # Single/double char keys (r, go, c#) — exact match only
                if normalised == k:
                    candidates.append(k)
            else:
                # Use word-boundary regex for multi-char keys
                pattern = self._patterns.get(k)
                if pattern and pattern.search(normalised):
                    candidates.append(k)

        if candidates:
            best_key = max(candidates, key=len)
            return self._taxonomy[best_key]

        # 3. Fuzzy match (optional)
        if self.fuzzy:
            result = self._rf_process.extractOne(
                normalised,
                self._keys,
                score_cutoff=self.fuzzy_threshold,
            )
            if result:
                return self._taxonomy[result[0]]

        return "UNCATEGORIZED"

    def filter_entities(self, entities: list[dict]) -> list[dict]:
        """
        Remove false-positive entities using ENTITY_BLOCKLIST and
        label-specific pattern rules.
        """
        filtered = []
        for entity in entities:
            text = entity.get("text", "").strip().lower()
            label = entity.get("label", "")

            # Check the global blocklist
            if text in ENTITY_BLOCKLIST:
                continue

            # Check JOB_TITLE-specific patterns
            if label == "JOB_TITLE":
                if any(p.search(text) for p in _JOB_TITLE_BLOCKLIST_PATTERNS):
                    continue

            filtered.append(entity)
        return filtered

    def enrich(self, entities: list[dict]) -> list[dict]:
        """
        Enrich a list of GLiNER entity dicts by adding a 'category' key
        to every TECHNICAL_SKILL entity.

        Input format (from model.predict_entities):
            [{"text": "React", "label": "TECHNICAL_SKILL", "score": 0.91}, ...]

        Output format:
            [{"text": "React", "label": "TECHNICAL_SKILL",
              "category": "JS_ECOSYSTEM", "score": 0.91}, ...]
        """
        enriched = []
        for entity in self.filter_entities(entities):
            e = dict(entity)
            if e.get("label") == "TECHNICAL_SKILL":
                e["category"] = self.map(e["text"])
            enriched.append(e)
        return enriched

    def summary(self, entities: list[dict]) -> dict[str, list[str]]:
        """
        Returns a dict grouping skill texts by their fine-grained category.

        Example output:
            {
              "PYTHON_ECOSYSTEM": ["Django", "FastAPI"],
              "CLOUD": ["AWS", "Azure"],
              "UNCATEGORIZED": ["some-obscure-tool"],
            }
        """
        enriched = self.enrich(entities)
        result: dict[str, list[str]] = {}
        for e in enriched:
            label = e.get("category", e["label"])
            result.setdefault(label, [])
            text = e["text"]
            if text not in result[label]:
                result[label].append(text)
        return dict(sorted(result.items()))

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, collapse whitespace, strip punctuation edges."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text


# ── Demo / smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mapper = TaxonomyMapper(fuzzy=False)

    test_cases = [
        ("Python",          "PROGRAMMING_LANGUAGE"),
        ("FastAPI",         "PYTHON_ECOSYSTEM"),
        ("React Native",    "MOBILE"),
        ("AWS Lambda",      "CLOUD"),
        ("Kubernetes",      "DEVOPS"),
        ("PostgreSQL",      "DATABASE"),
        ("PyTorch",         "AI_ML"),
        ("React",           "JS_ECOSYSTEM"),
        ("Spring Boot",     "JAVA_ECOSYSTEM"),
        ("Jira",            "IT_TOOLS"),
        ("Leadership",      None),   # SOFT_SKILL — not mapped by taxonomy
        ("UnknownTool",     None),   # → UNCATEGORIZED
        # ── New alias entries ──
        ("ReactJS",         "JS_ECOSYSTEM"),
        ("AngularJS",       "JS_ECOSYSTEM"),
        ("ExpressJS",       "JS_ECOSYSTEM"),
        ("jQuery",          "JS_ECOSYSTEM"),
        ("Socket.IO",       "JS_ECOSYSTEM"),
        ("HTML",            "JS_ECOSYSTEM"),
        ("CSS",             "JS_ECOSYSTEM"),
        ("HTML/CSS",        "JS_ECOSYSTEM"),
        ("ExponentJS",      "MOBILE"),
        ("Grunt",           "DEVOPS"),
        ("Tomcat",          "JAVA_ECOSYSTEM"),
        ("REST",            "DISTRIBUTED_SYSTEMS"),
        ("REST API",        "DISTRIBUTED_SYSTEMS"),
    ]

    print("── Taxonomy Mapper Smoke Test ──\n")
    all_pass = True
    for skill, expected in test_cases:
        result = mapper.map(skill)
        expected_str = expected or "UNCATEGORIZED"
        status = "✅" if result == expected_str else "❌"
        if result != expected_str:
            all_pass = False
        print(f"  {status}  {skill:20s}  →  {result}  (expected: {expected_str})")

    print(f"\n{'All tests passed ✅' if all_pass else 'Some tests failed ❌'}")

    # Demo: enrich mock GLiNER output
    print("\n── Enrich demo ──\n")
    mock_entities = [
        {"text": "Python",     "label": "TECHNICAL_SKILL", "score": 0.95},
        {"text": "AWS",        "label": "TECHNICAL_SKILL", "score": 0.88},
        {"text": "React",      "label": "TECHNICAL_SKILL", "score": 0.91},
        {"text": "Docker",     "label": "TECHNICAL_SKILL", "score": 0.87},
        {"text": "PostgreSQL", "label": "TECHNICAL_SKILL", "score": 0.84},
        {"text": "Leadership", "label": "SOFT_SKILL",      "score": 0.79},
        {"text": "Senior Software Engineer", "label": "JOB_TITLE", "score": 0.92},
    ]

    enriched = mapper.enrich(mock_entities)
    for e in enriched:
        cat = f"  [{e['category']}]" if "category" in e else ""
        print(f"  {e['label']:20s}{cat:25s}  {e['text']}")

    print("\n── Summary grouped by category ──\n")
    summary = mapper.summary(mock_entities)
    for cat, skills in summary.items():
        print(f"  {cat:25s}: {', '.join(skills)}")

    # ── Blocklist filter test ──
    print("\n── Blocklist Filter Test ──\n")
    blocklist_entities = [
        {"text": "Python",                     "label": "TECHNICAL_SKILL", "score": 0.95},
        {"text": "file browser",               "label": "TECHNICAL_SKILL", "score": 0.80},
        {"text": "computer science",           "label": "TECHNICAL_SKILL", "score": 0.85},
        {"text": "application development",    "label": "TECHNICAL_SKILL", "score": 0.78},
        {"text": "call center engineering team","label": "JOB_TITLE",      "score": 0.70},
        {"text": "Senior Engineer",            "label": "JOB_TITLE",      "score": 0.92},
        {"text": "picoshell",                  "label": "TECHNICAL_SKILL", "score": 0.72},
        {"text": "React",                      "label": "TECHNICAL_SKILL", "score": 0.91},
    ]
    filtered = mapper.filter_entities(blocklist_entities)
    kept_texts = {e["text"] for e in filtered}
    expected_kept = {"Python", "Senior Engineer", "React"}
    expected_removed = {"file browser", "computer science", "application development",
                        "call center engineering team", "picoshell"}

    bl_pass = True
    for e in blocklist_entities:
        txt = e["text"]
        if txt in expected_kept:
            if txt in kept_texts:
                print(f"  ✅  KEPT      {txt}")
            else:
                print(f"  ❌  WRONGLY REMOVED  {txt}")
                bl_pass = False
        elif txt in expected_removed:
            if txt not in kept_texts:
                print(f"  ✅  BLOCKED   {txt}")
            else:
                print(f"  ❌  WRONGLY KEPT     {txt}")
                bl_pass = False

    all_pass = all_pass and bl_pass
    print(f"\n{'All tests passed ✅' if all_pass else 'Some tests failed ❌'}")