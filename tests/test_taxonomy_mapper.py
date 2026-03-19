"""
Tests for TaxonomyMapper — taxonomy mapping, blocklist filtering, normalisation.
"""

import pytest
from src.taxonomy_mapper import TaxonomyMapper, ENTITY_BLOCKLIST, RAW_TAXONOMY


@pytest.fixture
def mapper():
    return TaxonomyMapper(fuzzy=False)


# ── Exact match ──────────────────────────────────────────────────────────────

class TestExactMatch:
    @pytest.mark.parametrize("skill,expected", [
        ("Python",       "PROGRAMMING_LANGUAGE"),
        ("python",       "PROGRAMMING_LANGUAGE"),
        ("PYTHON",       "PROGRAMMING_LANGUAGE"),
        ("FastAPI",      "PYTHON_ECOSYSTEM"),
        ("django",       "PYTHON_ECOSYSTEM"),
        ("react",        "JS_ECOSYSTEM"),
        ("React",        "JS_ECOSYSTEM"),
        ("spring boot",  "JAVA_ECOSYSTEM"),
        ("Spring Boot",  "JAVA_ECOSYSTEM"),
        ("kubernetes",   "DEVOPS"),
        ("PostgreSQL",   "DATABASE"),
        ("mongodb",      "DATABASE"),
        ("aws",          "CLOUD"),
        ("jira",         "IT_TOOLS"),
        ("apache spark", "BIG_DATA"),
        ("microservices","DISTRIBUTED_SYSTEMS"),
    ])
    def test_exact_match(self, mapper, skill, expected):
        assert mapper.map(skill) == expected

    def test_uncategorized(self, mapper):
        assert mapper.map("UnknownTool9999") == "UNCATEGORIZED"

    def test_empty_string(self, mapper):
        assert mapper.map("") == "UNCATEGORIZED"


# ── Alias entries ────────────────────────────────────────────────────────────

class TestAliases:
    @pytest.mark.parametrize("skill,expected", [
        ("ReactJS",    "JS_ECOSYSTEM"),
        ("AngularJS",  "JS_ECOSYSTEM"),
        ("ExpressJS",  "JS_ECOSYSTEM"),
        ("jQuery",     "JS_ECOSYSTEM"),
        ("Socket.IO",  "JS_ECOSYSTEM"),
        ("HTML",       "JS_ECOSYSTEM"),
        ("CSS",        "JS_ECOSYSTEM"),
        ("HTML/CSS",   "JS_ECOSYSTEM"),
        ("ExponentJS", "MOBILE"),
        ("Grunt",      "DEVOPS"),
        ("Tomcat",     "JAVA_ECOSYSTEM"),
        ("REST",       "DISTRIBUTED_SYSTEMS"),
        ("REST API",   "DISTRIBUTED_SYSTEMS"),
    ])
    def test_alias_maps_correctly(self, mapper, skill, expected):
        assert mapper.map(skill) == expected


# ── Word boundary matching ───────────────────────────────────────────────────

class TestWordBoundary:
    def test_java_not_in_somaiya(self, mapper):
        """'java' should NOT match inside 'somaiya'."""
        assert mapper.map("somaiya") == "UNCATEGORIZED"

    def test_r_not_in_master(self, mapper):
        """'r' should NOT match inside 'master'."""
        assert mapper.map("master") == "UNCATEGORIZED"

    def test_go_not_in_golang(self, mapper):
        """'go' should not match inside 'golang' — 'golang' has its own entry."""
        result = mapper.map("golang")
        assert result == "PROGRAMMING_LANGUAGE"

    def test_aws_lambda_matches_cloud(self, mapper):
        """Multi-word input 'AWS Lambda' should match 'lambda' → CLOUD."""
        assert mapper.map("AWS Lambda") == "CLOUD"

    def test_react_native_matches_mobile(self, mapper):
        """'React Native' should match the longest key 'react native' → MOBILE."""
        assert mapper.map("React Native") == "MOBILE"


# ── Blocklist filtering ──────────────────────────────────────────────────────

class TestBlocklistFiltering:
    @pytest.mark.parametrize("text", list(ENTITY_BLOCKLIST))
    def test_blocklisted_skill_is_removed(self, mapper, text):
        entities = [{"text": text, "label": "TECHNICAL_SKILL", "score": 0.85}]
        filtered = mapper.filter_entities(entities)
        assert len(filtered) == 0, f"'{text}' should be blocklisted"

    def test_legitimate_skill_not_blocked(self, mapper):
        entities = [
            {"text": "Python", "label": "TECHNICAL_SKILL", "score": 0.95},
            {"text": "React", "label": "TECHNICAL_SKILL", "score": 0.91},
        ]
        filtered = mapper.filter_entities(entities)
        assert len(filtered) == 2

    def test_job_title_team_pattern(self, mapper):
        entities = [
            {"text": "call center engineering team", "label": "JOB_TITLE", "score": 0.70},
        ]
        filtered = mapper.filter_entities(entities)
        assert len(filtered) == 0

    def test_valid_job_title_not_blocked(self, mapper):
        entities = [
            {"text": "Senior Software Engineer", "label": "JOB_TITLE", "score": 0.92},
        ]
        filtered = mapper.filter_entities(entities)
        assert len(filtered) == 1

    def test_blocklist_case_insensitive(self, mapper):
        entities = [
            {"text": "Computer Science", "label": "TECHNICAL_SKILL", "score": 0.80},
            {"text": "FILE BROWSER", "label": "TECHNICAL_SKILL", "score": 0.75},
        ]
        filtered = mapper.filter_entities(entities)
        assert len(filtered) == 0


# ── Enrich ───────────────────────────────────────────────────────────────────

class TestEnrich:
    def test_adds_category_to_technical_skill(self, mapper):
        entities = [
            {"text": "Python", "label": "TECHNICAL_SKILL", "score": 0.95},
        ]
        enriched = mapper.enrich(entities)
        assert len(enriched) == 1
        assert enriched[0]["category"] == "PROGRAMMING_LANGUAGE"

    def test_does_not_add_category_to_job_title(self, mapper):
        entities = [
            {"text": "Software Engineer", "label": "JOB_TITLE", "score": 0.90},
        ]
        enriched = mapper.enrich(entities)
        assert len(enriched) == 1
        assert "category" not in enriched[0]

    def test_enrich_filters_blocklisted(self, mapper):
        entities = [
            {"text": "Python", "label": "TECHNICAL_SKILL", "score": 0.95},
            {"text": "computer science", "label": "TECHNICAL_SKILL", "score": 0.80},
        ]
        enriched = mapper.enrich(entities)
        assert len(enriched) == 1
        assert enriched[0]["text"] == "Python"


# ── Summary ──────────────────────────────────────────────────────────────────

class TestSummary:
    def test_groups_by_category(self, mapper):
        entities = [
            {"text": "Python", "label": "TECHNICAL_SKILL", "score": 0.95},
            {"text": "Django", "label": "TECHNICAL_SKILL", "score": 0.90},
            {"text": "AWS", "label": "TECHNICAL_SKILL", "score": 0.88},
        ]
        summary = mapper.summary(entities)
        assert "PROGRAMMING_LANGUAGE" in summary
        assert "PYTHON_ECOSYSTEM" in summary
        assert "CLOUD" in summary
        assert "Python" in summary["PROGRAMMING_LANGUAGE"]
        assert "Django" in summary["PYTHON_ECOSYSTEM"]

    def test_deduplicates_skills(self, mapper):
        entities = [
            {"text": "Python", "label": "TECHNICAL_SKILL", "score": 0.95},
            {"text": "Python", "label": "TECHNICAL_SKILL", "score": 0.90},
        ]
        summary = mapper.summary(entities)
        assert summary["PROGRAMMING_LANGUAGE"].count("Python") == 1


# ── Extra taxonomy ───────────────────────────────────────────────────────────

class TestExtraTaxonomy:
    def test_extra_entries_override(self):
        mapper = TaxonomyMapper(extra_taxonomy={"python": "CUSTOM_CATEGORY"})
        assert mapper.map("Python") == "CUSTOM_CATEGORY"

    def test_extra_entries_add(self):
        mapper = TaxonomyMapper(extra_taxonomy={"myframework": "CUSTOM_CATEGORY"})
        assert mapper.map("myframework") == "CUSTOM_CATEGORY"


# ── Taxonomy completeness sanity check ───────────────────────────────────────

class TestTaxonomySanity:
    def test_all_values_are_known_categories(self):
        known_categories = {
            "PROGRAMMING_LANGUAGE", "PYTHON_ECOSYSTEM", "JS_ECOSYSTEM",
            "JAVA_ECOSYSTEM", "AI_ML", "CLOUD", "DEVOPS", "DATABASE",
            "MOBILE", "IT_TOOLS", "BIG_DATA", "DISTRIBUTED_SYSTEMS",
        }
        for skill, cat in RAW_TAXONOMY.items():
            assert cat in known_categories, f"'{skill}' maps to unknown category '{cat}'"

    def test_no_duplicate_keys(self):
        """RAW_TAXONOMY is a dict so duplicates are impossible at runtime,
        but this checks the key count matches expectations."""
        assert len(RAW_TAXONOMY) > 200
