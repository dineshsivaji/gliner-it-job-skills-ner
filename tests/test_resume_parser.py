"""
Tests for resume_parser — chunking, skill normalisation, and utility functions.

These tests do NOT require the GLiNER model; they cover pure logic functions.
"""

import pytest
from src.resume_parser import chunk_text, normalise_skill, CHUNK_MAX, CHUNK_OVERLAP


# ── chunk_text ───────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "Python, Java, React"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_text(text, max_chars=20)
        assert len(chunks) == 2
        assert "First" in chunks[0]
        assert "Second" in chunks[1]

    def test_no_chunk_exceeds_max(self):
        text = "word " * 500  # ~2500 chars
        chunks = chunk_text(text, max_chars=200, overlap=50)
        for chunk in chunks:
            assert len(chunk) <= 200 + 10  # small buffer for word boundary snapping

    def test_overlap_produces_more_chunks(self):
        text = "word " * 500
        chunks_no_overlap = chunk_text(text, max_chars=200, overlap=0)
        chunks_with_overlap = chunk_text(text, max_chars=200, overlap=100)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_overlap_chunks_share_content(self):
        """With overlap, consecutive chunks should share some text."""
        # Build a long single block (no paragraph breaks)
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, max_chars=100, overlap=40)
        if len(chunks) >= 2:
            # The end of chunk 0 should overlap with the start of chunk 1
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            shared = words_0 & words_1
            assert len(shared) > 0, "Overlapping chunks should share some words"

    def test_default_params(self):
        """chunk_text uses module-level defaults."""
        text = "short"
        chunks = chunk_text(text)
        assert chunks == ["short"]


# ── normalise_skill ──────────────────────────────────────────────────────────

class TestNormaliseSkill:
    @pytest.mark.parametrize("input_text,expected", [
        ("React",      "react"),
        ("react",      "react"),
        ("React.js",   "react"),
        ("ReactJS",    "react"),
        ("react-js",   "react"),
        ("Express.js", "express"),
        ("Node.js",    "node"),
        ("Vue.js",     "vue"),
        ("Next.js",    "next"),
    ])
    def test_js_suffix_stripping(self, input_text, expected):
        assert normalise_skill(input_text) == expected

    def test_whitespace_collapse(self):
        result = normalise_skill("  React   Native  ")
        assert result == "react native"
        assert "  " not in result  # no double spaces

    def test_punctuation_stripping(self):
        assert normalise_skill("(Python)") == "python"
        assert normalise_skill("Python,") == "python"

    def test_preserves_non_js_names(self):
        assert normalise_skill("Python") == "python"
        assert normalise_skill("AWS") == "aws"
        assert normalise_skill("Docker") == "docker"
        assert normalise_skill("PostgreSQL") == "postgresql"

    def test_empty_string(self):
        result = normalise_skill("")
        assert result == ""

    def test_only_js(self):
        """Edge case: the entire string is 'js' — should not be emptied."""
        result = normalise_skill("js")
        # 'js' after removing suffix could be empty, but the function returns
        # the original if stripped becomes empty
        assert result == "js"

    @pytest.mark.parametrize("input_text,expected", [
        ("three.js", "three.js"),
        ("p5.js",    "p5.js"),
        ("d3.js",    "d3.js"),
        ("chart.js", "chart.js"),
        ("Three.js", "three.js"),
    ])
    def test_preserves_known_js_frameworks(self, input_text, expected):
        """Known .js frameworks should keep their suffix."""
        assert normalise_skill(input_text) == expected

    @pytest.mark.parametrize("input_text,expected", [
        ("three.js", "three.js"),
        ("p5.js",    "p5.js"),
        ("d3.js",    "d3.js"),
        ("chart.js", "chart.js"),
        ("Three.js", "three.js"),
    ])
    def test_preserves_known_js_frameworks(self, input_text, expected):
        """Known .js frameworks should keep their suffix."""
        assert normalise_skill(input_text) == expected
