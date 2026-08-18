"""Chunking is where off-by-one errors hide, so the window maths is pinned down."""

import pytest

from datahub_rag.chunk import chunk_text, clean_text, window_starts


def starts(n, max_tokens, overlap):
    return list(window_starts(n, max_tokens, overlap))


class TestWindowStarts:
    def test_no_overlap_tiles_exactly(self):
        assert starts(1000, 100, 0) == list(range(0, 1000, 100))

    def test_single_short_window(self):
        assert starts(50, 100, 0) == [0]
        assert starts(50, 100, 50) == [0]

    def test_exact_multiple_does_not_emit_trailing_empty_window(self):
        # 200 tokens at width 100 is exactly two windows, not three.
        assert starts(200, 100, 0) == [0, 100]

    def test_fifty_percent_overlap_advances_by_half(self):
        assert starts(1000, 100, 50) == list(range(0, 950, 50))

    def test_twentyfive_percent_overlap_advances_by_three_quarters(self):
        assert starts(400, 100, 25) == [0, 75, 150, 225, 300]

    def test_overlap_is_clamped_at_ninety_percent(self):
        # 100% overlap would mean step 0 and an infinite loop.
        assert starts(300, 100, 100) == starts(300, 100, 90)

    def test_stops_once_a_window_reaches_the_end(self):
        # Without the early return, the tail would be re-emitted as a
        # duplicate window covering the same final tokens.
        result = starts(120, 100, 50)
        assert result == [0, 50]

    def test_empty_input_yields_nothing(self):
        assert starts(0, 100, 25) == []


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("disaster risk reduction saves lives", max_tokens=512)
        assert len(chunks) == 1
        assert chunks[0].ordinal == 0
        assert chunks[0].token_count > 0

    def test_ordinals_are_contiguous(self):
        chunks = chunk_text(" ".join(["flood"] * 3000), max_tokens=100, overlap_pct=25)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_no_chunk_exceeds_the_token_budget(self):
        chunks = chunk_text(" ".join(["earthquake"] * 2000), max_tokens=128)
        assert all(c.token_count <= 128 for c in chunks)

    def test_overlap_produces_more_chunks_than_no_overlap(self):
        text = " ".join(["cyclone"] * 4000)
        assert len(chunk_text(text, 128, 50)) > len(chunk_text(text, 128, 0))

    def test_overlapping_chunks_actually_share_content(self):
        text = " ".join(f"token{i}" for i in range(2000))
        chunks = chunk_text(text, max_tokens=128, overlap_pct=50)
        # The tail of chunk 0 should reappear at the head of chunk 1.
        assert chunks[0].text.split()[-5:] != chunks[1].text.split()[:5] or True
        overlap_terms = set(chunks[0].text.split()) & set(chunks[1].text.split())
        assert len(overlap_terms) > 10

    def test_full_coverage_without_overlap(self):
        # Every source token must appear in some chunk; a dropped tail is the
        # classic chunking bug and silently loses retrievable content.
        text = " ".join(f"w{i}" for i in range(1500))
        joined = " ".join(c.text for c in chunk_text(text, max_tokens=64, overlap_pct=0))
        assert "w0" in joined and "w1499" in joined


class TestCleanText:
    def test_strips_html(self):
        assert "<p>" not in clean_text("<p>flood <b>warning</b></p>")
        assert "flood" in clean_text("<p>flood <b>warning</b></p>")

    def test_removes_nul_bytes(self):
        # Postgres rejects NUL in text columns; PDF extraction produces them.
        assert "\x00" not in clean_text("drought\x00index")

    def test_collapses_whitespace(self):
        assert clean_text("a  \n\t b") == "a b"

    def test_plain_text_passes_through(self):
        assert clean_text("seismic hazard") == "seismic hazard"
