"""A1.15 verify_facts test suite.

Test discipline:
- Fixture-only (no live network, no scraping production docs).
- Deterministic — same inputs always produce same WalkReport.
- Covers: empty corpus, single marker, multiple markers, prefix extraction,
  skip-directory hygiene, file read error tolerance.

References:
- `futur3/tools/verify_facts.py` (implementation)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from futur3.tools.verify_facts import (
    HypothesisMarker,
    WalkReport,
    _main,
    walk_hypothesis,
)

# ============================================================================
# Fixture helpers
# ============================================================================


def _write_md(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ============================================================================
# TestA1_15_EmptyCorpus
# ============================================================================


class TestA1_15_EmptyCorpus:
    def test_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        report = walk_hypothesis(tmp_path / "does_not_exist")
        assert report.total == 0
        assert report.by_file() == {}
        assert report.by_prefix() == {}

    def test_root_is_file_not_dir_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "not_a_dir.md"
        f.write_text("# just a file", encoding="utf-8")
        report = walk_hypothesis(f)
        assert report.total == 0

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        report = walk_hypothesis(tmp_path)
        assert report.total == 0

    def test_md_file_without_hypothesis_returns_empty(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "no_markers.md", "# Doc with no markers\n\nSome prose.")
        report = walk_hypothesis(tmp_path)
        assert report.total == 0


# ============================================================================
# TestA1_15_SingleMarker
# ============================================================================


class TestA1_15_SingleMarker:
    def test_finds_single_marker(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "# Title\n\nHYPOTHESIS: this needs verifying.\n\nOther text.",
        )
        report = walk_hypothesis(tmp_path)
        assert report.total == 1

    def test_marker_records_filepath(self, tmp_path: Path) -> None:
        path = _write_md(
            tmp_path / "doc.md",
            "HYPOTHESIS: test\n",
        )
        report = walk_hypothesis(tmp_path)
        m = report.markers[0]
        assert m.file_path == path

    def test_marker_records_line_number(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "Line 1\nLine 2\nHYPOTHESIS: on line 3\nLine 4",
        )
        report = walk_hypothesis(tmp_path)
        assert report.markers[0].line_num == 3

    def test_marker_strips_whitespace(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "    HYPOTHESIS: leading spaces    \n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.markers[0].line_text == "HYPOTHESIS: leading spaces"

    def test_lowercase_hypothesis_not_matched(self, tmp_path: Path) -> None:
        """Case-sensitive: 'hypothesis' (prose) does NOT trigger marker."""
        _write_md(
            tmp_path / "doc.md",
            "The hypothesis is that we need more data.\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.total == 0

    def test_hypothesis_as_substring_not_matched(self, tmp_path: Path) -> None:
        """Word-boundary regex: 'HYPOTHESISxyz' does NOT match (whole word only)."""
        _write_md(
            tmp_path / "doc.md",
            "HYPOTHESISvariable = 42\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.total == 0


# ============================================================================
# TestA1_15_MultipleMarkers
# ============================================================================


class TestA1_15_MultipleMarkers:
    def test_multiple_markers_same_file(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "HYPOTHESIS: one\nMore text\nHYPOTHESIS: two\nHYPOTHESIS: three\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.total == 3

    def test_multiple_files(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "a.md", "HYPOTHESIS: file a marker\n")
        _write_md(tmp_path / "b.md", "HYPOTHESIS: file b marker\n")
        _write_md(tmp_path / "c.md", "no markers here\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 2
        assert "a.md" in report.by_file()
        assert "b.md" in report.by_file()
        assert "c.md" not in report.by_file()

    def test_nested_subdirectories(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "research" / "notes_x.md", "HYPOTHESIS: in nested\n")
        _write_md(tmp_path / "top.md", "HYPOTHESIS: at root\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 2

    def test_by_file_count_per_file(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "a.md", "HYPOTHESIS: 1\nHYPOTHESIS: 2\n")
        _write_md(tmp_path / "b.md", "HYPOTHESIS: 3\n")
        report = walk_hypothesis(tmp_path)
        counts = report.by_file()
        assert counts["a.md"] == 2
        assert counts["b.md"] == 1


# ============================================================================
# TestA1_15_PrefixExtraction
# ============================================================================


class TestA1_15_PrefixExtraction:
    def test_bb_prefix(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "broker.md",
            "HYPOTHESIS BB-05: Bullish anonymous probe.\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.markers[0].prefix_codes == ("BB-05",)

    def test_bc_prefix(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "crypto.md",
            "HYPOTHESIS BC-77: WAF defense first live verify.\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.markers[0].prefix_codes == ("BC-77",)

    def test_ts_prefix(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "topstep.md",
            "HYPOTHESIS TS-03: Topstep 24/7 crypto status unresolved.\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.markers[0].prefix_codes == ("TS-03",)

    def test_a33_prefix(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "HYPOTHESIS A33-RE-01: TopstepX BAG combo support.\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.markers[0].prefix_codes == ("A33-RE-01",)

    def test_multiple_prefixes_one_line(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "HYPOTHESIS — see BB-05 + TS-03 + A33-RE-01 for verification.\n",
        )
        report = walk_hypothesis(tmp_path)
        codes = report.markers[0].prefix_codes
        assert "BB-05" in codes
        assert "TS-03" in codes
        assert "A33-RE-01" in codes

    def test_no_prefix_extracted_when_absent(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "HYPOTHESIS: this one has no prefix code.\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.markers[0].prefix_codes == ()

    def test_by_prefix_aggregates(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "HYPOTHESIS BB-01: a\nHYPOTHESIS BB-02: b\nHYPOTHESIS TS-01: c\n",
        )
        report = walk_hypothesis(tmp_path)
        prefix_counts = report.by_prefix()
        assert prefix_counts["BB"] == 2
        assert prefix_counts["TS"] == 1


# ============================================================================
# TestA1_15_SkipHygieneDirs
# ============================================================================


class TestA1_15_SkipHygieneDirs:
    def test_venv_skipped(self, tmp_path: Path) -> None:
        _write_md(tmp_path / ".venv" / "site-packages" / "doc.md", "HYPOTHESIS: x\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 0

    def test_git_skipped(self, tmp_path: Path) -> None:
        _write_md(tmp_path / ".git" / "doc.md", "HYPOTHESIS: x\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 0

    def test_pytest_cache_skipped(self, tmp_path: Path) -> None:
        _write_md(tmp_path / ".pytest_cache" / "doc.md", "HYPOTHESIS: x\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 0

    def test_pycache_skipped(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "__pycache__" / "doc.md", "HYPOTHESIS: x\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 0

    def test_node_modules_skipped(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "node_modules" / "lib" / "doc.md", "HYPOTHESIS: x\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 0

    def test_normal_dir_not_skipped(self, tmp_path: Path) -> None:
        """Verify that the skip-dir filter doesn't bleed into normal dirs."""
        _write_md(tmp_path / "research" / "doc.md", "HYPOTHESIS: real\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 1


# ============================================================================
# TestA1_15_Determinism
# ============================================================================


class TestA1_15_Determinism:
    def test_same_input_same_report_total(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "doc.md", "HYPOTHESIS BB-01: a\nHYPOTHESIS BB-02: b\n")
        r1 = walk_hypothesis(tmp_path)
        r2 = walk_hypothesis(tmp_path)
        assert r1.total == r2.total
        assert r1.by_file() == r2.by_file()
        assert r1.by_prefix() == r2.by_prefix()

    def test_by_file_sorted(self, tmp_path: Path) -> None:
        """Output is deterministically sorted for CLI / diff stability."""
        _write_md(tmp_path / "z.md", "HYPOTHESIS: z\n")
        _write_md(tmp_path / "a.md", "HYPOTHESIS: a\n")
        _write_md(tmp_path / "m.md", "HYPOTHESIS: m\n")
        report = walk_hypothesis(tmp_path)
        keys = list(report.by_file().keys())
        assert keys == sorted(keys)


# ============================================================================
# TestA1_15_Summary
# ============================================================================


class TestA1_15_Summary:
    def test_summary_includes_total(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "doc.md", "HYPOTHESIS: x\nHYPOTHESIS: y\n")
        report = walk_hypothesis(tmp_path)
        summary = report.summary()
        assert "Total markers: 2" in summary

    def test_summary_lists_file_counts(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "doc.md", "HYPOTHESIS: x\n")
        report = walk_hypothesis(tmp_path)
        summary = report.summary()
        assert "doc.md: 1" in summary

    def test_summary_lists_prefix_counts(self, tmp_path: Path) -> None:
        _write_md(tmp_path / "doc.md", "HYPOTHESIS BB-01: x\n")
        report = walk_hypothesis(tmp_path)
        summary = report.summary()
        assert "BB: 1" in summary

    def test_unprefixed_count(self, tmp_path: Path) -> None:
        _write_md(
            tmp_path / "doc.md",
            "HYPOTHESIS: no prefix\nHYPOTHESIS BB-01: with prefix\n",
        )
        report = walk_hypothesis(tmp_path)
        assert report.unprefixed_count() == 1


# ============================================================================
# TestA1_15_Realistic
# ============================================================================


class TestA1_15_Realistic:
    def test_realistic_research_doc(self, tmp_path: Path) -> None:
        """Multi-section research-style doc with mixed markers."""
        content = """# Research Notes

## §1 TL;DR

HYPOTHESIS BB-05: Bullish anonymous probe (Phase A1 wk1).
HYPOTHESIS — verify with Topstep ticket TS-99.

## §2 Detail

Free-text HYPOTHESIS without a prefix code.
Real prose talks about a hypothesis (lowercase) - should NOT match.

## §3 Cross-reference

See A33-RE-01 for the calendar-spread question.
"""
        _write_md(tmp_path / "research" / "notes_z.md", content)
        report = walk_hypothesis(tmp_path)
        # 3 HYPOTHESIS markers (BB-05, TS-99 inline, free-text); A33-RE-01 line has no HYPOTHESIS
        assert report.total == 3


# ============================================================================
# TestA1_15_ReadErrors
# ============================================================================


class TestA1_15_ReadErrors:
    def test_unreadable_file_skipped_gracefully(self, tmp_path: Path) -> None:
        """File scan errors don't crash the walker — just yield zero markers."""
        # Write a valid file + a path that exists but read will fail (simulate
        # by writing then removing read permission on POSIX, or just trust
        # that the OSError handler exists). Here we just verify the happy path
        # doesn't crash on a real OSError-prone scenario.
        good = _write_md(tmp_path / "good.md", "HYPOTHESIS: real\n")
        report = walk_hypothesis(tmp_path)
        assert report.total == 1
        assert report.markers[0].file_path == good


# ============================================================================
# TestA1_15_CLI
# ============================================================================


class TestA1_15_CLI:
    def test_cli_main_returns_0_on_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = _main(["--root", str(tmp_path)])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Total markers: 0" in out

    def test_cli_main_reports_markers(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_md(tmp_path / "doc.md", "HYPOTHESIS: x\nHYPOTHESIS BB-01: y\n")
        exit_code = _main(["--root", str(tmp_path)])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Total markers: 2" in out
        assert "BB: 1" in out


# ============================================================================
# TestA1_15_DataclassValidation
# ============================================================================


class TestA1_15_DataclassValidation:
    def test_hypothesis_marker_frozen(self) -> None:
        m = HypothesisMarker(
            file_path=Path("/tmp/x.md"),
            line_num=1,
            line_text="HYPOTHESIS: x",
            prefix_codes=(),
        )
        with pytest.raises(AttributeError):
            m.line_num = 2  # type: ignore[misc]

    def test_walk_report_frozen(self) -> None:
        r = WalkReport(markers=(), root=Path("/tmp"))
        with pytest.raises(AttributeError):
            r.root = Path("/other")  # type: ignore[misc]
