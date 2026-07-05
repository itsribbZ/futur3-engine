"""verify_facts — HYPOTHESIS walker (Phase A1.15).

Convention: every load-bearing technical claim
in research deliverables gets a `HYPOTHESIS` marker so it can be verified
externally (via support tickets, paper smoke, empirical observation) before
promotion to RESOLVED. This walker enumerates markers across the project
documentation corpus so the operator has a single command to see "what still needs
verifying".

## Scope (v1 STUB)

- ENUMERATE: scan all `*.md` files under the futur3 root for occurrences of
  `HYPOTHESIS` (case-sensitive whole-word; avoids matching the word inside
  variable names or unrelated prose).
- CATEGORIZE: extract recognized prefix codes (BB-XX broker, BC-XX crypto,
  BA-XX, TS-XX Topstep, H-XX, VS-XX, A33-XX, BD-XX) when present in
  the HYPOTHESIS context line.
- REPORT: aggregate counts by file + by prefix-root + total + return a
  structured `WalkReport` for both CLI display and programmatic consumption.

v1 does NOT cross-reference RESOLVED markers — status promotion is manual.
v2 will add a status-file (e.g., `verified_facts.jsonl`) that the walker
reconciles against the markers in the corpus.

## Contracts

- **Read-only** — never modifies the markdown corpus.
- **Deterministic** — same inputs (same .md content) -> same WalkReport.
- **Skip hygiene dirs** — `.venv`, `.git`, `.pytest_cache`, `__pycache__`,
  `node_modules`, `.mypy_cache`, `.ruff_cache` are NOT walked.
- **Truthful reporting** — `WalkReport.summary()` reports counts as
  measured; never sugar-coats "we have X HYPOTHESIS items remaining".

## CLI

    python -m futur3.tools.verify_facts [--root <path>]

Defaults `--root` to the current working directory.

Read-only utility; see the module tests for the walking + categorization contract.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Whole-word HYPOTHESIS match. Case-sensitive: avoids matching prose like
# "the hypothesis is..." (lowercase) where research doc convention reserves
# "HYPOTHESIS" (all-caps) for load-bearing markers.
_HYPOTHESIS_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bHYPOTHESIS\b")

# Recognized prefix codes for categorization. Extend this set as new code
# families emerge in the research docs (e.g., new agent letters).
_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(BB-\d+|BC-\d+|BA-\d+|BD-\d+|TS-\d+|H-\d+|VS-\d+|A33-[\w-]+)\b"
)

# Directories to skip during walk. Hygiene/cache + venv + git internals.
_SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".venv",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
    }
)


@dataclass(frozen=True)
class HypothesisMarker:
    """Single HYPOTHESIS occurrence in the corpus."""

    file_path: Path
    line_num: int  # 1-indexed
    line_text: str  # stripped of leading/trailing whitespace
    prefix_codes: tuple[str, ...]  # may be empty if no prefix found


@dataclass(frozen=True)
class WalkReport:
    """Aggregate result of a `walk_hypothesis()` call."""

    markers: tuple[HypothesisMarker, ...]
    root: Path

    @property
    def total(self) -> int:
        return len(self.markers)

    def by_file(self) -> dict[str, int]:
        """Marker count per file (sorted file path -> count)."""
        counts: dict[str, int] = {}
        for m in self.markers:
            key = str(m.file_path.relative_to(self.root))
            counts[key] = counts.get(key, 0) + 1
        # Insertion-stable already (Python dict ordering); re-sort by key for
        # deterministic CLI output.
        return dict(sorted(counts.items()))

    def by_prefix(self) -> dict[str, int]:
        """Marker count per prefix family (e.g., 'BB', 'BC', 'TS', 'A33')."""
        counts: dict[str, int] = {}
        for m in self.markers:
            for prefix in m.prefix_codes:
                family = prefix.split("-")[0]
                counts[family] = counts.get(family, 0) + 1
        return dict(sorted(counts.items()))

    def unprefixed_count(self) -> int:
        """Markers where no prefix code was extracted (free-text HYPOTHESIS)."""
        return sum(1 for m in self.markers if not m.prefix_codes)

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        lines = [
            f"HYPOTHESIS walker — root={self.root}",
            f"  Total markers: {self.total}",
            f"  Unprefixed (free-text):   {self.unprefixed_count()}",
            "",
            "  By file:",
        ]
        for fname, count in self.by_file().items():
            lines.append(f"    {fname}: {count}")
        lines.append("")
        lines.append("  By prefix family:")
        for family, count in self.by_prefix().items():
            lines.append(f"    {family}: {count}")
        return "\n".join(lines)


def _is_skipped_dir(path: Path) -> bool:
    """True if any path segment is in _SKIP_DIRS or starts with '.'.

    Top-level '.' files (like .gitignore) are NOT skipped; only directories.
    """
    return any(part in _SKIP_DIRS for part in path.parts)


def _extract_prefix_codes(line: str) -> tuple[str, ...]:
    """Return all distinct prefix codes found on `line`, preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _PREFIX_PATTERN.finditer(line):
        code = match.group(1)
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return tuple(ordered)


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    """Yield all `*.md` files under `root` excluding _SKIP_DIRS."""
    for md_path in sorted(root.rglob("*.md")):
        if _is_skipped_dir(md_path.relative_to(root)):
            continue
        yield md_path


def _scan_file(md_path: Path) -> list[HypothesisMarker]:
    """Return all HYPOTHESIS markers in `md_path`. Returns [] on read error."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    markers: list[HypothesisMarker] = []
    for line_num, raw in enumerate(text.splitlines(), start=1):
        if not _HYPOTHESIS_PATTERN.search(raw):
            continue
        markers.append(
            HypothesisMarker(
                file_path=md_path,
                line_num=line_num,
                line_text=raw.strip(),
                prefix_codes=_extract_prefix_codes(raw),
            )
        )
    return markers


def walk_hypothesis(root: Path) -> WalkReport:
    """Scan all `*.md` files under `root` for HYPOTHESIS markers.

    Args:
        root: Project root directory to walk recursively. Hygiene directories
              (`.venv`, `.git`, caches, `__pycache__`) are skipped.

    Returns:
        `WalkReport` with markers + helpers for by-file + by-prefix counts.
        Empty if no `.md` files exist under `root` or `root` does not exist.
    """
    if not root.exists() or not root.is_dir():
        return WalkReport(markers=(), root=root)
    all_markers: list[HypothesisMarker] = []
    for md_path in _iter_markdown_files(root):
        all_markers.extend(_scan_file(md_path))
    return WalkReport(markers=tuple(all_markers), root=root)


def _main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 OK, 1 on error)."""
    parser = argparse.ArgumentParser(
        prog="python -m futur3.tools.verify_facts",
        description=(
            "HYPOTHESIS walker — enumerate load-bearing claims in research "
            "+ canonical docs that need external verification."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (default: cwd)",
    )
    args = parser.parse_args(argv)

    report = walk_hypothesis(args.root)
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(_main())


__all__: list[str] = [
    "HypothesisMarker",
    "WalkReport",
    "walk_hypothesis",
]
