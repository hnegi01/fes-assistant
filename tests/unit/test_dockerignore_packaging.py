"""The build context must actually contain the files the images COPY.

`.dockerignore` is applied to the build CONTEXT, so a `COPY skills ./skills`
can succeed and still ship an empty directory when a broad pattern filters the
files inside it. That is not hypothetical: `**/*.md` (added for repo docs) also
matched `skills/*/SKILL.md`, and 2.7.0 shipped the entire skills feature inert
-- the planner saw no skills, every local check passed, and nothing was logged,
because `load_skills` returns an empty dict silently when the glob finds
nothing.

These tests evaluate the real `.dockerignore` against the real files on disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]


def _patterns() -> List[str]:
    raw = (ROOT / ".dockerignore").read_text().splitlines()
    return [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]


def _to_regex(pattern: str) -> re.Pattern[str]:
    """Docker's context matcher, for the subset of syntax this repo uses.

    `**/` spans zero or more directories, `**` spans anything, `*` stops at a
    separator. A trailing slash means "this directory and everything under it".
    """
    pat = pattern.rstrip("/")
    out, i = [], 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    # match the path itself, or anything beneath it when it names a directory
    return re.compile("^" + "".join(out) + "(?:/.*)?$")


def _is_excluded(rel_path: str) -> bool:
    """Last matching pattern wins; a leading `!` re-includes."""
    excluded = False
    for pattern in _patterns():
        negated = pattern.startswith("!")
        body = pattern[1:] if negated else pattern
        if _to_regex(body).match(rel_path):
            excluded = not negated
    return excluded


def test_every_skill_file_reaches_the_build_context() -> None:
    skill_files = sorted((ROOT / "skills").glob("*/SKILL.md"))
    assert skill_files, "no SKILL.md on disk -- this guard would pass vacuously"
    for f in skill_files:
        rel = f.relative_to(ROOT).as_posix()
        assert not _is_excluded(rel), (
            f"{rel} is excluded from the Docker build context, so the deployed "
            f"backend would run with zero skills while every local check passes. "
            f"Add a `!` negation AFTER the pattern that excludes it."
        )


def test_repo_markdown_is_still_excluded() -> None:
    """The skills negation must not re-admit ordinary repo documentation."""
    for rel in ("README.md", "CLAUDE.md", "docs/architecture.md", "docs/design/skills.md"):
        assert _is_excluded(rel), f"{rel} should not be shipped in images"


def test_matcher_agrees_with_the_bug_we_shipped() -> None:
    """Pin the matcher itself: `**/*.md` alone must exclude a skill file."""
    saved = _patterns
    try:
        globals()["_patterns"] = lambda: ["**/*.md"]
        assert _is_excluded("skills/x/SKILL.md")
        globals()["_patterns"] = lambda: ["**/*.md", "!skills/**/*.md"]
        assert not _is_excluded("skills/x/SKILL.md")
    finally:
        globals()["_patterns"] = saved
