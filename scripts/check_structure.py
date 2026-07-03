#!/usr/bin/env python3
"""Mechanical instruction-separation and structure checks for this repo.

This is the executable form of Section 1 of validation/release-checklist.md.
It verifies the things an agent could otherwise get subtly wrong: the two
AGENTS.md files carry the right MODE markers, required directories exist, skills
live under skills/ (never .github/skills/), and the builder file tells agents not
to follow the user template.

Run from the repository root:

    python3 scripts/check_structure.py

Exit code 0 = all checks passed, 1 = one or more failed. No dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that must exist for the kit's structure to be coherent.
REQUIRED_DIRS = [
    "skills",
    ".agent-build/skills",
    "docs",
    "examples",
    "scripts",
    "tests",
    "validation",
    "templates/user-repo",
]

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))


def read(rel: str) -> str | None:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.is_file() else None


# --- Required directories ---------------------------------------------------
for d in REQUIRED_DIRS:
    check((ROOT / d).is_dir(), f"directory exists: {d}/")

# --- Root AGENTS.md (builder mode) ------------------------------------------
root_agents = read("AGENTS.md")
check(root_agents is not None, "AGENTS.md exists")
if root_agents:
    check("MODE: REPOSITORY-BUILDER" in root_agents,
          "AGENTS.md declares MODE: REPOSITORY-BUILDER")
    check("templates/user-repo/AGENTS.md" in root_agents,
          "AGENTS.md references the user template")
    check("Do not treat them as active instructions" in root_agents
          or "not followed as active" in root_agents
          or "do not treat" in root_agents.lower(),
          "AGENTS.md says not to follow the user template as active instructions")

# --- User template (user SBI mode) ------------------------------------------
user_agents = read("templates/user-repo/AGENTS.md")
check(user_agents is not None, "templates/user-repo/AGENTS.md exists")
if user_agents:
    check("MODE: USER-SBI-ASSISTANT" in user_agents,
          "user template declares MODE: USER-SBI-ASSISTANT")

# --- Skills live under skills/, never .github/skills/ -----------------------
check((ROOT / "skills").is_dir(), "canonical user-facing skills dir skills/ exists")

offenders = []
for md in ROOT.rglob("*.md"):
    if ".git/" in md.as_posix():
        continue
    if ".github/skills" in md.read_text(encoding="utf-8"):
        offenders.append(md.relative_to(ROOT).as_posix())
check(not offenders,
      "no .github/skills references in markdown"
      + (f" (found in: {', '.join(offenders)})" if offenders else ""))

# --- Report -----------------------------------------------------------------
passed = sum(1 for ok, _ in results if ok)
for ok, msg in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
print(f"\n{passed}/{len(results)} checks passed")

sys.exit(0 if passed == len(results) else 1)
