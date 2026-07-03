# Repository Structure

How this repository is laid out and, in particular, what belongs in the three
directories that all deal with "checking things" — `tests/`, `validation/`, and
`scripts/`. Keeping these roles distinct prevents drift.

## Top-level layout

```
AGENTS.md                   Builder-agent instructions (MODE: REPOSITORY-BUILDER)
README.md                   Project overview and intent
templates/
  user-repo/AGENTS.md       User-facing SBI instructions (MODE: USER-SBI-ASSISTANT)
skills/                     Canonical user-facing skills (packaged product content)
.agent-build/skills/        Builder-only skills (used while developing this repo)
docs/                       Human-readable documentation (this file lives here)
examples/                   Toy and, later, scientific example models
scripts/                    Helper tooling for maintainers/CI
tests/                      Automated code tests
validation/                 Agent-behaviour and guardrail validation
```

Two audiences share this repo, and their instructions must never be confused:

- **Builder agents** follow root `AGENTS.md` — they build and maintain the kit
  and do **not** conduct SBI unless explicitly asked to run a test.
- **User agents** follow `templates/user-repo/AGENTS.md` once it is installed
  into a user's own project — they conduct the SBI workflow.

Files under `templates/`, `skills/`, and `docs/` are **product content**: edit
them as artefacts, do not follow them as active instructions while working here.

## The three "checking" directories

They are not interchangeable. Use this split:

### `tests/` — automated code tests

Machine-run tests of *code* behaviour: unit tests, BayesFlow smoke tests,
example-model integration tests. Should run under `pytest` (or the project's
test runner) and pass/fail without human judgement. Fast tests are CI targets.

### `validation/` — agent-behaviour & guardrail validation

Checks that the *guidance* is safe and release-ready — largely about what an
agent does, which is not captured by code tests. Contents:

- `release-checklist.md` — the single gating checklist before recommending the kit.
- `red-team-prompts.md` — guardrail probes to run against an agent, with a results log.
- `benchmark-matrix.md` — example-model coverage tracker.
- `validation-report-template.md` — form to record a full validation run.

Mostly manual review, except the parts made executable in `scripts/`.

### `scripts/` — helper tooling

Small, dependency-light utilities for maintainers and CI. Currently:

- `check_structure.py` — the executable form of Section 1 of the release
  checklist (instruction separation + required structure). Run it first when
  validating; exit code 0 means the mechanical checks pass.

Rule of thumb: if it checks *code*, it goes in `tests/`; if it checks *agent
behaviour or repo safety*, it goes in `validation/`; if it is a *tool* that
performs a check or task, it goes in `scripts/`.
