# Instruction Separation Checklist

Use this checklist before adding or changing agent-facing instructions.

## Purpose

Ensure that agents building this repository do not confuse repository-development instructions with downstream user-facing SBI instructions.

## Required structure

- [ ] Root `AGENTS.md` exists.
- [ ] Root `AGENTS.md` contains `MODE: REPOSITORY-BUILDER`.
- [ ] Root `AGENTS.md` clearly states that agents are building the guidance kit, not conducting SBI.
- [ ] Root `AGENTS.md` says not to run SBI unless explicitly asked for a smoke test, validation test, or example test.
- [ ] Root `AGENTS.md` says not to follow `templates/user-repo/AGENTS.md` as active instructions.
- [ ] `templates/user-repo/AGENTS.md` exists.
- [ ] `templates/user-repo/AGENTS.md` contains `MODE: USER-SBI-ASSISTANT`.
- [ ] Canonical user-facing skills live in `skills/`.
- [ ] Builder-only skills live in `.agent-build/skills/`.
- [ ] User-facing skills live under `skills/`.

## Required warnings

Root `AGENTS.md` must say:

- [ ] Do not train BayesFlow models unless explicitly requested for testing.
- [ ] Do not infer parameters.
- [ ] Do not interpret posterior distributions.
- [ ] Do not modify mechanistic simulator logic.
- [ ] Treat files in `templates/`, `skills/`, and `docs/` as product content while working in this repository.

## Pass/fail

This checklist passes only if all items above are checked.
