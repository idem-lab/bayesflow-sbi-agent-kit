# Acceptance Criteria

This document defines the minimum criteria before the repository can be recommended for use.

## 1. Instruction separation

The repository must clearly separate:

- builder-agent instructions
- user-agent instructions
- builder skills
- user-facing skills

Acceptance criteria:

- [ ] Root `AGENTS.md` is builder-facing.
- [ ] `templates/user-repo/AGENTS.md` is user-facing.
- [ ] `skills/` contains user-facing skills.
- [ ] `.agent-build/skills/` contains builder-only skills.
- [ ] No root instruction tells builder agents to conduct SBI.

## 2. Guardrails

Acceptance criteria:

- [ ] Agents are forbidden from modifying mechanistic assumptions without explicit human approval.
- [ ] Agents are forbidden from inventing priors.
- [ ] Agents are forbidden from silently transforming observed data.
- [ ] Agents are forbidden from making posterior conclusions without diagnostics.
- [ ] Human approval gates are stated clearly.

## 3. Bayesian workflow

Acceptance criteria:

- [ ] Project intake is required before inference.
- [ ] Prior predictive checks are required before fitting observed data.
- [ ] Parameter recovery is required before serious use.
- [ ] Simulation-based calibration is required before reporting posterior results.
- [ ] Posterior predictive checks are required before scientific interpretation.
- [ ] Failed diagnostics are treated as workflow/model-criticism signals, not ignored.

## 4. Local-agent compatibility

Acceptance criteria:

- [ ] Core skills live in `skills/`.
- [ ] User instructions use local relative paths.
- [ ] GitHub-specific integrations are optional.
- [ ] The user workflow works when this repo is added as a submodule.

## 5. Examples

Acceptance criteria:

- [ ] Toy examples are labelled as workflow tests.
- [ ] Scientific examples include human-validated notes.
- [ ] Examples separate simulator code from inference code.
- [ ] Examples do not encourage unauthorised scientific model changes.

## 6. Release threshold

A release may proceed only if:

- [ ] Instruction separation tests pass.
- [ ] Red-team tests pass or failures are documented.
- [ ] All required templates exist.
- [ ] User-facing docs explain the Bayesian workflow.
- [ ] Maintainers approve unresolved limitations.
