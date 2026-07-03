# Release Checklist

The single gating checklist before this guidance kit can be recommended for use.

Companion files in this directory:

- `red-team-prompts.md` — guardrail probes to run against an agent, with a results log.
- `benchmark-matrix.md` — example-model coverage tracker (aspirational; toy model first).
- `validation-report-template.md` — form to record a full validation run.

Much of Section 1 is checked automatically by `scripts/check_structure.py`.
Run that first; the remaining sections are manual review.

---

## 1. Instruction separation

Builder-agent instructions and user-facing SBI instructions must never be confused.

Structure:

- [ ] Root `AGENTS.md` exists and contains `MODE: REPOSITORY-BUILDER`.
- [ ] Root `AGENTS.md` states that agents are building the guidance kit, not conducting SBI.
- [ ] Root `AGENTS.md` says not to run SBI unless explicitly asked for a smoke test, validation test, or example test.
- [ ] Root `AGENTS.md` says not to follow `templates/user-repo/AGENTS.md` as active instructions.
- [ ] `templates/user-repo/AGENTS.md` exists and contains `MODE: USER-SBI-ASSISTANT`.
- [ ] Canonical user-facing skills live under `skills/`.
- [ ] Builder-only skills live under `.agent-build/skills/`.

Required warnings in root `AGENTS.md`:

- [ ] Do not train BayesFlow models unless explicitly requested for testing.
- [ ] Do not infer parameters.
- [ ] Do not interpret posterior distributions.
- [ ] Do not modify mechanistic simulator logic.
- [ ] Treat files in `templates/`, `skills/`, and `docs/` as product content while working in this repository.

## 2. Guardrails

- [ ] Agents are forbidden from modifying mechanistic assumptions without explicit human approval.
- [ ] Agents are forbidden from inventing priors.
- [ ] Agents are forbidden from silently transforming observed data.
- [ ] Agents are forbidden from making posterior conclusions without diagnostics.
- [ ] Human approval gates are stated clearly.

## 3. Bayesian workflow

Required workflow order (agents must not skip or reorder without human approval):

1. Project intake
2. Simulator audit
3. Prior specification
4. Prior predictive checks
5. BayesFlow workflow design
6. Pilot training
7. Parameter recovery
8. Simulation-based calibration
9. Fit to observed data
10. Posterior predictive checks
11. Human review

Gating criteria:

- [ ] Project intake is required before inference.
- [ ] Prior predictive checks are required before fitting observed data.
- [ ] Parameter recovery is required before serious use.
- [ ] Simulation-based calibration is required before reporting posterior results.
- [ ] Posterior predictive checks are required before scientific interpretation.
- [ ] Failed diagnostics are treated as workflow/model-criticism signals, not ignored.

## 4. Templates & examples

- [ ] Prior templates say priors require human approval.
- [ ] Inference templates do not edit simulator code.
- [ ] Diagnostic templates distinguish inference failure from model failure.
- [ ] Templates separate simulator, inference, and diagnostics.
- [ ] Toy examples are labelled as workflow tests, not scientific evidence.
- [ ] Scientific examples include human-validated notes.
- [ ] Examples do not encourage unauthorised scientific model changes.

## 5. Local-agent compatibility

- [ ] Core skills live in `skills/`.
- [ ] User instructions use local relative paths.
- [ ] Any GitHub-specific integrations are optional, not required.
- [ ] The user workflow works when this repo is added as a submodule.

## 6. Red-team

- [ ] All prompts in `red-team-prompts.md` have been run.
- [ ] Red-team results are recorded and any failures documented with remediation.

## 7. Release threshold

A release may proceed only if:

- [ ] `scripts/check_structure.py` passes.
- [ ] Instruction separation (Section 1) passes.
- [ ] Red-team tests pass or failures are documented.
- [ ] All required templates exist.
- [ ] User-facing docs explain the Bayesian workflow.
- [ ] Maintainers approve any unresolved limitations.
