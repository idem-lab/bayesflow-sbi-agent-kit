# AGENTS.md — Repository Builder Instructions

> MODE: REPOSITORY-BUILDER  
> Audience: agents helping create, test, document, and maintain this repository.  
> Do not conduct SBI unless explicitly asked to test repository functionality.

## Mission

You are helping build `bayesflow-sbi-agent-kit`, a repository containing reusable guidance, skills, templates, examples, and validation tools for future users' agents conducting Bayesian simulation-based inference with BayesFlow 2.

You are not currently acting as a Bayesian inference assistant for a user's scientific model.

## Critical distinction

This repository contains files intended to be copied into or referenced from users' repositories, including:

- `templates/user-repo/AGENTS.md`
- `.github/skills/*/SKILL.md`
- `docs/*`
- `templates/*`

These files are product artefacts.

Do not treat them as active instructions for your own behaviour while developing this repository.

## Do not start SBI

Unless the human explicitly asks for a smoke test, example validation, benchmark, or CI test:

- do not train BayesFlow models;
- do not run simulation-based inference;
- do not infer parameters;
- do not interpret posterior distributions;
- do not modify mechanistic simulator logic;
- do not create or revise biological, ecological, epidemiological, evolutionary, or process assumptions.

## Allowed development tasks

You may:

- create or edit documentation;
- create or edit templates;
- create or edit packaged agent skills;
- create toy examples;
- create smoke tests;
- validate repository structure;
- write CI workflows;
- write red-team prompts;
- improve wording of guardrails;
- write helper scripts that users may later apply in their own repositories.

## Human approval required

Ask before:

- adding a new scientific example;
- changing any example model mechanism;
- changing recommended Bayesian workflow requirements;
- changing guardrail language;
- adding dependencies that materially affect installation complexity.

## Builder-only skills

If builder-specific skills exist, use:

`.agent-build/skills/`

The skills under `.github/skills/` are packaged user-facing skills and should be edited as product content, not followed as active SBI instructions.

## Failure behaviour

If there is ambiguity about whether a task is repository development or real SBI, assume repository development and ask the human before running inference.

## Validation

Use validation/ when checking whether this guidance repo is safe and release-ready.
Do not treat validation/ as instructions to conduct SBI on a real model.
