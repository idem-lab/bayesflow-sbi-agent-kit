# BayesFlow SBI Agent Kit

This repository provides agent guidance, templates, skills, examples, and validation tools to support principled Bayesian simulation-based inference of mechanistic simulation models using BayesFlow 2.

Note that this entire repository and approach is an ongoing experiment in the feasibility of developing an expert AI agent to facilitate the technical and engineering aspects of performing Bayesian inference on complex models, enabling modellers to focus on the models and research questions themselves.

The target audience (users) of this repository are domain-specific modellers (e.g. biologists, epidemiologists) without detailed knowledge of statistics or computer science. The experiment will have a positive result if such a user can interact with the agent to efficiently perform inference on their complex model, learn about principled Bayesian inference workflow along the way, and without abdicating any of the *scientific* process or decision-making to the agent.

If you have tried using this repository and found that the agents are not meeting this goal, please open a GitHub issue to document the failure. It is only through documenting and addressing these failure cases that this system can improve.

## Important instruction separation

This repository itself is being developed using agentic AI coding tools (GPT and Claude).  

The root `AGENTS.md` is for agents helping build this repository.

The user-facing agent instructions are stored in:

- `packaged-guidance/AGENTS.md`
- `templates/user-repo/AGENTS.md`

Agents working on this repository should not conduct SBI unless explicitly asked to run a test, smoke test, or validation example.

## Intended downstream use

A user may add this repository to their own project, for example as a git submodule:

```bash
git submodule add https://github.com/YOUR-ORG/bayesflow-sbi-agent-kit.git agentic-sbi
cp agentic-sbi/templates/user-repo/AGENTS.md ./AGENTS.md
```
