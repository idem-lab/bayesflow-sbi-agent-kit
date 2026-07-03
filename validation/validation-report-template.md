# Validation Report

## Repository version

- Repository:
- Commit:
- Branch:
- Date:
- Validator:
- Agent/tool used:

## Scope

What was validated?

- [ ] Instruction separation
- [ ] Builder-agent behaviour
- [ ] User-agent behaviour
- [ ] Skills
- [ ] Templates
- [ ] Documentation
- [ ] Examples
- [ ] BayesFlow smoke tests
- [ ] R/reticulate smoke tests
- [ ] Red-team prompts

## Summary result

Overall result:

- [ ] Pass
- [ ] Pass with minor issues
- [ ] Fail

Summary:

```text
Brief summary of validation outcome.
```

## Instruction separation results

- Root AGENTS.md checked: yes/no
- User template AGENTS.md checked: yes/no
- Skills checked: yes/no

Findings:

```text
Record findings here.
```

## Red-team results

Number of prompts tested:

Number passed:

Number failed:

Critical failures:

```text
List any critical failures.
```

## Template results

Templates checked:

- [ ] templates/user-repo/AGENTS.md
- [ ] project templates
- [ ] Python templates
- [ ] R templates

Findings:

```text
Record findings here.
```

## Example results
Examples checked:
```text
List examples checked.
```

Findings:
```text
Record findings here.
```

## Issues found


| Issue ID | Issue | Severity | File/location | Required action | Owner | Status |
|---|---|---|---|---|---|---|
| ISSUE-001 | Brief description of the issue. | Low / Medium / High / Critical | `path/to/file.md` | Describe the fix required. | TBD | Open |
| ISSUE-002 | Brief description of the issue. | Low / Medium / High / Critical | `path/to/file.md` | Describe the fix required. | TBD | Open |

## Severity definitions

- **Critical**: Could cause agents to modify scientific mechanisms, invent priors, skip required diagnostics, or confuse builder/user instructions.
- **High**: Could materially mislead users or cause unsafe/incomplete SBI workflows.
- **Medium**: Important documentation, template, or workflow issue, but unlikely to cause immediate unsafe behaviour.
- **Low**: Minor wording, formatting, clarity, or consistency issue.

## Release recommendation

- [ ] Ready for release
- [ ] Ready after minor fixes
- [ ] Not ready

Recommendation:

```text
Explain release recommendation.
```

## Human approvals

Approvals required before release:

- [ ] Maintainer approval
- [ ] Statistical workflow review
- [ ] Domain expert review for scientific examples
- [ ] Documentation review



