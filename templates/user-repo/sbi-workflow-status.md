# SBI workflow status

_A living record of this Bayesian simulation-based inference project. The assistant
maintains it; you can read it any time. It tracks where we are, what has been
decided, and every time we looped back to an earlier step. **If you ever feel lost,
read the "You are here" block and the iteration log below.**_

_The workflow is a loop, not a line — going back a step when a check fails is the
process working, not failing. See the `workflow-orchestration` skill for the full
map._

---

## You are here

- **Stage:** _<n>_/10 — _<stage name>_
- **Iteration:** _<k>_
- **Waiting on:** _nothing_ — or — _a decision from you: <what>_
- **Next:** _<what happens next>_
- **Last updated:** _<date>_

---

## Stage checklist

Legend:  ☐ not started · ▶ in progress · ✅ passed · ⚠ failed → looped back

**Act I — set up the model & your beliefs**
- ☐ 1. Project intake
- ☐ 2. Prior review — **needs your approval**
- ☐ 3. Prior predictive check

**Act II — prove the method works on *simulated* data**
- ☐ 4. BayesFlow workflow design
- ☐ 5. Pilot training
- ☐ 6. Parameter recovery
- ☐ 7. Calibration (SBC)
- ☐ 8. Posterior predictive check — **needs your approval for any model change**

**Act III — use it on your *real* data & review**
- ☐ 9. Real-data inference + reliability (out-of-distribution) check — **needs your approval to widen priors**
- ☐ 10. Human review — **you interpret the results and sign off**

---

## Decisions log

Scientific decisions you have approved (priors, model or simulator changes). Nothing
here is changed without your explicit say-so.

| Date | Decision | Approved by |
|------|----------|-------------|
|      |          |             |

---

## Iteration log

Each time a check sent us back to an earlier stage, written as one plain sentence so
the path stays easy to follow.

- _(none yet)_
