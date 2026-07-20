---
name: real-data-reliability
description: Runs the real-data inference + reliability (out-of-distribution) check — put the actual observations through the trained network AND verify the amortised posterior is trustworthy for this data. An amortised posterior is only reliable on data resembling the training simulations; a real dataset unlike anything seen in training is confidently wrong with no error message. Check the real data is typical under the prior-predictive/training distribution before trusting or interpreting the posterior. Use when moving to real data (Act III); it gates the posterior predictive check and human review. On OOD, route back to stage 2 — widening priors/simulator ranges is the human's scientific call.
---

# Real-data inference + reliability (out-of-distribution) check

This is the first Act III stage: the point where the actual observations finally
drive inference. Its job is **two things at once** — produce the posterior for the
real data, *and* decide whether that posterior can be trusted at all. The second is
not optional, and it is the single most important safeguard specific to **amortised**
SBI.

## Why the reliability check is non-negotiable in amortised SBI

An amortised posterior is a network trained to do inference over the region the
**prior predictive distribution** samples (see `prior-elicitation`, `prior-predictive-check`).
It has only ever seen simulated data from that region. Hand it a real dataset that
looks like nothing in training and it does not error, refuse, or widen — it
**extrapolates and returns a confident, wrong posterior with no warning.** There is
no likelihood evaluation to catch it, which for many intractable-likelihood SBI
problems is precisely why you are using amortised SBI in the first place.

This is the same concern as **coverage** at stage 3, now tested for real. Stage 3
asked "will the prior predictive *cover* the plausible real data?" as a *prediction*;
this stage *tests that prediction against the actual observations*. If stage-3
coverage was assessed honestly against a good description of the real data, this stage
rarely surprises. If it was skipped or done loosely, this is where it bites.

## What to do

1. **Produce the posterior for the real observation(s)** via `workflow.sample(...)`.
   Keep it — but do not interpret it yet.
2. **Run the out-of-distribution (OOD) check *before* trusting the posterior.**
   Verify the real dataset is *typical* under the training / prior-predictive
   distribution:
   - Compute the **summary-network embedding** (or, as a fallback, simple data
     summaries) for the real data **and** for a large prior-predictive sample.
   - Check the real data sits **inside that cloud** — e.g. Mahalanobis distance of the
     real summary vs. the simulated-summary distribution, or eyeball low-dimensional
     projections (PCA) with the real point overlaid. This is the same simulated-summary
     cloud you built at stage 3; now you are plotting the real data into it.
   - Formal option: a summary-space two-sample discrepancy (e.g. MMD between real and
     simulated summaries) as used in the SBI misspecification-detection literature
     (Schmitt et al. 2023).
3. **Also check the posterior lands in a well-trained region.** Data-space typicality
   is necessary but not the whole story: if the posterior mass sits in a corner of
   parameter space the training barely sampled, the network is extrapolating there
   too. Sanity-check that the posterior sits within the prior's bulk, not out on a tail
   the flow rarely saw.
4. **If the real data is OOD, the posterior is *not* to be trusted as-is** — stop and
   route (below). Do **not** hand an OOD posterior on to the posterior predictive check
   or to human review; both would be interpreting a number the method cannot stand
   behind.

## On fail (real data is OOD)

An OOD result means the training distribution did not cover the real data — the priors
or simulator ranges are likely too narrow, or the model is misspecified. Route back to
**stage 2**. Your job is to **diagnose and present**, not to force the data
in-distribution:

- Show *which* summaries of the real data fall outside the simulated cloud, and which
  priors or mechanisms would have to move to cover them.
- **GATE:** widening a prior or a simulator range is a **scientific** change the human
  must decide and approve — never something you do on your own to make the real data
  look in-distribution. Once they decide, you retrain, then re-run this check.

**Rescue options (use with care, and say what they assume):**

- **Importance-sampling reweighting** of the amortised draws can rescue *mild* OOD
  cases by correcting the amortised posterior toward the true one — but it needs a
  usable importance weight and degrades as the mismatch grows.
- A **likelihood-based fallback** — e.g. MCMC seeded from the amortised draws — is the
  gold-standard escalation *when a likelihood is available*. For many
  intractable-likelihood SBI problems it is **not**, which is exactly why the OOD check
  carries so much weight here: often there is no cheap second opinion, so detecting the
  problem is the whole defence.

These rescues are engineering, but the underlying question — is the model/prior wide
enough or right? — is science, and the retrain-after-widening path goes through the
stage-2 gate.

## On pass

The real data is in-distribution and the posterior is trustworthy as an amortised
result. Two things follow:

- Proceed to the **posterior predictive check**, which this stage gates: because the
  posterior is now known to be reliable *and* the engine was already validated on
  simulated data (recovery + SBC, stages 6–7), a posterior predictive *failure* can be
  read as **model misspecification** rather than an inference-network fault. That chain
  — in-distribution here + calibrated there + poor fit → the model, not the engine — is
  the SBI-specific reasoning that makes the posterior predictive check conclusive.
- Carry the posterior, the OOD result, and all Act II diagnostics into **human review**.

## Precedence note

Whatever the surrounding stage numbering, **this reliability/OOD check must precede
interpreting the posterior predictive check and the human review** — an untrustworthy
(OOD) posterior makes both meaningless. See `posterior-predictive-check`.

## Using this skill

- **Entering from Act II:** you have a trained, recovery- and SBC-validated network.
  Now, and only now, bring in the real observations.
- **On OOD:** route to `prior-elicitation` (stage 2) with a specific diagnosis; the
  widen-and-retrain decision is the human's, through the gate.
- **Engineering mechanics** (`workflow.sample`, summary embeddings) are in
  `bayesflow-implementation`. The worked reference is
  `examples/toy-normal/reliability.py`, which runs exactly this check — BayesFlow's
  `summary_space_comparison` (MMD in summary space vs. a prior-predictive reference,
  with a bootstrap null) on an in-distribution and an out-of-distribution sample.

## References

- *Amortized Bayesian Workflow* (2024) — the amortised-specific additions used here:
  out-of-distribution detection at inference time, importance-sampling correction, and
  the MCMC fallback. arXiv:2409.04332 — https://arxiv.org/abs/2409.04332
- Schmitt, Radev, Bürkner & Köthe (2023), *Detecting Model Misspecification in
  Amortized Bayesian Inference with Neural Networks* — summary-space (MMD) detection of
  data that the trained network cannot be trusted on. arXiv:2112.08866 —
  https://arxiv.org/abs/2112.08866
- Hermans, Delaunoy, Rozet, Wehenkel, Begy & Louppe (2021), *A Trust Crisis in
  Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful* — why
  amortised/SBI posteriors must be checked for reliability rather than trusted
  blindly. arXiv:2110.06581 — https://arxiv.org/abs/2110.06581
