# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-02 12:55] - method_runs
- Read the paper and identified the actual algorithm: structured multimodal parsing followed by Bayesian inference over object-placement hypotheses.
- Implemented the paper's hypothesis enumeration, visual rejection sampling, audio posterior products, posterior normalization, and marginalization.
- Added audio-only and vision-only ablations and a method-agnostic evaluator.
- Ran a reduced-sample smoke test end to end; full model output, ablation outputs, and `scoring/scores.json` were produced successfully.

## Stop Justification
- Not stopped. Submitting a CPU job to run the configured 1000-sample reproduction and validation.
