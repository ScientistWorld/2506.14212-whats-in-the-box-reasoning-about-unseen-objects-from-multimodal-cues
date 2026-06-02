# Reproduction Milestones

**Current: core_claim**

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

### [2026-06-02 12:58] - core_claim
- CPU job `abff144f-319` completed with exit code 0 inside the container.
- The configured 1000-sample reproduction produced `human_correlation = 0.983` for the full method.
- The full method beat both unimodal ablations on the same scaled benchmark: audio-only `0.804`, vision-only `0.678`.
- `python validate.py --compare` passed in the job, including scoring structure and separation checks.

## Stop Justification
- Completed at milestone `core_claim`.
