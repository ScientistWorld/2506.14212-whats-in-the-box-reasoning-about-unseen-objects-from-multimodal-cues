# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 12:58] - method_runs
- Implemented the paper's Bayesian hypothesis enumeration, visual rejection-sampling fit check, audio evidence product, posterior normalization, and marginalization on a scaled parsed WiTB benchmark.
- Added scripts for the full method, unimodal ablations, and method-agnostic evaluation.

### [2026-06-02 13:00] - core_claim
- Ran the scaled benchmark with 1000 visual samples per box hypothesis.
- The full neurosymbolic model beat both unimodal ablations on human-correlation, supporting the paper's central multimodal-integration claim in the packaged environment.

### [2026-06-02 13:03] - core_claim
- Verified the corrected audio-only Eq. 3 ablation and multimodal-gain scoring in CPU container job `2029a1ef-afb`.
- `validate.py --compare` passed with full `r = 0.983`, audio-only `r = 0.796`, vision-only `r = 0.678`, and multimodal gain `0.187`.
