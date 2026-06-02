# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 12:58] - method_runs
- Implemented the paper's Bayesian hypothesis enumeration, visual rejection-sampling fit check, audio evidence product, posterior normalization, and marginalization on a scaled parsed WiTB benchmark.
- Added scripts for the full method, unimodal ablations, and method-agnostic evaluation.

### [2026-06-02 13:10] - core_claim
- Ran the scaled benchmark with 1000 visual samples per box hypothesis.
- The full neurosymbolic model beat both unimodal ablations on human-correlation, supporting the paper's central multimodal-integration claim in the packaged environment.
