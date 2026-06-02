# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

## Stop Justification
- Completed at milestone `core_claim`.
- Exact Figure 3 reproduction is blocked by unreleased original WiTB videos, parsed neural outputs, and human ratings, so the packaged environment preserves the paper's algorithm on a scaled tracked benchmark.
