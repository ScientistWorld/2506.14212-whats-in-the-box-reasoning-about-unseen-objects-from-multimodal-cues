# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `joint_gain_over_late_fusion = 0.044`.

## Stop Justification
- Completed at milestone `secondary_claims`.
