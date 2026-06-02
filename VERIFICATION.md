# Verification Report

## Initial Assessment

The workspace started at milestone `secondary_claims`. The claimed milestone is justified: the paper's symbolic multimodal reasoning procedure runs end-to-end on the included parsed WiTB benchmark, beats the unimodal reference artifacts, and reports an additional late-fusion comparison. Prior CPU container jobs completed successfully, and local validation passed at the start of final verification.

## Issues Found

- Minor: `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and `scoring/DIRECTION.md` contained wording that exposed the paper's internal inference framing in files intended to be method-agnostic or method-safe.
- Minor: the top-level evaluator only discovered `witb_*_predictions.json`, which could silently ignore a future scientist's correctly formatted output with a different method prefix.
- Minor: the late-fusion gain metric name used implementation-specific wording.

## Fixes Applied

- Rewrote the method-agnostic documentation to describe the observable evaluation goal: multi-cue probability judgments should improve over single-cue methods and simple post-hoc averaging.
- Broadened `eval/evaluate_witb.py` to score any `*_predictions.json` artifact in the output directory, matching the train/test evaluator interface.
- Renamed `joint_gain_over_late_fusion` to `gain_over_late_fusion` across evaluators, reference scores, reproduced scores, and documentation.
- Re-ran `bash scripts/evaluate.sh`, `bash scripts/evaluate_train.sh`, `bash scripts/evaluate_test.sh`, and `python validate.py --compare`; all checks passed.

## Final Judgment

The workspace is ready to serve as a research gym in improve mode and is usable in from-scratch mode with the documented limitation that the original WiTB videos, parser outputs, and participant ratings were not publicly released. The problem/evaluation/data files are method-agnostic, evaluation reads prediction artifacts rather than method code, scoring includes benefit and constraint/comparison metrics, and train/test artifacts are present and consistent.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: medium
- Key limitation: the scored dataset is a small tracked parsed substitute benchmark rather than the paper's unreleased full human/video dataset.
