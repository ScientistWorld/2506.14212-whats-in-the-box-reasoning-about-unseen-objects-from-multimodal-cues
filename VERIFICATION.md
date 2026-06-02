# Verification Report

## Initial Assessment

The workspace started at milestone `secondary_claims`. The claimed milestone is
substantively justified for the documented scaled reproduction: the paper's
Bayesian multimodal reasoning engine runs end-to-end on the included parsed
WiTB benchmark, the full method beats the audio-only and vision-only reference
artifacts, and the late-fusion diagnostic supports a secondary comparison.

`python validate.py --compare` passed at the start of verification. The main
remaining risk was gym usability for future methods, especially the train/test
split evaluators.

## Issues Found

- Minor: `eval/train/evaluate_witb_train.py` and `eval/test/evaluate_witb_test.py`
  effectively treated `full_neurosymbolic` as the only candidate split-scored
  method. That was acceptable for the reproduced artifact but weak for future
  scientist agents with different method names.
- Minor: `briefing/evaluation.md` and `scoring/CONSTRAINTS.md` described the
  output path as `outputs/witb_*_predictions.json`, even though the evaluator
  can score any `*_predictions.json` artifact.
- Known limitation: the original WiTB videos, parser outputs, and participant
  ratings are not publicly released. The scored data is a small tracked parsed
  substitute benchmark, and the reproduced correlations are higher than the
  paper's full benchmark numbers. This is documented and keeps confidence at
  medium rather than high.

No critical issues were found in method/evaluation separation, scored metric
definitions, URL/download portability, or train/test rewiring.

## Fixes Applied

- Updated the train/test split scorers so they exclude only known diagnostic
  baselines (`audio_only_ablation`, `vision_only_ablation`, and
  `uniform_chance_baseline`) and automatically score any other prediction
  artifact as a candidate method.
- Updated method-agnostic documentation to describe the generic
  `outputs/*_predictions.json` artifact interface.
- Re-ran `bash scripts/evaluate_train.sh outputs`,
  `bash scripts/evaluate_test.sh outputs`, and `python validate.py --compare`;
  all checks passed.

## Final Judgment

The workspace is ready to serve as a research gym in improve mode and is usable
in from-scratch mode. The core problem and evaluation files are method-agnostic,
the evaluator reads prediction artifacts rather than importing the method, and
the scored experiment combines benefit metrics with unimodal and late-fusion
comparison constraints.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: medium
- Key limitation: the scored dataset is a small parsed substitute benchmark
  rather than the paper's unreleased full human/video dataset.
