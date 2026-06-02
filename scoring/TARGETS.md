# Evaluation Targets

## Primary: witb_human_alignment

Tests whether object-location probability judgments align with human judgments
on the hidden-object guessing task.
- **Metric**: `human_correlation` (higher is better)
- **Run**: `bash scripts/evaluate.sh outputs data/witb_parsed_benchmark.json`

## Constraint: witb_human_alignment

The same experiment must include unimodal reference methods so improvement
cannot come from ignoring the paper's multimodal-integration claim. A valid
submission should compare the main system against audio-only and vision-only
outputs using the same scenario inputs and target judgments.
- **Metric**: `human_correlation` for `audio_only_ablation` and `vision_only_ablation` (higher is better for each baseline, but the main system should exceed both)
- **Metric**: `multimodal_gain_over_best_unimodal` for `full_neurosymbolic` (higher is better; paper reference is `0.78 - max(0.55, 0.52) = 0.23`)
- **Run**: `bash scripts/reproduce.sh`

The evaluator also reports `_diagnostics.uniform_chance_baseline` as a
non-informative reference. This diagnostic is not a paper-reported baseline, but
it checks that the scaled parsed benchmark is not solved by constant object
location probabilities.

The auxiliary `scores_train.json` and `scores_test.json` files are split-slice
diagnostics for the main method; the paper-comparable ablation constraint is
scored in `scores.json`.

## Secondary: multi_cue_vs_late_fusion

The benchmark should distinguish integrated multi-cue reasoning from a simple
post-hoc average of single-cue probabilities. The evaluator therefore computes
a late-fusion average from the audio-only and vision-only artifacts and reports
the main system's gain over it.
- **Metric**: `gain_over_late_fusion` for `full_neurosymbolic` (higher is better)
- **Diagnostic**: `_diagnostics.late_fusion_average._human_correlation`
- **Run**: `bash scripts/evaluate.sh outputs data/witb_parsed_benchmark.json`
