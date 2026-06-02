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
- **Run**: `bash scripts/reproduce.sh`
