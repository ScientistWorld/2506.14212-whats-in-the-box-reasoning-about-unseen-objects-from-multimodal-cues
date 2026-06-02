# Evaluation

## Metrics

The primary metric is Pearson correlation between a system's object-in-left-box
probabilities and average human object-in-left-box probability ratings across
all scenario/object queries. Higher correlation is better.

## Baselines and Targets

The paper reports these correlations on its unreleased 20-stimulus analysis set:
full system `r = 0.78`, audio-only `r = 0.55`, vision-only `r = 0.52`, and
Gemini 2.0 Flash `r = 0.31`. In this scaled environment, success is measured by
running the same evaluation metric on the included parsed benchmark and checking
that the multimodal system beats both unimodal reference methods.

## Evaluation Protocol

Methods write prediction artifacts to `outputs/witb_*_predictions.json`. Each
artifact must contain per-scenario object-in-left-box probabilities. Run
`bash scripts/evaluate.sh outputs data/witb_parsed_benchmark.json` to write
`scoring/scores.json`. Evaluation code must not call into `method/`; it only
reads predictions and the benchmark targets.
