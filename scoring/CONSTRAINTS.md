# Scientist Constraints

## Data

Use `data/witb_parsed_benchmark.json` for scored evaluation. Do not add new
evaluation scenarios, replace target ratings, or tune on hidden variants and
then report them under `witb_human_alignment`.

## Outputs

Submissions must write object-in-left-box probabilities in the same JSON format
used by `outputs/witb_*_predictions.json`. The evaluator must remain
method-agnostic and must not import from `method/`.

## Resource Scaling

Do not improve scores by changing the benchmark targets, adding external human
ratings, or evaluating on a different scenario distribution under the same
experiment name. Any larger models or external parsers used to improve the
structured cue estimates must be run before compute submission and documented.

## Compliance Checklist

- [ ] Evaluation uses `data/witb_parsed_benchmark.json`.
- [ ] `scoring/scores.json` is written by `scripts/evaluate.sh`.
- [ ] Evaluation code does not import from `method/`.
- [ ] Reported `witb_human_alignment` scores use the fixed benchmark targets.
