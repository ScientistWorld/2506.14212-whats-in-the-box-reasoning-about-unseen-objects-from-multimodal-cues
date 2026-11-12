# Scientist Constraints

## Interface

- A method is `method/predictors/<name>.sh`, called as
  `bash method/predictors/<name>.sh INPUT_JSON OUTPUT_JSON` from the workspace
  root. Output format and validity rules are in `briefing/evaluation.md`.
- Every predictor on disk is scored, on the visible train split by
  `scripts/evaluate_train.sh` and on the hidden test by
  `scripts/evaluate_test.sh`.
- Predictors run offline on one CPU core, with a 2-hour limit each, and must
  be deterministic (fix any random seeds).

## Hold fixed

- Do not modify `eval/`, `scripts/evaluate*.sh`, `scoring/reference.json` or
  the visible data `data/witb_train.json`. Scores are recomputed from your
  predictor's `box_probs`; score files or extra fields you write are ignored.
- Use only the cue fields in the input. Do not try to read the hidden
  evaluator's files or otherwise obtain test labels.
- You may fit parameters on `data/witb_train.json` (its true placements are
  visible) and ship them with your predictor. The hidden test does not
  retrain anything.

## Compliance checklist

- [ ] My method is a `method/predictors/<name>.sh` script that reads INPUT_JSON and writes OUTPUT_JSON.
- [ ] It produces a valid distribution for every scenario and object.
- [ ] It runs offline, deterministically, within 2 hours on one CPU.
- [ ] I did not modify evaluation code, reference scores or the visible data.
