> **Superseded (iteration 11).** This note describes the earlier five-scenario gym. See `HARDENING.md` for the current design.

# Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side.
  The train script invokes `eval/train/evaluate_witb_train.py` and writes
  `scoring/scores_train.json`; the test script invokes
  `eval/test/evaluate_witb_test.py` and writes `scoring/scores_test.json`.
- Grepped for train/test cross-imports, hardcoded `METHOD_NAME` defaults, and
  paper-agreement metric names. No cross-imports or method-name defaults were
  found in the split evaluators.
- Confirmed the split evaluators score method-authored prediction artifacts from
  `outputs/*_predictions.json` rather than importing from `method/`.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not
  exist at handoff.
- Ran `bash -n` on both split entrypoint scripts.

## Changed

- Moved the held-out test slice away from the paper's explicitly discussed
  qualitative scenario B. Train now scores scenarios A, B, and C; test scores
  scenarios D and E.
- Added `brier_score` to both split evaluators to penalize overconfident
  probability errors that correlation can hide.
- Added `within_scenario_rank_accuracy` to both split evaluators to catch
  methods that match global trends while getting scenario-local object ordering
  wrong.
- Added schema/range validation in both split evaluators for duplicate scenario
  IDs, missing object probabilities, and non-finite or out-of-range probability
  values.
- Updated `scoring/reference.json` so every active metric has
  `higher_is_better`, `coefficient`, and `description`. The error metrics now
  use negative coefficients, matching the intended scoring direction.

## Runtime Concerns For Next Iteration

- The benchmark has only five parsed scenarios, so even the improved 3/2 split
  is small. The runner should watch for saturation, especially on
  `within_scenario_rank_accuracy`, which has few pairwise comparisons.
- `data/witb_parsed_benchmark.json` contains human target probabilities for all
  scenarios. A future agent with access to the full data file could shortcut by
  copying targets into prediction artifacts. This pass was limited to
  eval/script/reference edits, so I did not move or mask test targets. If the
  benchmark protocol allows a data change later, the held-out target labels
  should be separated from the visible data.
- I did not run either evaluator or generate scores, per the source-only
  instructions. The next runner should verify that all prediction artifacts emit
  the new metrics on both slices and that the full reproduction does not produce
  all-perfect or all-zero train/test scores.
- The prompt's paper path was stale; the local paper copy used for context was
  `paper/2506.14212.md`.
