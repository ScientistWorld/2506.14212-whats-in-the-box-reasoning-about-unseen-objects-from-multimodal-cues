# Hardening audit (iteration 11)

`python3 redteam/hardening/audit.py all` (run by `scripts/run.sh`) does the
following:

1. Fits the better variants' parameters on the visible train split only
   (`variants/fit.py` writes `variants/fitted_params.json`).
2. Installs the audit predictors (`predictors/*.sh`) next to the reference
   predictor in `method/predictors/`, runs `bash scripts/evaluate_train.sh` and
   `bash scripts/evaluate_test.sh` (no arguments), and saves the score files to
   `results/`. Every method sees the same 2000 hidden scenarios, so all
   comparisons are paired.
3. Runs the state-mutating shortcuts (`sc_direct_scores`,
   `sc_rewrite_visible`) in a separate group.
4. Runs a late-registration check: train evaluation first, then a new
   predictor added before the test evaluation.
5. Restores the workspace state and deletes the audit predictors.
6. Analyses the runs: per-method metrics, aggregate rank difference vs the
   reference (|coefficient| × improvement / population σ across the honest
   cohort), and a paired scenario bootstrap (1000 resamples) for every method
   vs the reference (`results/summary.json`, `results/summary.md`).

`results/extra_paired.json` has paired bootstraps between selected variants.
`results_run1_800/` is the first audit run, on an 800-scenario test, before
two fixes: floating-point slack in probability validation, and a
regularised stacker.

`variants/witb_variants.py` holds every variant and shortcut. Variants import
the agent-visible `method/witb_bayes.py`; none reads hidden data.
