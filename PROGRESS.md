# Progress

## Current state (iteration 11, hardening)

- Benchmark: a simulated What's-in-the-Box world (`eval/test/witb_world.py`,
  hidden). 240 labelled train scenarios are visible in `data/witb_train.json`;
  2000 hidden test scenarios (7075 objects) are in `eval/test/`. Targets are
  true object placements. See `data/README.md`.
- Method: `method/witb_bayes.py` implements the paper's Bayesian engine. It
  enumerates the K!·S(N,K) placements, uses a uniform prior, estimates visual
  fit by 1000-sample rejection sampling, uses Eq. 2 audio evidence, and
  normalises and marginalises. It runs as the predictor
  `method/predictors/full_neurosymbolic.sh`.
- Baselines: `baseline/witb_baselines.py` has audio-only (Eq. 3), vision-only,
  the late-fusion average and chance. Fixed copies are in
  `eval/{train,test}/fixed_baselines_*.json`.
- Evaluation: `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` take
  no arguments and run every `method/predictors/*.sh`. Metrics are
  truth_correlation, brier_score and log_loss; a failure scores worst on all
  three.

## Results (hidden test, 2000 scenarios)

| system | r | Brier | log loss |
|---|---:|---:|---:|
| full model (paper) | 0.802 | 0.0893 | 0.346 |
| audio-only (Eq. 3) | 0.770 | 0.1038 | 0.340 |
| late-fusion average | 0.764 | 0.1191 | 0.394 |
| vision-only | 0.454 | 0.2029 | 0.655 |
| chance | 0 | 0.25 | 0.693 |

The paper reports r = 0.78 / 0.55 / 0.52 against human ratings (not
comparable). `HARDENING.md` has the variant and shortcut audit.

## Deviations from the paper

- Data and target: a simulated world with true placements, instead of 20
  unreleased videos and the mean ratings of 54 participants.
- Neural parsers: the LLM, Gemini and CLAP outputs are simulated, not run;
  there is no Gemini baseline.
- The fit predicate is an axis-aligned guillotine packer (the paper does not
  specify one).
