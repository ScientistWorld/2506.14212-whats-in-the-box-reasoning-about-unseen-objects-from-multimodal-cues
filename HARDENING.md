# Hardening report (iteration 11)

## Summary

The red-team audit found that the old gym could not reward better research.
The two hidden test scenarios had six object queries whose target ratings
were hand-written and visible in `data/`. The comparison baselines were
agent-owned files, and missing values and failed runs cost nothing. The paper
gives no public data to fall back on, so patching that evaluator would still
have left a test of agreement with six authored numbers. I rebuilt the
benchmark and the evaluators. The paper's algorithm is unchanged.

- **Data.** A simulated What's-in-the-Box world (`eval/test/witb_world.py`,
  hidden). It has 50 household object types, 2-5 objects in two boxes as in
  the paper, and usually one large and one small box, as in the paper's
  stimuli. Cues are imperfect in the ways the paper describes: biased,
  over-confident per-type size parses; noisy box measurements; a simulated
  zero-shot sound classifier on the mixed sound of each box's real contents;
  and a noisy "how heavy does the box look" estimate. Ground truth is the
  actual placement.
  - 240 labelled scenarios are visible (`data/witb_train.json`).
  - 2000 hidden scenarios (7075 objects) exist only in
    `eval/test/witb_test_hidden.json`.
- **Methods are predictors run on hidden inputs.** A method is
  `method/predictors/<name>.sh INPUT OUTPUT`.
  - `scripts/evaluate_test.sh` (no arguments) runs every predictor on disk on
    the hidden, unlabelled inputs. It writes them to a temporary directory and
    makes `eval/test/` unreadable while predictors run.
  - It recomputes every metric from the raw `box_probs`.
- **Metrics.** Against the true placements:
  - `truth_correlation` (primary): the paper's correlation statistic, with
    truth as the target.
  - `brier_score` and `log_loss`: proper scoring rules, so calibration counts.
  - Every method gets every metric. A crash, timeout, missing scenario or
    object, or invalid or unnormalised probability scores the worst value on
    all three. Those worst values are written before any predictor runs.
- **Comparisons are fixed by `eval/`.** Paper ablations and chance are stored
  prediction files in `eval/train/` and `eval/test/`, reported under
  `_baselines`, not ranked.
- **Paper numbers.** Kept in `scoring/reference.json` under `_paper_reported`;
  they are on a different target.

All numbers below come from the final job (`3a4a3ee9-40a`), via
`bash scripts/evaluate_test.sh` with no arguments, on the 2000 hidden
scenarios. Aggregate Δ is the rank formula:
Σ |coefficient| × (improvement in the better direction) / σ, with σ the
population standard deviation over the 13 honest runs below. Coefficients are
1.0 for correlation, 1.0 for Brier and 0.5 for log loss. CIs are 95% paired
scenario-bootstrap intervals (1000 resamples) of the difference from the
reference (`redteam/hardening/results/summary.json`).

## 1. Do better results score higher?

**Yes, now.** Before, a genuine calibration fix scored higher (+0.63 to
+0.66), but so did removing vision (+0.29) and simple late fusion (+0.50).

| run (hidden test) | r ↑ | Brier ↓ | log loss ↓ | train r | aggregate Δ | paired 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **reference: paper's full model** (1000 samples, seed 17) | 0.8022 | 0.0893 | 0.3460 | 0.7985 | 0 | |
| better: + lifted-weight cue, fitted cue weights (train-fitted) | 0.8905 | 0.0518 | 0.1693 | 0.8837 | **+2.18** | [+1.95, +2.43] |
| better: train-fitted cue weights + visual-fit floor | 0.8258 | 0.0795 | 0.2593 | 0.8169 | **+0.69** | [+0.50, +0.88] |
| better: train-fitted visual-fit floor only | 0.8244 | 0.0805 | 0.2644 | 0.8152 | **+0.64** | [+0.46, +0.81] |
| learned late fusion (3-parameter logistic on the two ablations, train-fitted) | 0.8209 | 0.0815 | 0.2679 | 0.8174 | +0.58 | [+0.38, +0.76] |
| more compute: 5000 samples | 0.8057 | 0.0878 | 0.3294 | 0.8093 | +0.11 | [+0.03, +0.20] |
| reference, seed 29 | 0.8036 | 0.0887 | 0.3432 | 0.8003 | +0.03 | [−0.06, +0.13] |
| reference, seed 43 | 0.8007 | 0.0899 | 0.3483 | 0.7952 | −0.03 | [−0.14, +0.07] |
| weaker: audio only (paper Eq. 3) | 0.7695 | 0.1038 | 0.3401 | 0.7534 | −0.62 | [−0.89, −0.32] |
| weaker: late-fusion average | 0.7637 | 0.1191 | 0.3940 | 0.7645 | −1.19 | [−1.37, −1.00] |
| weaker: lifted-weight cue only | 0.5890 | 0.1633 | 0.4722 | 0.5963 | −3.99 | [−4.35, −3.61] |
| weaker: one visual sample | 0.6175 | 0.1606 | 0.8445 | 0.5885 | −4.75 | [−5.22, −4.28] |
| weaker: vision only | 0.4545 | 0.2029 | 0.6546 | 0.4726 | −6.63 | [−6.92, −6.32] |

- **Seed noise.** Spread across seeds 17/29/43 is 0.0028 r, 0.0012 Brier,
  0.0051 log loss and **0.067 aggregate**.
  - The three better variants beat the reference by 10-30× that spread, with
    paired CIs far from zero.
  - Every weaker variant scores lower, also beyond noise.
- **Better variants keep the paper's engine.** They keep its enumeration,
  rejection-sampled fit and audio evidence. Using only the visible train
  labels, they fit three things:
  - cue exponents and a small floor on the visual-fit probability: the paper's
    own future direction of learning cue reliabilities;
  - a likelihood for the lifted-weight cue: the paper's own stated missing cue;
  - nothing on test.
- **Train and test agree.** Every honest run scores within about 0.02 r of its
  train score.
- **Paper's claim, in its direction.**
  - The full model beats audio-only on correlation and Brier score. Paired
    aggregate CI is [+0.32, +0.87]; on log loss the two are level.
  - It beats the late-fusion average (CI [+1.00, +1.37]) and vision-only on
    every metric.
  - Joint inference with learned cue weights beats the learned late fusion
    (CI [+0.035, +0.194], better on all three metrics;
    `results/extra_paired.json`). So the gym rewards joint inference even when
    both sides are calibrated.
- **Before / after variant scores.** Before, aggregates were in the old
  audit's σ units on 6 authored queries:
  - calibrated variants +0.63 to +0.66;
  - 10k samples +0.006;
  - audio-only **+0.29** and simple late fusion **+0.50** (both outranked the
    reference);
  - vision-only −1.35, one sample −3.52.
  After: the table above. Audio-only and the late-fusion average are now
  below the reference.

## 2. Shortcuts: fix and score before → after

| shortcut (old audit) | fix | before (old aggregate Δ) | after (hidden test) |
|---|---|---|---|
| copy visible answers (`human_left_prob` in `data/`) | test labels and inputs exist only in `eval/test/`; ids are opaque; train labels are for other scenarios | +1.04, dominates | `sc_copy_train_labels`: matched 0/2000 ids → r 0, Brier 0.25, log loss 0.693, **−11.76**. It is perfect on train only. |
| rewrite the target ratings in `data/` | test targets loaded from `eval/test/` before predictors run | +1.01 | `sc_rewrite_visible` (rewrites every train label and plants reversed baseline files, submits the reference's own predictions): **ties the reference (0.00)**; the rewrites change nothing |
| replace the comparison artifacts (reversed audio/vision files) | no scored metric depends on any agent artifact; comparisons are fixed files in `eval/` and are not ranked | +2.16, dominates | same `sc_rewrite_visible` run: planted reversed baselines ignored; score = its own predictions = reference |
| omit comparison artifacts → gain metrics missing | every method gets all three metrics; worst values pre-written for every predictor; failure ⇒ worst everywhere | 0 (tie through skipped metrics) | not possible: no optional metric. Failing runs: `sc_crash`, `sc_missing_half`, `sc_nan`, `sc_out_of_range`, `sc_unnormalized` all get (−1, 1, 9.21), **−63.1** |
| method added after train evaluation is left out of test `results` | test discovers every `method/predictors/*.sh` itself; it no longer reads `scores_train.json` | 0 (tie) | `sc_late_added`: present in test `results` with its own score (chance, −11.76) |
| near-constant probabilities + sabotaged baselines | baselines not scored; calibration metrics penalise flattening | +1.49 | `sc_near_constant`: r unchanged (0.802) but Brier 0.25, log loss 0.693, **−4.92** |
| empty / null / out-of-range / missing artifact left a stale score file | worst values written first; failure scored worst on every metric; atomic writes | rejected, stale file kept | worst on every metric (rows above) |
| write `scores_test.json` directly | evaluator overwrites the file after recomputing | −5.65 (failed) | `sc_direct_scores`: still overwritten → its real (chance) score, −11.76 |
| self-reported metrics/status fields | only `box_probs` is read | failed | `sc_self_report`: identical to the predictions it copies (reference) |
| *new*: read hidden labels from `eval/test/` at test time | inputs handed over in a temp dir; `eval/test/` made unreadable while predictors run | n/a | `sc_read_hidden`: "Permission denied", matched 0/2000 → chance, −11.76 |
| *new*: memorise train labels, otherwise reference | test scenarios are unseen | n/a | `sc_train_memorizer`: train r = 1.0, test = reference (0.00): train-tuned behaviour gains nothing |
| *new*: overconfident rounding to 0/1 | proper scoring rules | n/a | `sc_extreme`: −3.16 |

## 3. Shortcuts left open, and why

- **Deliberately undoing the sandbox.** The predictor runs as the same user as
  the evaluator. A predictor that `chmod`s `eval/test/` back, or reads the
  evaluator's memory or parent process, could reach the hidden labels. The
  block is best-effort; this is a deliberate attack the separate shortcut
  grader must catch.
- **Learning quirks of the simulated world.** The world's per-type quirks are
  fixed across splits: size-parse biases, sound-classifier label biases,
  loudness, and the tare of cardboard boxes. A method can learn them from the
  240 labelled train scenarios. That is honest learning (it transfers to the
  hidden test), but part of such a gain would not transfer to the real WiTB
  stimuli. The generator itself is hidden and has per-type latent
  sound/text-embedding parameters, so it cannot be copied. A
  physically-motivated model of a cue, such as `better_weight_cue`, is the kind
  of improvement the paper itself proposes.
- **More Monte Carlo samples.** 5000 samples instead of 1000 gives +0.11 (CI
  [+0.03, +0.20]): a more precise estimate of the paper's own visual
  likelihood. That is 1.7× the seed spread but less than a fifth of the
  smallest scientific improvement. It is not a brute force the paper avoids,
  so it is left.
- **Learned late fusion outranks the training-free reference** (+0.58). It is
  not one of the paper's baselines and not a loophole. It fits 3 parameters
  (bias, audio and vision weights) on the train labels, which the paper's
  model never uses. Its gain is the cue-reliability calibration the paper
  names as missing. It cannot be removed without changing the paper's method.
  The gym still ranks joint inference with the same kind of learned weighting
  above it (paired CI [+0.035, +0.194]). The untrained paper baselines are all
  clearly below the reference.

## 4. Reference score before and after

- **Before:** r = 0.974, MAE 0.132, Brier 0.0193, rank accuracy 1.0. That was
  on 6 hand-authored "human-like" ratings in 2 visible scenarios.
- **After:** r = **0.8022**, Brier **0.0893**, log loss **0.3460**. This is
  the paper's full model on 2000 hidden scenarios against true placements,
  computed from scratch (no cache) in 66 s on one CPU.
- Train split: 0.7985 / 0.0908 / 0.3778.

The two are different targets, so the drop is not a regression. The paper
reports r = 0.78 against mean human ratings; that number is kept under
`_paper_reported`.

## 5. Headroom check

- **No scored metric is at its bound** (r ≤ 1, Brier ≥ 0, log loss ≥ 0). The
  hidden placements are uncertain given the cues, so the bound is out of
  reach. The best variant reaches 0.891 / 0.052 / 0.169.
- **Where the room comes from.** Room above the reference comes from the
  paper's own stated shortfalls: equal cue weights, the unused box-weight cue,
  and audio that misses mixed and quiet sounds. The lifted-weight cue was
  added to the inputs for this reason. The paper's method ignores it, and
  using it is worth +2.18.
- **No simple variant reaches the bound.** None of the paper's untrained
  baselines (audio-only, vision-only, the late-fusion average) outranks the
  reference.
- **Brute force.** The paper's method already enumerates every hypothesis
  (≤ 30 for 5 objects in 2 boxes); there is no cheaper search for a brute
  force to beat. More sampling of the paper's own estimator gains little
  (above).
- **Private search.** The objective needs the hidden placements, so a
  predictor cannot compute it on its own copy of public code. A predictor
  never sees the test labels, and the test inputs only at evaluation time.
- **No extra experiment.** The single experiment has headroom, so none was
  added. The weight stays 1.0.

## 6. Weights

There is one experiment (weight 1.0). Metric coefficients:

- correlation **1.0**: the paper's statistic;
- Brier **1.0**: proper, bounded, and protects calibration;
- log loss **0.5**: proper, but heavy-tailed: a single confident miss costs up
  to 9.2, so it is down-weighted.

The old gain-over-unimodal and gain-over-late-fusion metrics were removed.
With baselines fixed by `eval/`, they are the candidate's own correlation
shifted by a constant, so they would double-count it. The old
within-scenario rank accuracy was removed because the reference sat at its
bound (1.0).

## 7. What could not be fixed

- **No real data.** The paper's videos, parsed outputs and 54 participants'
  ratings are not public (searched again this iteration). The gym's targets
  are true placements in a simulated world, not human judgments. The paper's
  claim is reproduced in direction on that world (full > audio-only,
  vision-only and averaging), not on human data. Whether a better method here
  is more human-like is argued, not measured: the paper's own error analysis
  says humans are more confident and use more cues than the model.
- **No neural parsers.** Llama 3.1 70B, Gemini 2.0 Flash and CLAP are not run.
  Their outputs are simulated, so the gym tests the symbolic integration stage
  on parsed cues. The Gemini baseline (r = 0.31) is not reproduced.
- **Sandbox is best-effort** (section 3).

## Files

- **Reproduce:** `bash scripts/run.sh` runs fixed baselines, the reproduction,
  this audit and the reference-only scores; `scripts/reproduce.sh` covers the
  visible-data part only.
- **Audit:** `redteam/hardening/`, with `results/` and the first run on an
  800-scenario test in `results_run1_800/`. The first run exposed two
  problems, both fixed before the final run:
  - a valid probability of 1.0000000000000002 was rejected; the evaluators now
    allow 1e-6 of floating-point slack;
  - the late-fusion stacker's fit diverged; it is now regularised.
- **Old audit:** `redteam/*` (superseded; see `redteam/README.md`).
