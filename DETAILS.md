# Overview

- **Paper ID:** 2506.14212
- **Title:** What's in the Box? Reasoning about Unseen Objects from Multimodal Cues
- **Domain:** Cognitive Science / Multimodal Reasoning
- **TL;DR:** The paper models human-like guesses about hidden objects by parsing multimodal cues and integrating them with Bayesian inference over object placements.

## Short Summary

The paper introduces the What's-in-the-Box task, where people and models infer
which listed objects are hidden in which of two boxes after seeing and hearing
the boxes handled. Its model uses neural parsers to turn object names, box
images and audio into structured evidence, then performs Bayesian inference
over all object-placement hypotheses. The full model correlates much more
strongly with human judgments than audio-only, vision-only or Gemini 2.0 Flash
baselines.

## Key Results (paper, correlation with mean human ratings, 20 stimuli)

- Full model: `r = 0.78`.
- Audio-only `r = 0.55`, vision-only `r = 0.52`.
- Gemini 2.0 Flash `r = 0.31`.

## This gym

The original videos and ratings are not public. The gym scores methods on 2000
hidden scenarios from a simulated What's-in-the-Box world against the objects'
true placements: correlation (primary), Brier score and log loss. Each method
is a predictor script that the evaluator runs on the hidden inputs. The
paper's full model reaches correlation 0.802, Brier 0.089 and log loss 0.346;
its audio-only, vision-only and averaged ablations score lower.

---

# Problem Definition

## Research Question

Several household objects are hidden in two closed boxes. An observer is told
which objects there are, sees the boxes, and hears them being shaken. How can
a system turn these indirect, noisy cues into a probability, for every object,
of which box it is in?

## Why It Matters

People routinely reason about things they cannot see by combining what they
know about objects, what they see and what they hear. A system that does this
well, with well-calibrated probabilities, is a model of that human ability and
a building block for robots that must reason about occluded objects.

## Inputs (per scenario)

The cues have already been parsed into structured form, the way
language, vision and audio models would parse them:

- **Objects** (2-5 per scenario): name, and a language-model estimate of their
  dimensions in cm (mean and standard deviation per dimension), weight in kg
  (mean and standard deviation), material, and rigidity (0 = fully squashable,
  1 = rigid). These estimates come from the object's name alone, so they are
  generic for that kind of object and can be biased or over-confident.
- **Boxes** (2 per scenario, `left` and `right`): a vision-model estimate of
  the inner dimensions in cm (mean and standard deviation) and an estimate of
  how heavy the box is when it is lifted and shaken (`lifted_weight_kg`, total
  of box and contents, noisy).
- **Sound**: for each box, a zero-shot sound classifier's posterior over the
  scenario's object names, computed from the sound of that box being shaken.
  Sounds of several objects mix, quiet objects can be masked by loud ones,
  tightly packed boxes rattle less, and the classifier is imperfect.

## Output

For every object, a probability distribution over the two boxes.

## Success Criteria

Probabilities that track where the objects really are and are calibrated, on
scenarios the system has never seen. Every box holds at least one object, and
every object is in exactly one box.

## Data

The original study's videos and participant ratings are not public. The gym
uses a simulated What's-in-the-Box world with household objects, box sizes and
cue noise modelled on the original study. 240 labelled scenarios are visible
in `data/witb_train.json`; scoring uses 2000 hidden scenarios from the same
world.

---

# Paper's Method

## Key Contribution

A neurosymbolic model of the What's-in-the-Box game. Neural models parse the
object names (an LLM: dimensions with uncertainty, weight, material,
rigidity), the first video frame (Gemini: box dimensions) and the sound of
each box being shaken (CLAP: posterior over the scenario's object names). A
Bayesian engine then integrates these cues over explicit object-placement
hypotheses.

## Algorithm (implemented in `method/witb_bayes.py`)

1. Enumerate every assignment of the N objects to the K boxes that leaves no
   box empty (|H| = K!·S(N, K)); uniform prior over them.
2. Visual evidence P(O_i | H_i): draw object and box dimensions 1000 times
   from the parsed normal distributions and count how often the hypothesised
   objects physically fit in box i (rejection sampling).
3. Audio evidence P(H_i | A_i) = Π_{o∈H_i} P(o | A_i) from the per-box
   classifier posterior (paper Eq. 2).
4. P(H | O, A) ∝ Π_i P(O_i | H_i) P(H_i | A_i) (paper Eq. 1, cues treated as
   conditionally independent and equally weighted); normalise and
   marginalise into per-object box probabilities.

Implementation choices the paper does not specify: the fit check is an
axis-aligned guillotine packer that tries all six orientations of each object
(at most eight free regions); sampled dimensions are clamped at 0.1 cm; one set
of 1000 draws per scenario is shared by all hypotheses; the random stream is
seeded by (seed 17, scenario id). The parsed weight, rigidity, material and
lifted-weight fields are not used, as in the paper's equations.

The predictor `method/predictors/full_neurosymbolic.sh` runs this model with
1000 samples and seed 17.

## Paper ablations (in `baseline/witb_baselines.py`)

- Audio-only (paper Eq. 3): P(o in box i) = P(o|A_i) / Σ_j P(o|A_j).
- Vision-only: the same hypothesis inference with visual evidence only.
- Late-fusion average of the two (not a paper baseline; tests whether joint
  inference adds more than averaging).

## Main Claims

The full model correlates strongly with mean human ratings (r = 0.78), better
than audio-only (0.55), vision-only (0.52) and Gemini 2.0 Flash (0.31).
Joint inference over hypotheses is not a simple average of the unimodal
outputs. Limitations the paper names: equal cue weights (it suggests learning
cue reliabilities), no use of how heavy a box looks when handled, and an audio
model that misses quiet sounds or sounds mixed with others.

## In this gym

On the 2000 hidden simulated scenarios, the full model reaches correlation
0.802 with the true placements. Audio-only reaches 0.770, vision-only 0.454
and the late-fusion average 0.764. It is also better on Brier score than all
three; on log loss audio-only is level with it (0.340 vs 0.346).

---

# Evaluation

## How a method is run

A method is an executable shell script `method/predictors/<name>.sh`. The
evaluators call every such script, as

```
bash method/predictors/<name>.sh INPUT_JSON OUTPUT_JSON
```

with the workspace root as working directory. Every script on disk is
scored under its file name (`<name>`).

- `INPUT_JSON` has `{"scenarios": [...]}` with the fields described in
  `briefing/problem.md` and `data/README.md`, but **without** `true_box`.
- `OUTPUT_JSON` must be
  `{"scenarios": [{"id": "<scenario id>", "box_probs": {"<object name>": [p_left, p_right], ...}}, ...]}`.
  Every scenario id and every object name must be present; each list has one
  finite probability per box in [0, 1] (floating-point slack of 1e-6 is
  clipped) and sums to 1 (tolerance 1e-3). Anything else in the file is
  ignored.
- The script runs offline on one CPU core and must finish within 2 hours.
  It may use code and files in the workspace (for example parameters fitted
  on the train split).

A script that crashes, times out, or writes an incomplete or invalid output
**fails**. A failed method receives the worst possible value of every scored
metric (correlation −1, Brier 1, log loss −ln 1e-4 ≈ 9.21).

## Splits

- **Train (visible):** `data/witb_train.json`, 240 scenarios (881 objects)
  with their true placements. `bash scripts/evaluate_train.sh` (no arguments)
  runs every predictor on them and writes `scoring/scores_train.json`.
  `bash scripts/evaluate.sh` does the same and writes `scoring/scores.json`.
- **Test (hidden):** 2000 scenarios (7075 objects) from the same simulated
  world, held only by the hidden evaluator. `bash scripts/evaluate_test.sh`
  (no arguments) runs every predictor on them and writes
  `scoring/scores_test.json`. Predictors get the unlabelled inputs in a
  temporary directory; the evaluator's own directory is unreadable while they
  run. Every method sees exactly the same test inputs.

## Metrics (experiment `witb_placement_inference`)

All metrics are recomputed by the evaluator from `box_probs` and the true
placements.

| metric | better | coefficient | meaning |
|---|---|---:|---|
| `truth_correlation` (primary) | higher | 1.0 | Pearson correlation over all (object, box) queries between the predicted probability and whether the object is in that box |
| `brier_score` | lower | 1.0 | mean over objects of ½ Σ_box (p − 1[true box])² |
| `log_loss` | lower | 0.5 | mean over objects of −ln max(p_true box, 1e-4) |

Correlation measures how well a method ranks likely and unlikely placements
(the statistic the original study reports). The two proper scoring rules also
require the probabilities to be calibrated, so a method cannot gain by
sharpening or flattening its outputs.

## Reference and comparison numbers

`scoring/scores_test.json` holds the reference method's hidden-test scores
(correlation 0.802, Brier 0.0893, log loss 0.346). Each score file also lists,
under `_baselines`, fixed comparison predictions stored with the evaluators
(single-cue systems, an average of the two single-cue systems, and chance).
They are not ranked.

| comparison (hidden test) | correlation | Brier | log loss |
|---|---:|---:|---:|
| sound cue only | 0.770 | 0.104 | 0.340 |
| size cue only | 0.454 | 0.203 | 0.655 |
| average of the two single-cue outputs | 0.764 | 0.119 | 0.394 |
| chance (0.5 / 0.5) | 0.000 | 0.250 | 0.693 |

The original study reports correlations with human ratings (0.78 for its
model, 0.55 sound only, 0.52 vision only, 0.31 for a large vision-language
model). Those are on a different target and are kept in
`scoring/reference.json` under `_paper_reported`, not as numbers to match.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 12:55] - method_runs
- Read the paper and identified the actual algorithm: structured multimodal parsing followed by Bayesian inference over object-placement hypotheses.
- Implemented the paper's hypothesis enumeration, visual rejection sampling, audio posterior products, posterior normalization, and marginalization.
- Added audio-only and vision-only ablations and a method-agnostic evaluator.
- Ran a reduced-sample smoke test end to end; full model output, ablation outputs, and `scoring/scores.json` were produced successfully.

### [2026-06-02 12:58] - core_claim
- CPU job `abff144f-319` completed with exit code 0 inside the container.
- The configured 1000-sample reproduction produced `human_correlation = 0.983` for the full method.
- The full method beat both unimodal ablations on the same scaled benchmark: audio-only `0.804`, vision-only `0.678`.
- `python validate.py --compare` passed in the job, including scoring structure and separation checks.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 12:58] - method_runs
- Implemented the paper's Bayesian hypothesis enumeration, visual rejection-sampling fit check, audio evidence product, posterior normalization, and marginalization on a scaled parsed WiTB benchmark.
- Added scripts for the full method, unimodal ablations, and method-agnostic evaluation.

### [2026-06-02 13:00] - core_claim
- Ran the scaled benchmark with 1000 visual samples per box hypothesis.
- The full neurosymbolic model beat both unimodal ablations on human-correlation, supporting the paper's central multimodal-integration claim in the packaged environment.

### [2026-06-02 13:03] - core_claim
- Verified the corrected audio-only Eq. 3 ablation and multimodal-gain scoring in CPU container job `2029a1ef-afb`.
- `validate.py --compare` passed with full `r = 0.983`, audio-only `r = 0.796`, vision-only `r = 0.678`, and multimodal gain `0.187`.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `joint_gain_over_late_fusion = 0.044`.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `joint_gain_over_late_fusion = 0.044`.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `joint_gain_over_late_fusion = 0.044`.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `joint_gain_over_late_fusion = 0.044`.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `joint_gain_over_late_fusion = 0.044`.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `gain_over_late_fusion = 0.044`.

### [2026-06-02 13:28] - secondary_claims
- Tightened the visual likelihood to match the paper's described rejection-sampling hit rate without extra smoothing or undocumented dimension shrinkage.
- Expanded `scores.json` to include all declared calibration and within-scenario ranking metrics.
- Local validation passed with full multi-cue inference at `human_correlation = 0.980`, `multimodal_gain_over_best_unimodal = 0.184`, and `gain_over_late_fusion = 0.055`.

### [2026-06-02 13:40] - secondary_claims
- Final verification confirmed method/data/evaluation separation, standalone scoring, train/test consistency, and no committed `shared/` dependencies.
- Made final method-agnostic documentation and metric-name cleanup for the late-fusion comparison.
- `python validate.py --compare` passes after the final verification edits.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `gain_over_late_fusion = 0.044`.

### [2026-06-02 13:28] - secondary_claims
- Tightened the visual likelihood to match the paper's described rejection-sampling hit rate without extra smoothing or undocumented dimension shrinkage.
- Expanded `scores.json` to include all declared calibration and within-scenario ranking metrics.
- Local validation passed with full multi-cue inference at `human_correlation = 0.980`, `multimodal_gain_over_best_unimodal = 0.184`, and `gain_over_late_fusion = 0.055`.

### [2026-06-02 13:40] - secondary_claims
- Final verification confirmed method/data/evaluation separation, standalone scoring, train/test consistency, and no committed `shared/` dependencies.
- Made final method-agnostic documentation and metric-name cleanup for the late-fusion comparison.
- `python validate.py --compare` passes after the final verification edits.

### [2026-06-02 13:52] - secondary_claims
- Fixed a validator failure where split-slice unimodal diagnostics were being compared against full-paper baseline correlations.
- Train/test score files now expose the paper-comparable proposed method result and keep ablation artifacts only as diagnostics for gain metrics.
- `python validate.py` passes after regenerating `scores_train.json` and `scores_test.json`.

### [2026-06-02 13:59] - secondary_claims
- Added a neutral uniform-probability diagnostic baseline to document that the scaled parsed benchmark is non-trivial.
- Hardened the main evaluator against duplicate scenarios, missing objects, and non-finite or out-of-range probabilities.
- Re-ran `scripts/reproduce.sh`, split evaluations, and `python validate.py --compare`; all checks passed locally.

### [2026-06-02 14:05] - secondary_claims
- Final verification kept the `secondary_claims` milestone.
- Made train/test split scoring reusable for future non-baseline method artifact names.
- Confirmed `python validate.py --compare` passes and documented the remaining limitation: the benchmark is a small parsed substitute for unreleased WiTB data.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 13:04] - core_claim
- Verified the paper's Bayesian hypothesis algorithm runs end-to-end through `scripts/reproduce.sh`.
- Prior compute job `2029a1ef-afb` completed successfully with `SAMPLES=1000` and `python validate.py --compare` passing.
- Full neurosymbolic model reached `human_correlation = 0.983` on the scaled parsed benchmark, beating audio-only `0.796` and vision-only `0.678`.

### [2026-06-02 13:07] - core_claim
- Re-audited the continuation workspace after milestone reset.
- Confirmed the implementation is the paper's explicit Bayesian hypothesis algorithm, evaluation remains method-independent, and `scripts/evaluate.sh` works standalone.
- No public original WiTB code or dataset release was found, so `core_claim` remains the highest justified milestone for the packaged scaled benchmark.

### [2026-06-02 13:10] - secondary_claims
- Job `57d422b4-403` completed with exit code 0 and `python validate.py --compare` passing in the container.
- Added and verified a late-fusion diagnostic for the paper's secondary claim that joint hypothesis inference is not simply averaging unimodal outputs.
- Full joint inference reached `human_correlation = 0.983`; late fusion reached `0.940`, giving `gain_over_late_fusion = 0.044`.

### [2026-06-02 13:28] - secondary_claims
- Tightened the visual likelihood to match the paper's described rejection-sampling hit rate without extra smoothing or undocumented dimension shrinkage.
- Expanded `scores.json` to include all declared calibration and within-scenario ranking metrics.
- Local validation passed with full multi-cue inference at `human_correlation = 0.980`, `multimodal_gain_over_best_unimodal = 0.184`, and `gain_over_late_fusion = 0.055`.

### [2026-06-02 13:40] - secondary_claims
- Final verification confirmed method/data/evaluation separation, standalone scoring, train/test consistency, and no committed `shared/` dependencies.
- Made final method-agnostic documentation and metric-name cleanup for the late-fusion comparison.
- `python validate.py --compare` passes after the final verification edits.

### [2026-06-02 13:52] - secondary_claims
- Fixed a validator failure where split-slice unimodal diagnostics were being compared against full-paper baseline correlations.
- Train/test score files now expose the paper-comparable proposed method result and keep ablation artifacts only as diagnostics for gain metrics.
- `python validate.py` passes after regenerating `scores_train.json` and `scores_test.json`.

### [2026-06-02 13:59] - secondary_claims
- Added a neutral uniform-probability diagnostic baseline to document that the scaled parsed benchmark is non-trivial.
- Hardened the main evaluator against duplicate scenarios, missing objects, and non-finite or out-of-range probabilities.
- Re-ran `scripts/reproduce.sh`, split evaluations, and `python validate.py --compare`; all checks passed locally.

### [2026-06-02 14:05] - secondary_claims
- Final verification kept the `secondary_claims` milestone.
- Made train/test split scoring reusable for future non-baseline method artifact names.
- Confirmed `python validate.py --compare` passes and documented the remaining limitation: the benchmark is a small parsed substitute for unreleased WiTB data.

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
