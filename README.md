# Overview

- **Paper ID:** 2506.14212
- **Title:** What's in the Box? Reasoning about Unseen Objects from Multimodal Cues
- **Domain:** Cognitive Science / Multimodal Reasoning
- **TL;DR:** The paper models human-like guesses about hidden objects by parsing multimodal cues and integrating them with Bayesian inference over object placements.

## Short Summary

The paper introduces the What-is-in-the-Box task, where people and models infer
which listed objects are hidden in boxes after seeing and hearing the boxes
handled. Its model uses neural parsers to convert object names, box images, and
audio into structured evidence, then performs Bayesian inference over all
object-placement hypotheses. The reported full model correlates much more
strongly with human judgments than audio-only, vision-only, or Gemini 2.0 Flash
baselines.

## Key Results

- Full model human-correlation: `r = 0.78`.
- Unimodal ablations are lower: audio-only `r = 0.55`, vision-only `r = 0.52`.
- Gemini 2.0 Flash baseline is much lower: `r = 0.31`.

---

# Problem Definition

## Research Question

How can a system infer which unseen household objects are inside which boxes
from incomplete multimodal evidence: a list of candidate objects, visible box
sizes, and sounds produced when the boxes are handled?

## Why It Matters

Everyday physical reasoning often requires judging hidden states from indirect
cues. A good solution would support robots and AI systems that can reason about
occluded objects under uncertainty instead of relying only on direct visual
recognition.

## Success Criteria

A successful system outputs calibrated object-by-box probability judgments that
align with human judgments and improves over systems that rely on only one
modality. The evaluation should preserve the same object lists, boxes, and cue
inputs across methods.

---

# Paper's Method

## Key Contribution

The paper proposes a neurosymbolic model for hidden-object reasoning in the
What-is-in-the-Box game. Neural perception modules parse language, vision, and
audio into structured evidence, then a Bayesian symbolic engine integrates the
evidence over explicit object-placement hypotheses.

## Approach

The language parser estimates uncertain object physical attributes such as size
and rigidity from object names. The vision parser estimates box dimensions from
the first video frame. The audio parser uses CLAP with scenario object names as
candidate labels to produce per-box object probabilities.

The reasoning engine enumerates all non-empty assignments of objects to boxes,
uses a uniform prior, estimates visual likelihoods by rejection sampling whether
sampled object dimensions fit sampled box dimensions, multiplies audio
posteriors for objects assigned to each box, normalizes the posterior over
hypotheses, and marginalizes to per-object box probabilities.

## Main Claims

The primary claim is that multimodal Bayesian integration aligns strongly with
human probability judgments and outperforms unimodal ablations. The paper
reports `r = 0.78` for the full model, compared with `r = 0.55` for audio-only,
`r = 0.52` for vision-only, and `r = 0.31` for Gemini 2.0 Flash.

Secondary claims are that geometry helps for size-constrained objects, audio
helps for distinctive sounds, and joint hypothesis-level inference is more
informative than simply averaging unimodal outputs.

---

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


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
