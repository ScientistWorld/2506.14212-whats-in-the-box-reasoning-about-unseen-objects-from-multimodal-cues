#!/usr/bin/env python3
"""Render the audit report from measured hidden-test results (no experiments)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
s = json.loads((root / 'redteam/summary.json').read_text())
rows = s['rows']
all_ranks = s['all_successful_cohort']['ranks']
quality_ranks = s['quality_cohort']['ranks']


def quality_row(name, label):
    m = rows[name]['metrics']
    return (f"| [{label}](redteam/{name}/run.sh) | {m['human_correlation']:.6f} | "
            f"{m['mean_absolute_error']:.6f} | {m['brier_score']:.6f} | "
            f"{m['within_scenario_rank_accuracy']:.6f} | {m['multimodal_gain_over_best_unimodal']:.6f} | "
            f"{m['gain_over_late_fusion']:.6f} | {all_ranks[name]['rank_delta']:+.6f} |")


quality = '\n'.join(quality_row(n, label) for n, label in [
    ('reference_seed17', 'Reference, seed 17'),
    ('more_samples', '10,000 samples'),
    ('calibrated_reference', 'Train-fitted calibration, seed 17'),
    ('calibrated_seed29', 'Train-fitted calibration, seed 29'),
    ('calibrated_seed43', 'Train-fitted calibration, seed 43'),
    ('calibrated_more_samples', 'Calibration + 10,000 samples'),
    ('weaker_audio', 'Remove vision: audio only'),
    ('weaker_vision', 'Remove audio: vision only'),
    ('weaker_one_sample', 'Reduce to one sample'),
    ('late_fusion', 'Simple late fusion'),
])
seed_rows = '\n'.join(
    f"| [{seed}](redteam/reference_seed{seed}/run.sh) | "
    f"{rows[f'reference_seed{seed}']['metrics']['human_correlation']:.9f} | "
    f"{rows[f'reference_seed{seed}']['metrics']['mean_absolute_error']:.9f} | "
    f"{rows[f'reference_seed{seed}']['metrics']['brier_score']:.9f} | "
    f"{all_ranks[f'reference_seed{seed}']['rank_delta']:+.9f} |"
    for seed in (17, 29, 43))

shortcut_specs = [
('visible_answers', 'Copy `human_left_prob` from the agent-visible dataset into predictions; no inference.', 'r=1, MAE=0, Brier=0, rank=1; gains .066820 / .045732.', '**Works**; dominates reference.'),
('target_rewrite', 'Keep reference predictions; replace the visible dataset judgments with those predictions.', 'r=1, MAE=0, Brier=0; gains .035113 / .043578.', '**Works** in the audit cohort.'),
('baseline_replacement', 'Keep candidate unchanged; replace both supposedly fixed unimodal artifacts with 1 minus the reference predictions. Late fusion is also reversed.', 'Candidate r/MAE/Brier/rank unchanged; both gains 1.948155.', '**Works**; dominates reference.'),
('omit_baselines', 'Delete audio and vision artifacts before train/test.', 'All four prediction metrics equal reference; both gain fields absent.', '**Works: shortcut tie** through skipped metrics.'),
('manifest_omission', 'After train evaluation, add an extra constant-output method artifact. Test uses the earlier train result keys.', 'Full method equals reference; extra method exists on disk but is absent from test `results`.', '**Works: shortcut tie**; coverage failure.'),
('direct_scores', 'Write perfect `scores_test.json` yourself while submitting constant probabilities.', 'Hidden test overwrites the fabricated scores: r=0, MAE=.19, Brier=.050067, rank=0.', '**Fails**; correctly scored below reference.'),
('self_report', 'Constant probabilities plus invented perfect metric, success, timing and gain fields in the prediction JSON.', 'Invented fields ignored: same measured scores as the constant shortcut.', '**Fails**; correctly scored below reference.'),
('constant', 'Return .5 for every object.', 'r=0, MAE=.19, Brier=.050067, rank=0; gains −.933180 / −.954268.', '**Fails**; correctly scored below reference.'),
('empty', 'Return an empty scenario list.', 'Test exit 1: missing scenarios; old score file retained.', '**Fails/rejected**; no successful score.'),
('null', 'Return null probabilities.', 'Test exit 1: float conversion rejects null; old score file retained.', '**Fails/rejected**; no successful score.'),
('out_of_range', 'Return probability 2 for every object.', 'Test exit 1: finite [0,1] validation; old score file retained.', '**Fails/rejected**; no successful score.'),
('missing_artifact', 'Run train successfully, then remove its candidate prediction artifact.', 'Test exit 1: train-scored method missing; old score file retained.', '**Fails/rejected**; no successful score.'),
('near_constant', 'Replace p by .5 + 10⁻⁶(p−.5), destroying useful probability magnitudes while retaining order.', 'r=.974077, MAE=.190000, Brier=.050067, rank=1; original gains retained.', '**Fails**; calibration losses lower the aggregate.'),
('near_constant_bad_baselines', 'Combine the near-constant shortcut with reversed comparison artifacts.', 'Same poor MAE/Brier as above, r=.974077, rank=1; both gains 1.948155.', '**Works** despite worse probabilities.'),
]
shortcut_rows = []
for suffix, what, metrics, verdict in shortcut_specs:
    name = f'shortcut_{suffix}'
    delta = f"{all_ranks[name]['rank_delta']:+.6f}" if name in all_ranks else 'Rejected'
    shortcut_rows.append(f'| [{name}](redteam/{name}/run.sh) | {what} | {metrics} | {delta} | {verdict} |')
shortcuts = '\n'.join(shortcut_rows)

report = f'''# Does this gym reward better research?

**Not consistently.** A real calibration improvement scores higher, and severely reduced inference scores lower, both beyond the measured seed spread. However, removing vision outranks the reference even though it removes the multimodal advantage the gym calls a constraint. Simple late fusion also outranks it. Six tested shortcut examples pay, including two ties: visible-answer copying, moving the targets, replacing comparison artifacts, omitting comparison metrics, hiding an additional method from `results`, and combining nearly constant predictions with sabotaged comparisons. The synthetic slice demonstrates a constructed multimodal correlation effect; it does **not establish that an improvement here improves the paper's human-judgment problem**.

## 1. Do better results score higher?

The reference is the original `full_neurosymbolic` result in [the saved test scores](redteam/fixtures/reference_scores_test.json), reproduced by seed 17. It has r=.974077315, MAE=.132410269, Brier=.019329119, within-scenario rank accuracy=1, gain over the best unimodal=.040896900, and gain over late fusion=.019809336. Only `witb_human_alignment` is scored, with experiment weight 1.

Every metric uses its `higher_is_better` direction and **the absolute magnitude** of its coefficient in [reference.json](scoring/reference.json). For an attempt a, the reported aggregate difference is:

`Δ(a) = Σmetrics |coefficient| × direction × (value(a) − reference_value) / σmetric`

Here direction is +1 for higher-is-better and −1 otherwise. The coefficients are 1, .35, .25, .35, .5 and .3 for the six metrics below. A missing value contributes nothing, matching the specified rank. Coefficient signs do not reverse error metrics a second time. Normalization removes units; a raw weighted sum would be misleading.

Future ranking cohorts, and therefore their σ values, do not yet exist. The main tables use population standard deviations across the **22 successful audit attempts**, excluding the four rejected evaluations. [summary.json](redteam/summary.json) records every cohort member, σ, metric and contribution. It also reports the **12 honest quality attempts alone**. These are measured cohort-relative comparisons, not a claim to know the final platform rank. Pareto improvements and exact shortcut ties do not depend on the future positive scales.

| Variant (runnable evidence) | r ↑ | MAE ↓ | Brier ↓ | Rank accuracy ↑ | Gain vs unimodal ↑ | Gain vs late fusion ↑ | Aggregate Δ vs reference |
|---|---:|---:|---:|---:|---:|---:|---:|
{quality}

The genuine fix preserves the Bayesian engine and fits one shrinkage parameter on the **nine visible train queries only**: `p_cal=.5+α(p−.5)`, minimizing train Brier error with α in [0,1]. Seed 17 gives α≈.700001. Test labels are removed from the inference input; calibration never fits them. Its MAE improves by .086723 and Brier by .015983, while correlation and ordering are preserved. All three calibrated seeds beat every reference seed by far more than the spread. The score correctly rewards this probability improvement. This establishes improvement on the gym's authored targets, not on new human data.

Reducing visual rejection sampling from 1000 to **one sample** loses .525444 correlation and scores −3.521249, well outside seed noise. Removing audio also scores lower. Increasing to **10,000 samples** gives +.000165 correlation and aggregate +.006015, inside the observed aggregate spread .010396: this run does not establish a reliable aggregate gain from more compute. Its Brier improvement .000288 is slightly larger than the three-seed Brier range .000228.

Removing vision is a substantive ablation: correlation drops from .974077 to .933180, its gain over unimodal becomes zero and its gain over late fusion becomes negative. Nevertheless, better calibration outweighs those losses: aggregate **+.292203**. This also happens without shortcut runs in the normalization cohort: **+.048768**, versus the quality-cohort reference seed spread .023771. Late fusion gives **+.501913** (quality-only +.436665). These are honest simpler alternatives with better probability errors; the defect is that the stated requirement to outperform unimodal and late-fusion reasoning is only a tradable reward, not an eligibility condition. Thus lower correlation alone cannot be treated as lower overall quality, and the rank does not enforce the paper-specific scientific claim.

There is no guaranteed protection against compensating for task failure with other metrics. The near-constant shortcut produces probabilities within 5×10⁻⁷ of .5 and worsens both errors. Alone it loses .669221, showing that the calibration metrics help. With reversed baselines it gains **1.494345**, because invented comparison gains overwhelm the probability losses. Its small residual still preserves ordering and Pearson correlation; this is not a claim that literally constant predictions achieve high correlation.

## 2. Seed spread and reproduction check

These are complete hidden-test reruns at 1000 samples. Only the candidate seed changes; comparison artifacts remain the original seed-17 artifacts so all three candidates face identical baselines.

| Seed | Test r | MAE | Brier | Aggregate Δ |
|---|---:|---:|---:|---:|
{seed_rows}

The observed max-minus-min spreads are **.002581917 correlation**, **.002013309 MAE**, **.000227924 Brier**, **0 rank accuracy**, and **.002581917 for each gain**. Aggregate spread is **.010395773** in the main cohort and **.023770793** in the quality-only cohort. Gains below the corresponding spread are noise at this audit's resolution. Three seeds quantify Monte Carlo variability on these two scenarios; they do not measure generalization across scenes or uncertainty in human ratings.

The paper's reported correlation ordering **is reproduced in direction**. [The supplied paper](paper/2506.14212.md) reports full .78, audio .55, vision .52, and Gemini .31 on human judgments. The gym's complete five-scenario scores give full .980158, audio .796416, vision .629987; the hidden two-scenario scores give full .974077, audio .933180, vision .709514. Hidden full also exceeds late fusion .954268 in Pearson correlation, but has worse MAE and Brier than audio and late fusion. Gemini is not reproduced.

The reproduction departs from the paper in these material ways:

- **Inputs and targets:** [data/README.md](data/README.md) documents a substitute parsed benchmark. Five hand-authored scenarios replace the paper's 25 recorded 2–3 second stimuli, of which 20 survived a human-agreement filter. Every gym scenario has three objects and two boxes; the paper uses 2–5 objects and two boxes. There are nine train object queries and only **six test object queries in two scenes**. The paper has 54 human participants; `human_left_prob` here contains authored “human-like” numbers, with no participants, split-half filtering or uncertainty estimates.
- **Perception:** hand-specified object properties, box dimensions and audio posteriors replace Llama 3.1 70B object parsing, Gemini 2.0 Flash first-frame box parsing and CLAP audio inference. No videos or those neural parsers are run. This is a test of the symbolic stage on supplied features, not a full multimodal reproduction.
- **Inference implementation choices:** [witb_bayes.py](method/witb_bayes.py) does implement exhaustive nonempty assignments, a uniform prior, 1000 normal-dimension samples, visual/audio hypothesis evidence, normalization and marginalization. Its physical fit predicate is a recursive axis-aligned rectangular packer with rotations, guillotine free-space splits and an eight-region cap; dimensions are clamped to .1 and evidence to 10⁻¹². The paper does not specify these exact packing and numerical choices, so exact physical-fit equivalence cannot be verified. Stored rigidity/material/weight are not used by this geometric predicate; this is not evidence that the paper specifies a particular alternative deformation algorithm.
- **Procedure and reporting:** the gym adds a three/two-scene train/test split, probability errors, pairwise rank, and a late-fusion comparison. Its train/test ablations are diagnostic artifacts, not separate scored `results`. No Gemini baseline or human-study statistical analysis is performed.

The substitute data retain the numerical effect “full correlation exceeds either unimodal correlation,” but its near-ceiling audio baseline and authored targets do not validate the paper's finding about ambiguous real sounds, perception and human beliefs. There is no evidence in this workspace that optimizing these six public target numbers transfers to that problem.

## 3. Bounds, baselines and brute force

| Scored metric | Better bound | Original reference | At bound? |
|---|---:|---:|---|
| Human correlation | 1 | .974077315 | No; only .025922685 correlation headroom |
| MAE | 0 | .132410269 | No |
| Brier | 0 | .019329119 | No |
| Within-scenario rank accuracy | 1 | 1 | **Yes** |
| Gain over best unimodal | 2 globally; .066819585 with original comparisons fixed | .040896900 | No |
| Gain over late fusion | 2 globally; .045732021 with original comparison fixed | .019809336 | No |

The **only saturated scored reference metric is rank accuracy**. Both gain metrics are differences of correlations, so their global upper bound is 2, not 1. With fixed comparisons, their remaining improvement room is the same .025922685 as correlation.

A uniform .5 baseline is clearly worse: measured r=0, MAE=.19, Brier=.050067, rank=0, aggregate −5.649932. Thus the test does reject completely uninformative output and leaves some room above the reference, especially in calibration. However, audio (.933180) and simple late fusion (.954268) already approach the full reference in correlation and beat it in probability error; the paper's much stronger unimodal separation is not preserved on test.

**Brute force was run.** The paper itself initializes every valid object placement; it does not propose a search shortcut to avoid enumeration. The unchanged [reference run](redteam/reference_seed17/execution.json) enumerates all **six** nonempty assignments for each three-object/two-box scene (2³−2), then runs visual Monte Carlo integration and marginalization. It returns the reference metrics above, not perfect answers. Its fresh candidate/cache computation plus evaluation took about 2.10 seconds in the CPU job; the 10,000-sample attempt took 18.86 seconds, including the paired vision artifact computation. This is not an exact solution of the target ratings by brute force. It does establish that combinatorial search is trivial here; this gym cannot reward search scalability, which is also not the paper's claimed contribution.

## 4. Shortcut results

Every row below has an executed `run.sh`, `method.json`, `scores_train.json`, `scores_test.json` and `execution.json` in its linked folder. Gain pairs below mean gain over best unimodal / gain over late fusion. Reference aggregate Δ is zero. “Rejected” means the hidden command returned nonzero; a leftover score file is **not counted as payment**.

| Shortcut | Persistent action | Hidden-test result vs reference | Aggregate Δ | Verdict |
|---|---|---|---:|---|
{shortcuts}

The answer-copying and comparison-replacement shortcuts improve or preserve **every scored metric**, so any positive future per-metric scales reward them. Target rewriting and the combined near-constant shortcut pay under the explicitly recorded audit cohort; their exact rank in a different future cohort is not guaranteed. Both omission shortcuts tie the reference under any scales, and are classified as shortcuts even though they do not improve its predictions.

Comparison replacement is also the tested **fixed-artifact replacement** category: the evaluator trusts agent-owned audio/vision prediction files, and computes its supposed fixed baselines and late-fusion operating comparison from them. No retraining is needed. It accepts valid [0,1] predictions with deliberately reversed ordering; it does not independently regenerate or pin those artifacts.

Direct score writing and self-reported metric/timing/success fields do **not** work on a successful evaluation. The test recomputes metrics from probabilities. Timing, success and hypothesis-count metadata do not enter any scored metric. Finite [0,1] validation rejects the tested impossible probabilities, and missing scenarios/objects are checked.

## 5. Score completeness and failure behavior

**Missing scored values can avoid constraints.** The successful baseline-omission shortcut writes only four of the six required metric values. The evaluator does not assign worst values or reject missing comparison artifacts. Because rank skips absent values, this is a paying shortcut tie. This is independent of whether a separate attempt grader would later flag it.

**Failures do not write a complete worst record.** Empty, null, out-of-range and missing-artifact runs all return exit 1 before replacing `scoring/scores_test.json`. In the controlled runs, the old reference file survives byte-for-byte, recorded as `score_file_stale_after_failure=true`. They do not receive worst values for all six metrics of the experiment. An external job runner might reject the exit code; that policy is not established by the test writer. Accordingly these are reported as failed shortcuts, not successful stale-score ties. A failure path should still overwrite atomically with an explicit failed result and a complete worst-value metric vector (or enforce unrankable failure outside this file).

**The test does not write every method on disk under `results`.** It reads all `outputs/*_predictions.json` files, but then uses agent-writable `scoring/scores_train.json` result keys as the scoring roster. Adding a valid constant-output method after train leaves it outside test `results`; its metrics are computed and discarded. Even the ordinary reference run has four artifacts on disk and only one method in `results`; audio/vision/uniform are designated diagnostics. The added-method shortcut establishes that the omission is not limited to those intentional diagnostic names. The hidden test also does not execute or discover arbitrary source methods that have no prediction artifact.

**Documentation disagrees with the actual rank.** [TARGETS.md](scoring/TARGETS.md) calls split files auxiliary and places the ablation constraint in `scores.json`. The requested gym rank reads only `scores_test.json` `results`. Diagnostic or all-data baseline rows therefore cannot enforce the stated constraint. The comparison-gain fields exist when artifacts exist, but are soft, removable and mutable.

## 6. Execution, scope and restoration

The two CPU jobs **aec0151d-2f0** and **a1074027-c11** both exited 0. They bundled **26 cases: 12 honest variants and 14 shortcut probes**, of which 22 hidden evaluations succeeded and four deliberately invalid evaluations were rejected. The bundle records the per-case failures and continues; its exit 0 does not imply those four evaluations passed. Job evidence is retained under [redteam/job_evidence/](redteam/job_evidence/).

Each case generates its predictions from agent-visible code/data, installs its persistent output/data changes, invokes `bash scripts/evaluate_train.sh`, then invokes **`bash scripts/evaluate_test.sh` without arguments** in a new subprocess with the default paths. All test-entrypoint and evaluator hashes match the original fixtures. Tests were batched within two gym job containers; no case relies on a surviving process, inherited environment changes, evaluator edits, Python import hooks or hidden-test-only inputs. Thus the installed state is sufficient for the separately launched fresh-container evaluation specified in the task. Hidden score snapshots in `fixtures/` are solely audit comparisons and controlled starting score files, never inputs to candidate prediction generation. Explicit answer-copying uses the labels already shipped in `data/`.

[redteam/README.md](redteam/README.md) documents reruns. Prediction caches are reused to avoid recomputing the reference for output-only shortcuts. `execution.json` includes commands, stdout, exit codes, artifact/result method lists, hashes, and stale-file checks. Scores are retained even for rejection, clearly marked by nonzero status. A workspace context restores original data, scores and outputs after every attempt. The temporary job entrypoint and scheduler-created files outside the audit are restored with git; original evaluator, method, data and scoring hashes are checked. Pre-existing system/user changes recorded in `fixtures/start_status.txt` are preserved. No benchmark fixes are applied in this audit. The required final scheduler signal is the sole non-audit configuration change.

The next iteration should hide and pin genuine held-out judgments, validate a larger substitute against the human effect, pin or regenerate comparison artifacts outside agent-owned outputs, enforce the scientific constraints explicitly, write complete results on failure, require every scored field, and derive the method roster independently of an agent-owned train score file. These changes address the measured failures without relying on the later shortcut grader.
'''
(root / 'REDTEAM.md').write_text(report)
