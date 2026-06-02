# Progress

## What Works

- Paper analysis is complete in `notes/paper_analysis.md` and `briefing/`.
- A scaled parsed WiTB benchmark is tracked in `data/witb_parsed_benchmark.json`.
- `method/witb_bayes.py` implements the paper's symbolic inference algorithm: non-empty placement enumeration, visual fit rejection sampling, audio evidence scoring, posterior normalization, and marginalization.
- `baseline/witb_baselines.py` runs the paper's audio-only Eq. 3 ablation and the vision-only geometric ablation.
- `scripts/reproduce.sh` writes prediction artifacts and `scripts/evaluate.sh` scores them without importing from `method/`.
- `python validate.py --compare` passes on the current `outputs/` prediction artifacts.

## Results

Corrected compute validation (`SAMPLES=1000`, job `2029a1ef-afb`, exit code 0, `python validate.py --compare` passed):

- Full neurosymbolic: `human_correlation = 0.983`
- Audio-only ablation: `human_correlation = 0.796`
- Vision-only ablation: `human_correlation = 0.678`
- Multimodal gain over best unimodal: `0.187`

The full method beats both unimodal ablations, supporting the paper's core
claim on the scaled parsed benchmark.

Continuation audit on 2026-06-02 verified that `scripts/evaluate.sh` works
standalone from `outputs/`, `scripts/download.sh` and `environment/setup.sh`
complete successfully, and each three-object/two-box scenario enumerates the
paper-expected 6 non-empty placement hypotheses. A fresh web search found paper
pages but no public code or full dataset release.

Secondary-claim extension added a late-fusion diagnostic computed only from
audio-only and vision-only prediction artifacts. Job `57d422b4-403` verified the
path in the container with exit code 0 and `python validate.py --compare`
passing. Late fusion reached `human_correlation = 0.940`, while full joint
inference reached `0.983`, yielding `joint_gain_over_late_fusion = 0.044`.

Previous compute job `abff144f-319` completed successfully before the audio-only
baseline was corrected to the paper's Eq. 3 formula. Job `2029a1ef-afb`
verified the corrected code path in the container.

Earlier reduced-sample smoke test (`SAMPLES=50`) on the scaled parsed benchmark:

- Full neurosymbolic: `human_correlation = 0.954`
- Audio-only ablation: `human_correlation = 0.804`
- Vision-only ablation: `human_correlation = 0.696`

Paper reported on its unreleased analysis set:

- Full neurosymbolic: `r = 0.78`
- Audio-only ablation: `r = 0.55`
- Vision-only ablation: `r = 0.52`
- Gemini 2.0 Flash baseline: `r = 0.31`

## Remaining

- Higher milestones would require more of the original paper's evaluation surface: additional scenarios, a direct Gemini-style VLM baseline, or recovered original stimuli/human ratings.
- The current environment is packaged and validated through the secondary late-fusion claim.

## Issues

- The original 25 WiTB stimuli, parsed neural outputs, and human participant ratings are not released in the paper. Exact numerical reproduction of Figure 3 is therefore not possible from the paper alone.
- Gemini 2.0 Flash and Llama 3.1 70B are hosted or very large systems; this scaled reproduction uses tracked structured parser outputs instead of calling unavailable APIs on compute nodes.
- Compute logs for the corrected run are available under `jobs/2029a1ef-afb/`.

## Deviations from Paper

- Dataset: the paper used 20 analyzed stimuli from a new 25-stimulus human/video dataset. Because no full dataset download is provided, this environment uses a small tracked parsed benchmark with WiTB-style household-object scenarios and curated human-like target ratings.
- Neural parsers: the paper used Llama 3.1 70B, Gemini 2.0 Flash, and CLAP to produce structured object, box, and audio evidence. This environment stores structured parser outputs directly in `data/witb_parsed_benchmark.json` so compute jobs require no internet or paid APIs.
- Algorithm: the Bayesian inference procedure itself is implemented directly rather than replaced. The scaled benchmark reduces data/resource requirements only.
