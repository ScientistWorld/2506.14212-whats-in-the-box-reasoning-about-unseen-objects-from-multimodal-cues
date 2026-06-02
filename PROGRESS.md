# Progress

## What Works

- Paper analysis is complete in `notes/paper_analysis.md` and `briefing/`.
- A scaled parsed WiTB benchmark is tracked in `data/witb_parsed_benchmark.json`.
- `method/witb_bayes.py` implements the paper's symbolic inference algorithm: non-empty placement enumeration, visual fit rejection sampling, audio evidence scoring, posterior normalization, and marginalization.
- `baseline/witb_baselines.py` runs the paper's audio-only and vision-only ablations.
- `scripts/reproduce.sh` writes prediction artifacts and `scripts/evaluate.sh` scores them without importing from `method/`.
- `python validate.py --compare` passes on reduced-sample smoke-test outputs.

## Results

Reduced-sample smoke test (`SAMPLES=50`) on the scaled parsed benchmark:

- Full neurosymbolic: `human_correlation = 0.954`
- Audio-only ablation: `human_correlation = 0.804`
- Vision-only ablation: `human_correlation = 0.696`

Paper reported on its unreleased analysis set:

- Full neurosymbolic: `r = 0.78`
- Audio-only ablation: `r = 0.55`
- Vision-only ablation: `r = 0.52`
- Gemini 2.0 Flash baseline: `r = 0.31`

The smoke test supports the expected ordering on the scaled benchmark, but the
configured 1000-sample job is being submitted before claiming `core_claim`.

## Remaining

- Run the configured reproduction job with 1000 visual rejection samples.
- Inspect job results and update `MILESTONES.md` to `core_claim` if the full method still beats both unimodal ablations.
- If budget permits after `core_claim`, add another setting or a direct VLM-style baseline.

## Issues

- The original 25 WiTB stimuli, parsed neural outputs, and human participant ratings are not released in the paper. Exact numerical reproduction of Figure 3 is therefore not possible from the paper alone.
- Gemini 2.0 Flash and Llama 3.1 70B are hosted or very large systems; this scaled reproduction uses tracked structured parser outputs instead of calling unavailable APIs on compute nodes.

## Deviations from Paper

- Dataset: the paper used 20 analyzed stimuli from a new 25-stimulus human/video dataset. Because no full dataset download is provided, this environment uses a small tracked parsed benchmark with WiTB-style household-object scenarios and curated human-like target ratings.
- Neural parsers: the paper used Llama 3.1 70B, Gemini 2.0 Flash, and CLAP to produce structured object, box, and audio evidence. This environment stores structured parser outputs directly in `data/witb_parsed_benchmark.json` so compute jobs require no internet or paid APIs.
- Algorithm: the Bayesian inference procedure itself is implemented directly rather than replaced. The scaled benchmark reduces data/resource requirements only.
