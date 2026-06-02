# Paper Analysis: "What's in the Box? Reasoning about Unseen Objects from Multimodal Cues"

## Actual Method

The paper proposes a neurosymbolic model for the WiTB object-placement task. The method is not an end-to-end learned predictor. It has two stages:

1. Neural parsing of multimodal inputs into structured attributes.
2. Bayesian inference over explicit hypotheses of object placements.

Inputs are a list of hidden object names, a short video with audio of an experimenter shaking two boxes, and visual information about box sizes. The model builds structured representations as follows:

- Language parser: Llama 3.1 70B is prompted with each object name to estimate physical attributes, including dimensions, weight, material, rigidity, and standard deviations for uncertain quantities.
- Vision parser: Gemini 2.0 Flash is prompted on the first video frame to return box dimensions in a structured JSON-like format.
- Audio parser: CLAP is run on audio segments for each shaken box, using scenario object names as candidate labels, producing per-box probabilities over object labels.

The symbolic inference engine enumerates all hypotheses assigning the N objects across K boxes, with no empty boxes implied by the paper's count `K! S(N,K)`. Each hypothesis is an ordered tuple of object sets, one set per box. The prior over hypotheses is uniform.

For each hypothesis, the model scores:

- Visual/geometric likelihood: for each box, sample object and box dimensions from normal distributions and use rejection sampling to estimate the probability that the hypothesized object set can fit in that box. The paper uses 1000 samples.
- Audio score: for each box hypothesis, multiply CLAP posterior probabilities for every object assigned to that box: `prod_o P(o | audio_box_i)`.

The posterior is proportional to the product across boxes of visual fit probabilities and audio object probabilities. The model then normalizes across hypotheses and marginalizes the posterior to get per-object probabilities for each box.

Faithful reproduction must implement this hypothesis enumeration, visual rejection sampling, CLAP-based object-label audio scoring, posterior normalization, and marginalization. A generic video/audio model that directly predicts object placement would be a surrogate and would not reproduce the paper's method.

## Experiment Protocol

The authors created 25 WiTB stimuli. Each stimulus contains 2 to 5 household objects hidden inside 2 boxes. Objects vary in shape, size, and material, including examples such as water bottles, mugs, plates, laptops, yoga mats, pillows, water jugs, and coins. Each stimulus has a 2 to 3 second video of a human experimenter shaking the boxes.

Human evaluation:

- 54 Prolific participants.
- Participants saw the object list and watched the video with sound.
- For each object, participants rated the likelihood that it was in each box on continuous scales from 1 to 100.
- The UI enforced that ratings for an object summed to 100 across boxes.
- 5 stimuli were excluded because human split-half reliability was below 0.8, leaving 20 stimuli for reported analysis.

Model comparisons:

- Full model: language object attributes + vision box dimensions + CLAP audio labels + Bayesian hypothesis inference.
- Audio-only ablation: CLAP probabilities only, normalized across boxes for each object as `P(o in box_i | A_i) / sum_j P(o in box_j | A_j)`.
- Vision-only ablation: geometric fitting only, using object and box physical attributes.
- Gemini 2.0 Flash baseline: VLM receives the video with audio and the same task instructions as humans, directly producing probability distributions.

Primary analysis correlates model ratings against mean human probability judgments across all object-box queries. The plots report Pearson correlation `r`, with standard errors and 95% confidence intervals shown visually.

## Metrics and Reported Numbers

Primary benefit metric:

- Pearson correlation between model probability ratings and averaged human judgments.

Reported correlations:

- Full neurosymbolic model: `r = 0.78`.
- Audio-only ablation: `r = 0.55`.
- Vision-only ablation: `r = 0.52`.
- Gemini 2.0 Flash baseline: `r = 0.31`.

Important constraint metric for a benchmark:

- The method must outperform unimodal ablations while using the same object lists and box/audio observations.
- A faithful benchmark should report the full model's correlation plus at least the audio-only and vision-only correlations. Otherwise future agents could optimize human-correlation by ignoring the paper's multimodal integration claim.

Secondary qualitative claims:

- Full model is not a simple average of audio-only and vision-only outputs; joint inference over object-placement hypotheses changes marginal object probabilities.
- Vision helps with objects constrained by box size, such as a yoga mat or water jug.
- Audio helps with distinctive sound makers, such as coins.
- CLAP can miss subtle mixed sounds, and the current vision module does not use dynamic cues such as inferred weight from box motion.

## Dependencies and Data Availability Risks

Data availability is the largest risk. The paper describes a novel 25-stimulus WiTB dataset but does not provide a dataset download link in the paper text. Two qualitative YouTube links are given:

- Scenario A: `https://youtu.be/JE4ggHKfRss`
- Scenario B: `https://youtu.be/TdZHEkuGDgM`

The full set of videos, object lists, box labels, participant ratings, excluded-stimulus IDs, and exact train/eval-free protocol may not be publicly available from the paper alone. Without the original human ratings, exact reproduction of the reported correlations is not possible. A benchmark can still reproduce the method on a newly constructed or recovered subset, but should mark the data limitation clearly.

External model risks:

- Llama 3.1 70B may require large model weights and GPU memory, or hosted inference access. A scaled reproduction can replace hosted access with cached structured object-attribute files generated before compute jobs, but the parser role should remain LLM-based.
- Gemini 2.0 Flash is an API model and may be inaccessible, updated over time, or unavailable on compute nodes without internet. It affects both vision parsing and the VLM baseline.
- CLAP weights must be downloaded before job submission because compute nodes have no internet.
- YouTube video download must happen before submission and may fail if videos are removed, blocked, or rate limited.

Implementation risks:

- The paper does not specify exact prompts for Llama/Gemini, exact dimension units, box-coordinate conventions, object orientation assumptions, or the geometric packing algorithm used in rejection sampling.
- The visual likelihood requires deciding how to test whether multiple sampled 3D objects can fit in a box. A faithful scaled version should use a documented 3D rectangular-bin approximation with sampled dimensions and rotations, rather than substituting a learned predictor.
- CLAP posterior calibration can differ by model checkpoint and prompt label formatting.

## Scaled but Faithful Reproduction Plan

A scaled reproduction should preserve the paper's procedure and reduce only the number of stimuli, objects, or parser size where necessary.

Minimum faithful implementation:

1. Build a small WiTB-style dataset with 2-box scenarios, each containing 2 to 5 named household objects, per-box shaking audio/video, and human or proxy target ratings. If original ratings are unavailable, collect new ratings or use a clearly labeled synthetic/human-small subset.
2. Extract or annotate first-frame box dimensions for each scenario. If Gemini API is unavailable, precompute box dimensions manually or with a local vision model, but keep the output as the same structured box-attribute input to the Bayesian engine.
3. Generate object physical attributes from an LLM or from cached LLM outputs: dimensions, standard deviations, weight/material/rigidity if used. At minimum, dimensions and uncertainty are required for the visual likelihood.
4. Segment audio by box shaking event and run CLAP with the scenario's object names as labels to get `P(object | audio_box)`.
5. Enumerate all non-empty assignments of objects to the two boxes.
6. Score each hypothesis using visual rejection sampling and audio posterior products.
7. Normalize the posterior over hypotheses and marginalize to object-by-box probabilities.
8. Implement audio-only and vision-only ablations with the paper's formulas.
9. Evaluate Pearson correlation against human mean ratings and report full, audio-only, and vision-only scores.

Recommended benchmark scores:

- `full_human_corr`: Pearson r for the full model.
- `audio_only_human_corr`: Pearson r for the CLAP-only ablation.
- `vision_only_human_corr`: Pearson r for the geometry-only ablation.
- `multimodal_gain_over_best_unimodal`: full correlation minus max(audio-only, vision-only). The paper reference value is `0.23`.
- `hypothesis_validity`: fraction of evaluated examples where posterior mass is normalized and all per-object box probabilities sum to 1.

Milestone guidance:

- `method_runs`: Bayesian enumeration, visual rejection sampling, CLAP scoring, posterior marginalization, and at least one ablation run on any valid scenario.
- `core_claim`: full model evaluated on multiple scenarios with human or curated target judgments, showing better human-correlation than both unimodal ablations.
- `core_claim_plus`: include Gemini/VLM baseline or additional qualitative scenarios matching the paper's examples.
