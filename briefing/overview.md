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
