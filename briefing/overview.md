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
