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
