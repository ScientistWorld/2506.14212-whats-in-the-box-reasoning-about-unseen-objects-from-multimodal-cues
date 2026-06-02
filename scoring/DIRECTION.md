# Research Direction

## Research Problem

The research problem is probabilistic reasoning about unseen object placements
from indirect multimodal cues. Improvements should address how a system turns
ambiguous cue evidence into object-location probability judgments.

## Core Contribution

The paper's contribution is structured multimodal cue integration for hidden
object reasoning. Work should remain in this problem family: combining object
identity priors, visible box constraints, and per-box auditory evidence to infer
which objects are hidden where.

## Approach Scope

Scientists may improve cue estimates, uncertainty calibration, physical
compatibility judgments, or multi-cue probability estimation as long as the
output is a distribution over object placements for the same WiTB-style inputs.

## Out of Bounds

Directly fitting a generic predictor to the benchmark target ratings is out of
scope because it bypasses the hidden-state reasoning problem. Changing the task
to open-ended object recognition, using extra evaluation labels, or reporting a
different benchmark under the same experiment name is also out of bounds.
