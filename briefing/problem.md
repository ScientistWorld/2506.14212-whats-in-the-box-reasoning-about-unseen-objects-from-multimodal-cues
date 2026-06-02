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
