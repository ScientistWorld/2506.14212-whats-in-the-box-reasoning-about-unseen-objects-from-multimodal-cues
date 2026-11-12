# Experiment Overview

## witb_placement_inference

Establishes how well a system infers where unseen objects are, given parsed
object descriptions, box measurements, how heavy each box looks when shaken,
and per-box sound-classifier posteriors. Each system is run on 2000 hidden
two-box scenarios (2-5 objects each, 7075 objects in all) that it has never
seen. Its probabilities are compared with the objects' true placements.

- `truth_correlation` (primary) asks whether the system's graded judgments
  track where objects really are, the same correlation statistic the original
  study uses against human ratings.
- `brier_score` and `log_loss` ask whether those judgments are calibrated
  probabilities. They guard against improving the ranking by distorting the
  confidence of the outputs.

A system that fails to produce a valid, complete distribution for every
hidden object is scored at the worst value on all three metrics.

Fixed single-cue systems, their average and chance are reported for
comparison (`_baselines`). They show how much of the task a single kind of
cue solves and where integrating cues helps.
