# Experiment Overview

## witb_human_alignment

Tests whether a system can produce hidden-object location probabilities that
track human judgments across a set of two-box household-object scenarios. The
experiment also includes single-cue reference outputs so the benchmark measures
whether using multiple cue types improves alignment rather than simply solving a
unimodal version of the task. The reported gain metric summarizes whether the
main system exceeds the better of the two single-cue references.

The same experiment also reports a secondary diagnostic comparing the main
system against a simple average of the audio-only and vision-only predictions.
This checks whether a multi-cue system adds value beyond post-hoc averaging of
single-cue probability judgments.

A neutral constant-probability diagnostic is included to show that the scaled
benchmark has room below the paper ablations. It is not compared to a paper
number because the paper did not report this baseline.
