# Research Direction

## Research Problem

Probabilistic reasoning about unseen object placements from indirect,
imperfect cues: what each object is like (from its name), how big the boxes
are, how heavy they look, and what they sound like. Improvements should make
the object-in-box probabilities more accurate and better calibrated on unseen
scenarios.

## Core Contribution

Structured integration of several cue types into one coherent judgment about
where everything is. A good method should respect that each object is in
exactly one box, that boxes have limited room, and that cues differ in how
reliable they are.

## Approach Scope

In scope: better use of each cue (for example, how far to trust a size or
sound estimate, how sounds of several objects mix, squashable objects, how
heavy a box looks), better joint reasoning over placements, and calibration
learned from the visible train split.

## Out of Bounds

Obtaining hidden-test labels or inputs in any way, editing the evaluators or
data, and hard-coding answers. Also out of bounds is any behaviour that
depends on recognising particular scenarios rather than reasoning from their
cues.
