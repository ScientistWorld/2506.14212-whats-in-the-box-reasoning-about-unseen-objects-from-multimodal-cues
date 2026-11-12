# Paper's Method

## Key Contribution

A neurosymbolic model of the What's-in-the-Box game. Neural models parse the
object names (an LLM: dimensions with uncertainty, weight, material,
rigidity), the first video frame (Gemini: box dimensions) and the sound of
each box being shaken (CLAP: posterior over the scenario's object names). A
Bayesian engine then integrates these cues over explicit object-placement
hypotheses.

## Algorithm (implemented in `method/witb_bayes.py`)

1. Enumerate every assignment of the N objects to the K boxes that leaves no
   box empty (|H| = K!·S(N, K)); uniform prior over them.
2. Visual evidence P(O_i | H_i): draw object and box dimensions 1000 times
   from the parsed normal distributions and count how often the hypothesised
   objects physically fit in box i (rejection sampling).
3. Audio evidence P(H_i | A_i) = Π_{o∈H_i} P(o | A_i) from the per-box
   classifier posterior (paper Eq. 2).
4. P(H | O, A) ∝ Π_i P(O_i | H_i) P(H_i | A_i) (paper Eq. 1, cues treated as
   conditionally independent and equally weighted); normalise and
   marginalise into per-object box probabilities.

Implementation choices the paper does not specify: the fit check is an
axis-aligned guillotine packer that tries all six orientations of each object
(at most eight free regions); sampled dimensions are clamped at 0.1 cm; one set
of 1000 draws per scenario is shared by all hypotheses; the random stream is
seeded by (seed 17, scenario id). The parsed weight, rigidity, material and
lifted-weight fields are not used, as in the paper's equations.

The predictor `method/predictors/full_neurosymbolic.sh` runs this model with
1000 samples and seed 17.

## Paper ablations (in `baseline/witb_baselines.py`)

- Audio-only (paper Eq. 3): P(o in box i) = P(o|A_i) / Σ_j P(o|A_j).
- Vision-only: the same hypothesis inference with visual evidence only.
- Late-fusion average of the two (not a paper baseline; tests whether joint
  inference adds more than averaging).

## Main Claims

The full model correlates strongly with mean human ratings (r = 0.78), better
than audio-only (0.55), vision-only (0.52) and Gemini 2.0 Flash (0.31).
Joint inference over hypotheses is not a simple average of the unimodal
outputs. Limitations the paper names: equal cue weights (it suggests learning
cue reliabilities), no use of how heavy a box looks when handled, and an audio
model that misses quiet sounds or sounds mixed with others.

## In this gym

On the 2000 hidden simulated scenarios, the full model reaches correlation
0.802 with the true placements. Audio-only reaches 0.770, vision-only 0.454
and the late-fusion average 0.764. It is also better on Brier score than all
three; on log loss audio-only is level with it (0.340 vs 0.346).
