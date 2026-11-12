# Problem Definition

## Research Question

Several household objects are hidden in two closed boxes. An observer is told
which objects there are, sees the boxes, and hears them being shaken. How can
a system turn these indirect, noisy cues into a probability, for every object,
of which box it is in?

## Why It Matters

People routinely reason about things they cannot see by combining what they
know about objects, what they see and what they hear. A system that does this
well, with well-calibrated probabilities, is a model of that human ability and
a building block for robots that must reason about occluded objects.

## Inputs (per scenario)

The cues have already been parsed into structured form, the way
language, vision and audio models would parse them:

- **Objects** (2-5 per scenario): name, and a language-model estimate of their
  dimensions in cm (mean and standard deviation per dimension), weight in kg
  (mean and standard deviation), material, and rigidity (0 = fully squashable,
  1 = rigid). These estimates come from the object's name alone, so they are
  generic for that kind of object and can be biased or over-confident.
- **Boxes** (2 per scenario, `left` and `right`): a vision-model estimate of
  the inner dimensions in cm (mean and standard deviation) and an estimate of
  how heavy the box is when it is lifted and shaken (`lifted_weight_kg`, total
  of box and contents, noisy).
- **Sound**: for each box, a zero-shot sound classifier's posterior over the
  scenario's object names, computed from the sound of that box being shaken.
  Sounds of several objects mix, quiet objects can be masked by loud ones,
  tightly packed boxes rattle less, and the classifier is imperfect.

## Output

For every object, a probability distribution over the two boxes.

## Success Criteria

Probabilities that track where the objects really are and are calibrated, on
scenarios the system has never seen. Every box holds at least one object, and
every object is in exactly one box.

## Data

The original study's videos and participant ratings are not public. The gym
uses a simulated What's-in-the-Box world with household objects, box sizes and
cue noise modelled on the original study. 240 labelled scenarios are visible
in `data/witb_train.json`; scoring uses 2000 hidden scenarios from the same
world.
