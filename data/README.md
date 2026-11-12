# What's-in-the-Box data

The paper's 25 video stimuli, its parsed neural outputs and its 54
participants' ratings are not released. This gym therefore uses a simulated
What's-in-the-Box world modelled on the paper's setup:

- 2-5 objects (from 50 household object types) hidden in two boxes; a scenario
  usually pairs a box that can hold most objects with one that may only hold
  one or two small ones.
- Each object's true placement is drawn uniformly from the placements in
  which every box is non-empty and everything physically fits (squashable
  objects can be compressed).
- The cues are produced from the true scene and are imperfect in the ways the
  paper describes: the per-type object parses are biased and often
  over-confident; box measurements and lifted-weight estimates are noisy; and
  the sound posteriors come from a simulated zero-shot classifier on the mixed
  sound of the box's actual contents (masking, quiet objects, label/sound
  misalignment).

## Files

`witb_train.json` - 240 visible scenarios (881 objects) with true placements.
It is agent-authored and tracked in git; nothing needs downloading. The 2000
hidden test scenarios come from the same world and are held only by the
hidden evaluator.

Each scenario:

```json
{
  "id": "witb_ab402d40e4",
  "objects": [{"name": "boxed laptop", "dims_cm": [34.8, 23.3, 6.2], "dims_std_cm": [4.2, 2.8, 0.7],
               "weight_kg": 1.795, "weight_std_kg": 0.538, "material": "cardboard and electronics",
               "rigidity": 0.87}, "..."],
  "boxes": [{"name": "left", "dims_cm": [20.3, 20.8, 23.1], "dims_std_cm": [1.2, 1.2, 1.4], "lifted_weight_kg": 0.378},
            {"name": "right", "...": "..."}],
  "audio_posteriors": [{"headphone case": 0.93, "boxed laptop": 0.07}, {"...": "..."}],
  "true_box": {"headphone case": 0, "boxed laptop": 1}
}
```

- `dims_cm` / `dims_std_cm`: language-model (objects) or vision-model (boxes)
  estimate of the dimensions, as mean and standard deviation, in cm (inner
  dimensions for boxes).
- `weight_kg` / `weight_std_kg`, `material`, `rigidity`: language-model
  estimates for the object type (rigidity 0 = fully squashable, 1 = rigid).
- `lifted_weight_kg`: estimated total weight of the box and its contents from
  how it moves when handled.
- `audio_posteriors[i]`: sound-classifier posterior over the scenario's
  object names for box i.
- `true_box`: index of the box that actually holds each object (train only;
  removed from predictor inputs).
