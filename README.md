# What's in the Box? Reasoning about Unseen Objects from Multimodal Cues

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2506.14212"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2506.14212-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2506.14212](https://arxiv.org/abs/2506.14212) |
|:---|:---|
| **Domain** | Cognitive Science / Multimodal Reasoning |
| **TL;DR** | The paper models human-like guesses about hidden objects by parsing multimodal cues and integrating them with Bayesian inference over object placements. |

## Summary

The paper introduces the What-is-in-the-Box task, where people and models infer
which listed objects are hidden in boxes after seeing and hearing the boxes
handled. Its model uses neural parsers to convert object names, box images, and
audio into structured evidence, then performs Bayesian inference over all
object-placement hypotheses. The reported full model correlates much more
strongly with human judgments than audio-only, vision-only, or Gemini 2.0 Flash
baselines.

## Problem

How can a system infer which unseen household objects are inside which boxes
from incomplete multimodal evidence: a list of candidate objects, visible box
sizes, and sounds produced when the boxes are handled?

## The paper's method

The paper proposes a neurosymbolic model for hidden-object reasoning in the
What-is-in-the-Box game. Neural perception modules parse language, vision, and
audio into structured evidence, then a Bayesian symbolic engine integrates the
evidence over explicit object-placement hypotheses.

## Key results of the paper

- Full model human-correlation: `r = 0.78`.
- Unimodal ablations are lower: audio-only `r = 0.55`, vision-only `r = 0.52`.
- Gemini 2.0 Flash baseline is much lower: `r = 0.31`.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +118.5 |
| `gpt-5.6-luna-xhigh` | +79.5 |
| `gpt-5.5-xhigh` | -18.0 |
| `gpt-5.6-terra-xhigh` | -42.4 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2506.14212 --mode innovate --agent claude-code
sw evaluate 2506.14212
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

