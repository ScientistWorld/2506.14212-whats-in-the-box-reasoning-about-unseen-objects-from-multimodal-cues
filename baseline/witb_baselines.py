#!/usr/bin/env python3
"""Comparison baselines from the WiTB paper plus a chance baseline.

* ``audio_only_ablation`` -- paper Eq. 3: for each object, normalise the
  per-box audio posterior over boxes, P(o|A_i) / sum_j P(o|A_j).
* ``vision_only_ablation`` -- the paper's geometric ablation: the same
  hypothesis enumeration and rejection-sampled fit likelihood as the full
  model, without audio evidence.
* ``late_fusion_average`` -- a post-hoc average of the two unimodal outputs
  (no joint inference over hypotheses).
* ``uniform_chance`` -- equal probability for every box.

Outputs use the predictor format: {"scenarios": [{"id", "box_probs"}]}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "method"))

from witb_bayes import predict  # noqa: E402

NAMES = ["audio_only_ablation", "vision_only_ablation", "late_fusion_average", "uniform_chance"]


def audio_only(scenarios: list[dict]) -> list[dict]:
    out = []
    for scenario in scenarios:
        probs = {}
        for obj in scenario["objects"]:
            name = obj["name"]
            scores = [float(post[name]) for post in scenario["audio_posteriors"]]
            total = sum(scores)
            probs[name] = [s / total for s in scores] if total > 0 else [1.0 / len(scores)] * len(scores)
        out.append({"id": scenario["id"], "box_probs": probs})
    return out


def late_fusion(audio: list[dict], vision: list[dict]) -> list[dict]:
    out = []
    for a, v in zip(audio, vision):
        probs = {
            name: [0.5 * (pa + pv) for pa, pv in zip(a["box_probs"][name], v["box_probs"][name])]
            for name in a["box_probs"]
        }
        out.append({"id": a["id"], "box_probs": probs})
    return out


def uniform(scenarios: list[dict]) -> list[dict]:
    return [
        {
            "id": s["id"],
            "box_probs": {o["name"]: [1.0 / len(s["boxes"])] * len(s["boxes"]) for o in s["objects"]},
        }
        for s in scenarios
    ]


def run_all(scenarios: list[dict], samples: int, seed: int, cache_dir: Path | None) -> dict[str, list[dict]]:
    audio = audio_only(scenarios)
    vision = predict(scenarios, "vision", samples, seed, cache_dir)
    return {
        "audio_only_ablation": audio,
        "vision_only_ablation": vision,
        "late_fusion_average": late_fusion(audio, vision),
        "uniform_chance": uniform(scenarios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="write one predictor-format file per baseline")
    parser.add_argument("--combined-out", type=Path, default=None, help="write all baselines into one file")
    parser.add_argument("--which", choices=NAMES + ["all"], default="all")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    scenarios = json.loads(args.data.read_text())["scenarios"]
    results = run_all(scenarios, args.samples, args.seed, args.cache_dir)
    if args.which != "all":
        results = {args.which: results[args.which]}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, preds in results.items():
            (args.output_dir / f"{name}.json").write_text(json.dumps({"method": name, "scenarios": preds}) + "\n")
    if args.combined_out is not None:
        args.combined_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "description": "Fixed comparison predictions (paper ablations and chance) on this split.",
            "samples": args.samples,
            "seed": args.seed,
            "baselines": {name: {"method": name, "scenarios": preds} for name, preds in results.items()},
        }
        args.combined_out.write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
