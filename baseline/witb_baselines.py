#!/usr/bin/env python3
"""Unimodal ablation baselines from the WiTB paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "method"))

from witb_bayes import run  # noqa: E402


def run_audio_only(data_path: Path, output_path: Path, seed: int) -> None:
    """Run the paper's Eq. 3 audio-only ablation.

    For each object, normalize the per-box audio posterior over boxes:
    P(o in box_i | A_i) / sum_j P(o in box_j | A_j).
    """
    data = json.loads(data_path.read_text())
    scenarios = []
    for scenario in data["scenarios"]:
        pred_left_prob = {}
        for obj in scenario["objects"]:
            name = obj["name"]
            scores = [float(box_audio[name]) for box_audio in scenario["audio_posteriors"]]
            denom = sum(scores)
            pred_left_prob[name] = scores[0] / denom if denom > 0 else 1.0 / len(scores)
        scenarios.append(
            {
                "scenario_id": scenario["id"],
                "mode": "audio",
                "pred_left_prob": pred_left_prob,
                "num_hypotheses": 0,
            }
        )
    payload = {
        "method": "witb_audio",
        "mode": "audio",
        "data_path": str(data_path),
        "samples": 0,
        "seed": seed,
        "scenarios": scenarios,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/witb_parsed_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--which", choices=["audio", "vision", "all"], default="all")
    args = parser.parse_args()
    modes = ["audio", "vision"] if args.which == "all" else [args.which]
    for mode in modes:
        output_path = args.output_dir / f"witb_{mode}_predictions.json"
        if mode == "audio":
            run_audio_only(args.data, output_path, args.seed)
        else:
            run(args.data, output_path, mode, args.samples, args.seed)


if __name__ == "__main__":
    main()
