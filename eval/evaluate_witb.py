#!/usr/bin/env python3
"""Evaluate WiTB prediction artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METHOD_MAP = {
    "witb_full": "full_neurosymbolic",
    "witb_audio": "audio_only_ablation",
    "witb_vision": "vision_only_ablation",
}


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def mean_absolute_error(xs: list[float], ys: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(xs, ys)) / max(1, len(xs))


def load_targets(data_path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(data_path.read_text())
    return {scenario["id"]: scenario["human_left_prob"] for scenario in data["scenarios"]}


def flatten_prediction(pred: dict, targets: dict[str, dict[str, float]]) -> tuple[list[float], list[float], list[float]]:
    pred_values = []
    target_values = []
    hyp_counts = []
    for scenario in pred["scenarios"]:
        sid = scenario["scenario_id"]
        hyp_counts.append(float(scenario.get("num_hypotheses", 0)))
        for obj_name, target in targets[sid].items():
            pred_values.append(float(scenario["pred_left_prob"][obj_name]))
            target_values.append(float(target))
    return pred_values, target_values, hyp_counts


def evaluate_payload(pred: dict, targets: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    method_name = METHOD_MAP.get(pred["method"], pred["method"])
    pred_values, target_values, hyp_counts = flatten_prediction(pred, targets)
    return method_name, {
        "human_correlation": pearson(pred_values, target_values),
        "_mean_abs_error": mean_absolute_error(pred_values, target_values),
        "_mean_hypotheses": sum(hyp_counts) / max(1, len(hyp_counts)),
    }


def late_fusion_average_payload(audio_pred: dict, vision_pred: dict) -> dict:
    """Average unimodal object probabilities without joint hypothesis inference."""
    vision_by_sid = {scenario["scenario_id"]: scenario for scenario in vision_pred["scenarios"]}
    scenarios = []
    for audio_scenario in audio_pred["scenarios"]:
        sid = audio_scenario["scenario_id"]
        vision_scenario = vision_by_sid[sid]
        pred_left_prob = {}
        for obj_name, audio_prob in audio_scenario["pred_left_prob"].items():
            vision_prob = vision_scenario["pred_left_prob"][obj_name]
            pred_left_prob[obj_name] = 0.5 * (float(audio_prob) + float(vision_prob))
        scenarios.append(
            {
                "scenario_id": sid,
                "mode": "late_fusion_average",
                "pred_left_prob": pred_left_prob,
                "num_hypotheses": 0,
            }
        )
    return {
        "method": "late_fusion_average",
        "mode": "late_fusion_average",
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/witb_parsed_benchmark.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    targets = load_targets(args.data)
    results = {}
    diagnostics = {}
    payloads = {}
    for path in sorted(args.predictions_dir.glob("witb_*_predictions.json")):
        pred = json.loads(path.read_text())
        method, metrics = evaluate_payload(pred, targets)
        payloads[method] = pred
        results[method] = {k: v for k, v in metrics.items() if not k.startswith("_")}
        diagnostics[method] = {k: v for k, v in metrics.items() if k.startswith("_")}
    if not results:
        raise SystemExit(f"No prediction files found in {args.predictions_dir}")
    full = results.get("full_neurosymbolic", {}).get("human_correlation")
    unimodal = [
        results.get("audio_only_ablation", {}).get("human_correlation"),
        results.get("vision_only_ablation", {}).get("human_correlation"),
    ]
    if full is not None and all(x is not None for x in unimodal):
        results["full_neurosymbolic"]["multimodal_gain_over_best_unimodal"] = full - max(unimodal)
    if "audio_only_ablation" in payloads and "vision_only_ablation" in payloads and full is not None:
        late_payload = late_fusion_average_payload(
            payloads["audio_only_ablation"], payloads["vision_only_ablation"]
        )
        _, late_metrics = evaluate_payload(late_payload, targets)
        late_corr = late_metrics["human_correlation"]
        diagnostics["late_fusion_average"] = {
            "_human_correlation": late_corr,
            "_mean_abs_error": late_metrics["_mean_abs_error"],
        }
        results["full_neurosymbolic"]["joint_gain_over_late_fusion"] = full - late_corr
    payload = {
        "experiments": {
            "witb_human_alignment": {
                "results": results,
                "_diagnostics": diagnostics
            }
        }
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
