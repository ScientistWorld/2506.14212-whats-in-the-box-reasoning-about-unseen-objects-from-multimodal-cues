#!/usr/bin/env python3
"""Held-out WiTB test-slice evaluator.

Scores method-authored prediction artifacts without importing method code.
When train scores exist, this evaluator emits the same method keys.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SLICE = "test"
SCENARIO_IDS = {
    "scenario_d_plate_spoon_towel",
    "scenario_e_flashlight_keys_sponge",
}

METHOD_MAP = {
    "witb_full": "full_neurosymbolic",
    "witb_audio": "audio_only_ablation",
    "witb_vision": "vision_only_ablation",
    "uniform_chance": "uniform_chance_baseline",
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


def brier_score(xs: list[float], ys: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(xs, ys)) / max(1, len(xs))


def within_scenario_rank_accuracy(
    scenario_values: list[tuple[list[float], list[float]]],
) -> float:
    correct = 0
    total = 0
    for pred_values, target_values in scenario_values:
        for left in range(len(pred_values)):
            for right in range(left + 1, len(pred_values)):
                target_delta = target_values[left] - target_values[right]
                if target_delta == 0:
                    continue
                pred_delta = pred_values[left] - pred_values[right]
                total += 1
                if pred_delta * target_delta > 0:
                    correct += 1
    return correct / max(1, total)


def load_targets(data_path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(data_path.read_text())
    targets = {}
    for scenario in data["scenarios"]:
        if scenario["id"] in SCENARIO_IDS:
            targets[scenario["id"]] = scenario["human_left_prob"]
    missing = sorted(SCENARIO_IDS - targets.keys())
    if missing:
        raise SystemExit(f"Test slice scenario ids missing from {data_path}: {missing}")
    return targets


def load_train_methods(train_scores: Path) -> set[str] | None:
    if not train_scores.exists():
        return None
    payload = json.loads(train_scores.read_text())
    exp = payload.get("experiments", {}).get("witb_human_alignment", {})
    methods = exp.get("results", {})
    if not isinstance(methods, dict) or not methods:
        raise SystemExit(f"No method results found in {train_scores}")
    return set(methods)


def split_scored_methods(
    results: dict[str, dict[str, float]], required_methods: set[str] | None
) -> list[str]:
    """Return method keys for held-out split scoring.

    The split files are used as anti-shortcut diagnostics over a small fixed
    scenario slice. They should mirror the train-slice method keys, while
    ablation artifacts remain diagnostics for computing gain metrics.
    """
    if required_methods is not None:
        missing = sorted(required_methods - results.keys())
        if missing:
            raise SystemExit(f"Missing prediction artifacts for train-scored methods: {missing}")
        return sorted(required_methods)
    if "full_neurosymbolic" not in results:
        raise SystemExit("Missing full_neurosymbolic prediction artifact")
    return ["full_neurosymbolic"]


def flatten_prediction(
    pred: dict, targets: dict[str, dict[str, float]]
) -> tuple[list[float], list[float], list[float], list[tuple[list[float], list[float]]]]:
    pred_values = []
    target_values = []
    hyp_counts = []
    scenario_values = []
    scenario_ids = [scenario["scenario_id"] for scenario in pred["scenarios"]]
    duplicates = sorted({sid for sid in scenario_ids if scenario_ids.count(sid) > 1})
    if duplicates:
        method = pred.get("method", "<unknown>")
        raise SystemExit(f"{method} predictions contain duplicate scenarios: {duplicates}")
    by_id = {scenario["scenario_id"]: scenario for scenario in pred["scenarios"]}
    missing = sorted(set(targets) - by_id.keys())
    if missing:
        method = pred.get("method", "<unknown>")
        raise SystemExit(f"{method} predictions are missing test scenarios: {missing}")
    for sid in sorted(targets):
        scenario = by_id[sid]
        hyp_counts.append(float(scenario.get("num_hypotheses", 0)))
        scenario_pred_values = []
        scenario_target_values = []
        for obj_name, target in targets[sid].items():
            if obj_name not in scenario.get("pred_left_prob", {}):
                method = pred.get("method", "<unknown>")
                raise SystemExit(f"{method} predictions for {sid} are missing object {obj_name!r}")
            value = float(scenario["pred_left_prob"][obj_name])
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                method = pred.get("method", "<unknown>")
                raise SystemExit(
                    f"{method} prediction for {sid}/{obj_name} must be a finite probability in [0, 1]"
                )
            pred_values.append(value)
            target_values.append(float(target))
            scenario_pred_values.append(value)
            scenario_target_values.append(float(target))
        scenario_values.append((scenario_pred_values, scenario_target_values))
    return pred_values, target_values, hyp_counts, scenario_values


def evaluate_payload(pred: dict, targets: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    method_name = METHOD_MAP.get(pred["method"], pred["method"])
    pred_values, target_values, hyp_counts, scenario_values = flatten_prediction(pred, targets)
    return method_name, {
        "human_correlation": pearson(pred_values, target_values),
        "mean_absolute_error": mean_absolute_error(pred_values, target_values),
        "brier_score": brier_score(pred_values, target_values),
        "within_scenario_rank_accuracy": within_scenario_rank_accuracy(scenario_values),
        "_mean_hypotheses": sum(hyp_counts) / max(1, len(hyp_counts)),
        "_num_scenarios": float(len(targets)),
    }


def late_fusion_average_payload(audio_pred: dict, vision_pred: dict) -> dict:
    """Average unimodal object probabilities without joint hypothesis inference."""
    vision_by_sid = {scenario["scenario_id"]: scenario for scenario in vision_pred["scenarios"]}
    scenarios = []
    for audio_scenario in audio_pred["scenarios"]:
        sid = audio_scenario["scenario_id"]
        if sid not in SCENARIO_IDS:
            continue
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


def prediction_paths(predictions_dir: Path) -> list[Path]:
    return sorted(path for path in predictions_dir.glob("*_predictions.json") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/witb_parsed_benchmark.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    required_methods = load_train_methods(args.train_scores)
    targets = load_targets(args.data)
    all_results = {}
    all_diagnostics = {}
    payloads = {}
    for path in prediction_paths(args.predictions_dir):
        pred = json.loads(path.read_text())
        method, metrics = evaluate_payload(pred, targets)
        payloads[method] = pred
        all_results[method] = {k: v for k, v in metrics.items() if not k.startswith("_")}
        all_diagnostics[method] = {k: v for k, v in metrics.items() if k.startswith("_")}
    if not all_results:
        raise SystemExit(f"No prediction files found in {args.predictions_dir}")

    result_methods = split_scored_methods(all_results, required_methods)

    unimodal = [
        all_results.get("audio_only_ablation", {}).get("human_correlation"),
        all_results.get("vision_only_ablation", {}).get("human_correlation"),
    ]
    if all(x is not None for x in unimodal):
        best_unimodal = max(unimodal)
        for method, metrics in all_results.items():
            if method not in {"audio_only_ablation", "vision_only_ablation"}:
                metrics["multimodal_gain_over_best_unimodal"] = (
                    metrics["human_correlation"] - best_unimodal
                )

    if "audio_only_ablation" in payloads and "vision_only_ablation" in payloads:
        late_payload = late_fusion_average_payload(
            payloads["audio_only_ablation"], payloads["vision_only_ablation"]
        )
        _, late_metrics = evaluate_payload(late_payload, targets)
        late_corr = late_metrics["human_correlation"]
        all_diagnostics["late_fusion_average"] = {
            "_human_correlation": late_corr,
            "_mean_absolute_error": late_metrics["mean_absolute_error"],
        }
        for method, metrics in all_results.items():
            if method not in {"audio_only_ablation", "vision_only_ablation"}:
                metrics["gain_over_late_fusion"] = metrics["human_correlation"] - late_corr

    results = {method: all_results[method] for method in result_methods}
    diagnostics = {
        method: all_diagnostics[method]
        for method in result_methods
        if method in all_diagnostics
    }
    if "late_fusion_average" in all_diagnostics:
        diagnostics["late_fusion_average"] = all_diagnostics["late_fusion_average"]

    payload = {
        "experiments": {
            "witb_human_alignment": {
                "slice": SLICE,
                "scenario_ids": sorted(SCENARIO_IDS),
                "results": results,
                "_diagnostics": diagnostics,
            }
        }
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
