#!/usr/bin/env python3
"""Visible-train evaluator for What's-in-the-Box placement inference.

Runs every method on disk on the visible train scenarios
(``data/witb_train.json``) and scores it against their true placements, using
the same interface and metrics as the hidden test.

* A method is an executable ``method/predictors/<name>.sh`` called as
  ``bash <predictor> INPUT_JSON OUTPUT_JSON``.  INPUT_JSON holds scenarios
  without labels; OUTPUT_JSON must give, for every scenario id and every object
  name, a probability for each box (``box_probs``).
* A predictor that crashes, times out or returns an incomplete or invalid
  distribution is marked failed and gets the worst value of every metric.
* Single-cue and chance comparison numbers come from fixed prediction files
  (``eval/train/fixed_baselines_train.json``); they are diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_PATH = ROOT / "data" / "witb_train.json"
FIXED_BASELINES_PATH = HERE / "fixed_baselines_train.json"
PREDICTORS_DIR = ROOT / "method" / "predictors"
DEFAULT_SCORES = ROOT / "scoring" / "scores_train.json"
LOG_DIR = ROOT / "outputs" / "eval_logs" / "train"
EXPERIMENT = "witb_placement_inference"
PROB_FLOOR = 1e-4
SUM_TOL = 1e-3
RANGE_TOL = 1e-6  # floating-point slack; values are clipped into [0, 1]
PREDICT_TIMEOUT_S = 2 * 60 * 60
WORST_VALUES = {
    "truth_correlation": -1.0,
    "brier_score": 1.0,
    "log_loss": -math.log(PROB_FLOOR),
}


def read_split(path: Path) -> tuple[list[dict], list[dict[str, int]]]:
    data = json.loads(path.read_text())
    scenarios, labels = [], []
    for scenario in data["scenarios"]:
        scenarios.append({k: v for k, v in scenario.items() if k != "true_box"})
        labels.append({name: int(b) for name, b in scenario["true_box"].items()})
    return scenarios, labels


def check_output(payload, scenarios: list[dict]) -> list[dict[str, list[float]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError("output must be a JSON object with a 'scenarios' list")
    entries = {}
    for entry in payload["scenarios"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("every scenario entry needs a string 'id'")
        if entry["id"] in entries:
            raise ValueError(f"duplicate scenario id {entry['id']!r}")
        entries[entry["id"]] = entry
    checked = []
    for scenario in scenarios:
        sid = scenario["id"]
        if sid not in entries:
            raise ValueError(f"missing scenario {sid!r}")
        table = entries[sid].get("box_probs")
        if not isinstance(table, dict):
            raise ValueError(f"scenario {sid!r} has no 'box_probs' object")
        num_boxes = len(scenario["boxes"])
        per_object = {}
        for obj in scenario["objects"]:
            values = table.get(obj["name"])
            if not isinstance(values, list) or len(values) != num_boxes:
                raise ValueError(f"scenario {sid!r} object {obj['name']!r}: need {num_boxes} box probabilities")
            row = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"scenario {sid!r} object {obj['name']!r}: non-numeric probability")
                value = float(value)
                if not math.isfinite(value) or value < -RANGE_TOL or value > 1.0 + RANGE_TOL:
                    raise ValueError(f"scenario {sid!r} object {obj['name']!r}: probability outside [0, 1]")
                row.append(min(1.0, max(0.0, value)))
            total = sum(row)
            if abs(total - 1.0) > SUM_TOL:
                raise ValueError(f"scenario {sid!r} object {obj['name']!r}: probabilities sum to {total:.6f}, not 1")
            per_object[obj["name"]] = [value / total for value in row]
        checked.append(per_object)
    return checked


def compute_metrics(checked: list[dict[str, list[float]]], labels: list[dict[str, int]]) -> dict[str, float]:
    preds, targets, brier_terms, loss_terms = [], [], [], []
    for per_object, label in zip(checked, labels):
        for name, box in label.items():
            row = per_object[name]
            squared = 0.0
            for idx, value in enumerate(row):
                target = 1.0 if idx == box else 0.0
                preds.append(value)
                targets.append(target)
                squared += (value - target) ** 2
            brier_terms.append(0.5 * squared)
            loss_terms.append(-math.log(max(PROB_FLOOR, row[box])))
    count = len(preds)
    mean_p = sum(preds) / count
    mean_t = sum(targets) / count
    var_p = sum((p - mean_p) ** 2 for p in preds)
    var_t = sum((t - mean_t) ** 2 for t in targets)
    cov = sum((p - mean_p) * (t - mean_t) for p, t in zip(preds, targets))
    return {
        "truth_correlation": cov / math.sqrt(var_p * var_t) if var_p > 0 and var_t > 0 else 0.0,
        "brier_score": sum(brier_terms) / len(brier_terms),
        "log_loss": sum(loss_terms) / len(loss_terms),
    }


def save(scores_path: Path, results: dict, diagnostics: dict, baselines: dict, meta: dict) -> None:
    payload = {
        "experiments": {
            EXPERIMENT: {
                **meta,
                "results": results,
                "_diagnostics": diagnostics,
                "_baselines": baselines,
            }
        }
    }
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = scores_path.with_name(scores_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, scores_path)


def execute(predictor: Path, input_path: Path, output_path: Path, log_path: Path) -> tuple[bool, str, float]:
    began = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        try:
            proc = subprocess.Popen(
                ["bash", str(predictor), str(input_path), str(output_path)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return False, f"could not start predictor: {exc}", time.time() - began
        try:
            code = proc.wait(timeout=PREDICT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return False, f"timed out after {PREDICT_TIMEOUT_S} s", time.time() - began
    if code != 0:
        return False, f"exit code {code}", time.time() - began
    return True, "", time.time() - began


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()

    scenarios, labels = read_split(DATA_PATH)
    baselines = {}
    if FIXED_BASELINES_PATH.exists():
        fixed = json.loads(FIXED_BASELINES_PATH.read_text())
        for name, payload in sorted(fixed.get("baselines", {}).items()):
            try:
                baselines[name] = compute_metrics(check_output(payload, scenarios), labels)
            except ValueError as exc:
                baselines[name] = {"_error": str(exc)}
    predictors = sorted(p for p in PREDICTORS_DIR.glob("*.sh") if p.is_file()) if PREDICTORS_DIR.is_dir() else []
    meta = {
        "split": "train",
        "num_scenarios": len(scenarios),
        "num_objects": sum(len(label) for label in labels),
    }
    results = {p.stem: dict(WORST_VALUES) for p in predictors}
    diagnostics = {p.stem: {"status": "not_run"} for p in predictors}
    save(args.scores, results, diagnostics, baselines, meta)
    if not predictors:
        print(f"No predictors found in {PREDICTORS_DIR}", file=sys.stderr)
        return 1

    work = Path(tempfile.mkdtemp(prefix="witb_"))
    input_path = work / "scenarios.json"
    input_path.write_text(json.dumps({"format": "witb-scenarios-v2", "scenarios": scenarios}) + "\n")
    try:
        for predictor in predictors:
            name = predictor.stem
            output_path = work / f"{name}.out.json"
            ok, reason, seconds = execute(predictor, input_path, output_path, LOG_DIR / f"{name}.log")
            diag = {"runtime_s": round(seconds, 2)}
            if ok:
                try:
                    results[name] = compute_metrics(check_output(json.loads(output_path.read_text()), scenarios), labels)
                    diag["status"] = "ok"
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    ok, reason = False, f"invalid output: {exc}"
            if not ok:
                results[name] = dict(WORST_VALUES)
                diag["status"] = "failed"
                diag["reason"] = reason[:500]
                print(f"[evaluate_train] {name}: FAILED ({reason[:200]})", file=sys.stderr)
            else:
                print(f"[evaluate_train] {name}: {json.dumps(results[name])}")
            diagnostics[name] = diag
            save(args.scores, results, diagnostics, baselines, meta)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
