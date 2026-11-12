#!/usr/bin/env python3
"""Hidden-test evaluator for What's-in-the-Box placement inference.

Runs every method on disk on hidden test scenarios and scores the result.

* A method is an executable ``method/predictors/<name>.sh`` called as
  ``bash <predictor> INPUT_JSON OUTPUT_JSON``.  INPUT_JSON holds scenarios
  without labels; OUTPUT_JSON must give, for every scenario id and every object
  name, a probability for each box (``box_probs``).
* Hidden scenarios and their true placements live only in this directory and
  are loaded into memory before any predictor runs.  Predictors receive a copy
  of the inputs in a temporary directory outside the workspace, and this
  directory is made unreadable while they run.
* Every number is recomputed here from raw probabilities; nothing a method
  writes besides ``box_probs`` is read.  A predictor that crashes, times out,
  or returns an incomplete or invalid distribution is marked failed and gets
  the worst value of every scored metric.  The score file is first written
  with worst values for every discovered method, so an interrupted evaluation
  never leaves stale scores behind.
* Single-cue and chance comparison numbers come from fixed prediction files
  stored here (``fixed_baselines_test.json``); they are diagnostics only.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HIDDEN_PATH = HERE / "witb_test_hidden.json"
FIXED_BASELINES_PATH = HERE / "fixed_baselines_test.json"
PREDICTORS_DIR = ROOT / "method" / "predictors"
SCORES_PATH = ROOT / "scoring" / "scores_test.json"
LOG_DIR = ROOT / "outputs" / "eval_logs" / "test"
EXPERIMENT = "witb_placement_inference"
PROB_FLOOR = 1e-4
SUM_TOL = 1e-3
RANGE_TOL = 1e-6  # floating-point slack; values are clipped into [0, 1]
PREDICT_TIMEOUT_S = 2 * 60 * 60
WORST = {
    "truth_correlation": -1.0,
    "brier_score": 1.0,
    "log_loss": -math.log(PROB_FLOOR),
}


def load_hidden() -> tuple[list[dict], list[dict[str, int]]]:
    data = json.loads(HIDDEN_PATH.read_text())
    inputs, truth = [], []
    for scenario in data["scenarios"]:
        visible = {k: v for k, v in scenario.items() if k != "true_box"}
        inputs.append(visible)
        truth.append({name: int(b) for name, b in scenario["true_box"].items()})
    return inputs, truth


def discover_predictors() -> list[Path]:
    if not PREDICTORS_DIR.is_dir():
        return []
    return sorted(p for p in PREDICTORS_DIR.glob("*.sh") if p.is_file())


def validate_predictions(payload, inputs: list[dict]) -> list[dict[str, list[float]]]:
    """Return per-scenario {object: [p_box...]} or raise ValueError."""
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError("output must be a JSON object with a 'scenarios' list")
    by_id = {}
    for item in payload["scenarios"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("every scenario entry needs a string 'id'")
        if item["id"] in by_id:
            raise ValueError(f"duplicate scenario id {item['id']!r}")
        by_id[item["id"]] = item
    out = []
    for scenario in inputs:
        sid = scenario["id"]
        if sid not in by_id:
            raise ValueError(f"missing scenario {sid!r}")
        box_probs = by_id[sid].get("box_probs")
        if not isinstance(box_probs, dict):
            raise ValueError(f"scenario {sid!r} has no 'box_probs' object")
        k = len(scenario["boxes"])
        probs = {}
        for obj in scenario["objects"]:
            name = obj["name"]
            values = box_probs.get(name)
            if not isinstance(values, list) or len(values) != k:
                raise ValueError(f"scenario {sid!r} object {name!r}: need a list of {k} box probabilities")
            clean = []
            for v in values:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"scenario {sid!r} object {name!r}: non-numeric probability")
                v = float(v)
                if not math.isfinite(v) or v < -RANGE_TOL or v > 1.0 + RANGE_TOL:
                    raise ValueError(f"scenario {sid!r} object {name!r}: probability outside [0, 1]")
                clean.append(min(1.0, max(0.0, v)))
            total = sum(clean)
            if abs(total - 1.0) > SUM_TOL:
                raise ValueError(f"scenario {sid!r} object {name!r}: box probabilities sum to {total:.6f}, not 1")
            probs[name] = [v / total for v in clean]
        out.append(probs)
    return out


def score(probs: list[dict[str, list[float]]], truth: list[dict[str, int]]) -> dict[str, float]:
    flat_p, flat_y, briers, losses = [], [], [], []
    for scen_probs, scen_truth in zip(probs, truth):
        for name, true_box in scen_truth.items():
            p = scen_probs[name]
            sq = 0.0
            for b, pb in enumerate(p):
                y = 1.0 if b == true_box else 0.0
                flat_p.append(pb)
                flat_y.append(y)
                sq += (pb - y) ** 2
            briers.append(0.5 * sq)
            losses.append(-math.log(max(PROB_FLOOR, p[true_box])))
    n = len(flat_p)
    mp = sum(flat_p) / n
    my = sum(flat_y) / n
    vp = sum((x - mp) ** 2 for x in flat_p)
    vy = sum((y - my) ** 2 for y in flat_y)
    cov = sum((x - mp) * (y - my) for x, y in zip(flat_p, flat_y))
    corr = cov / math.sqrt(vp * vy) if vp > 0 and vy > 0 else 0.0
    return {
        "truth_correlation": corr,
        "brier_score": sum(briers) / len(briers),
        "log_loss": sum(losses) / len(losses),
    }


def write_scores(results: dict, diagnostics: dict, baselines: dict, meta: dict) -> None:
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
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCORES_PATH.with_name(SCORES_PATH.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, SCORES_PATH)


def _set_mode(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def run_predictor(predictor: Path, input_path: Path, output_path: Path, log_path: Path) -> tuple[bool, str, float]:
    start = time.time()
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
            return False, f"could not start predictor: {exc}", time.time() - start
        try:
            code = proc.wait(timeout=PREDICT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return False, f"timed out after {PREDICT_TIMEOUT_S} s", time.time() - start
    if code != 0:
        return False, f"exit code {code}", time.time() - start
    return True, "", time.time() - start


def fixed_baseline_scores(inputs: list[dict], truth: list[dict[str, int]]) -> dict:
    if not FIXED_BASELINES_PATH.exists():
        return {}
    data = json.loads(FIXED_BASELINES_PATH.read_text())
    out = {}
    for name, payload in sorted(data.get("baselines", {}).items()):
        try:
            out[name] = score(validate_predictions(payload, inputs), truth)
        except ValueError as exc:
            out[name] = {"_error": str(exc)}
    return out


def main() -> int:
    inputs, truth = load_hidden()
    baselines = fixed_baseline_scores(inputs, truth)
    predictors = discover_predictors()
    meta = {
        "split": "test",
        "num_scenarios": len(inputs),
        "num_objects": sum(len(t) for t in truth),
    }
    names = [p.stem for p in predictors]
    results = {name: dict(WORST) for name in names}
    diagnostics = {name: {"status": "not_run"} for name in names}
    write_scores(results, diagnostics, baselines, meta)
    if not predictors:
        print(f"No predictors found in {PREDICTORS_DIR}", file=sys.stderr)
        return 1

    keep_dir = os.environ.get("WITB_KEEP_PREDICTIONS_DIR")
    work = Path(tempfile.mkdtemp(prefix="witb_"))
    input_path = work / "scenarios.json"
    input_path.write_text(json.dumps({"format": "witb-scenarios-v2", "scenarios": inputs}) + "\n")
    os.chmod(input_path, 0o444)
    original_mode = stat.S_IMODE(HERE.stat().st_mode)
    try:
        for predictor in predictors:
            name = predictor.stem
            output_path = work / f"{name}.out.json"
            _set_mode(HERE, 0)
            try:
                ok, reason, seconds = run_predictor(predictor, input_path, output_path, LOG_DIR / f"{name}.log")
            finally:
                _set_mode(HERE, original_mode)
            diag = {"runtime_s": round(seconds, 2)}
            if ok:
                try:
                    payload = json.loads(output_path.read_text())
                    probs = validate_predictions(payload, inputs)
                    results[name] = score(probs, truth)
                    diag["status"] = "ok"
                    if keep_dir:
                        Path(keep_dir).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(output_path, Path(keep_dir) / f"{name}.json")
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    ok, reason = False, f"invalid output: {exc}"
            if not ok:
                results[name] = dict(WORST)
                diag["status"] = "failed"
                diag["reason"] = reason[:500]
                print(f"[evaluate_test] {name}: FAILED ({reason[:200]})", file=sys.stderr)
            else:
                print(f"[evaluate_test] {name}: {json.dumps(results[name])}")
            diagnostics[name] = diag
            write_scores(results, diagnostics, baselines, meta)
    finally:
        _set_mode(HERE, original_mode)
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
