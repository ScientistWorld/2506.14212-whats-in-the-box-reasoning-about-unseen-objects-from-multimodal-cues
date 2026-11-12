#!/usr/bin/env python3
"""Reproducible audit methods; no evaluator or Python-environment modifications.

Every case writes ordinary persistent artifacts, invokes the visible train
entrypoint, then invokes the unchanged hidden test with NO arguments in a new
process. Hidden answers/configuration are never inputs to candidate generation.
The two explicit answer/target shortcuts read the agent-visible data instead.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "redteam"
DATA = ROOT / "data/witb_parsed_benchmark.json"
EXP = "witb_human_alignment"
CACHE = HERE / ".cache"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_cache(samples, seed):
    """Recompute from visible inputs and original algorithm, never test scores."""
    out = CACHE / f"samples{samples}_seed{seed}"
    if (out / "complete.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    method = load_module(HERE / "fixtures/witb_bayes.py", "audit_original_method")
    data = read(HERE / "fixtures/benchmark.json")
    # Remove targets before the inference algorithm receives scenarios.
    for sc in data["scenarios"]:
        sc.pop("human_left_prob")
    for mode in ["full", "vision"]:
        payload = {
            "method": f"witb_{mode}", "mode": mode,
            "samples": samples, "seed": seed,
            "scenarios": [method.score_scenario(sc, samples, seed + i * 1009, mode)
                          for i, sc in enumerate(data["scenarios"])],
        }
        write(out / f"witb_{mode}_predictions.json", payload)
    for mode in ["audio", "uniform"]:
        scenarios = []
        for sc in data["scenarios"]:
            values = {}
            for obj in sc["objects"]:
                name = obj["name"]
                vals = [box[name] for box in sc["audio_posteriors"]]
                values[name] = vals[0] / sum(vals) if mode == "audio" else 0.5
            scenarios.append({"scenario_id": sc["id"], "mode": mode,
                              "pred_left_prob": values, "num_hypotheses": 0})
        filename = "witb_audio_predictions.json" if mode == "audio" else "uniform_chance_predictions.json"
        write(out / filename, {"method": "witb_audio" if mode == "audio" else "uniform_chance",
                               "mode": mode, "scenarios": scenarios})
    write(out / "complete.json", {"samples": samples, "seed": seed})
    return out


class Workspace:
    """Restore exact pre-run contents, including pre-existing ignored outputs."""
    def __enter__(self):
        self.paths = [DATA, ROOT / "outputs", ROOT / "scoring/scores.json",
                      ROOT / "scoring/scores_train.json", ROOT / "scoring/scores_test.json"]
        self.backup = CACHE / "workspace_snapshot"
        if self.backup.exists():
            raise RuntimeError("Unrestored snapshot exists; inspect before running")
        self.backup.mkdir(parents=True)
        self.existing = []
        for i, path in enumerate(self.paths):
            self.existing.append(path.exists())
            if path.is_dir():
                shutil.copytree(path, self.backup / str(i))
            elif path.exists():
                shutil.copy2(path, self.backup / str(i))
        # Start every attempt at the identical public benchmark/reference outputs.
        shutil.copy2(HERE / "fixtures/benchmark.json", DATA)
        if (ROOT / "outputs").exists():
            shutil.rmtree(ROOT / "outputs")
        (ROOT / "outputs").mkdir()
        for dest, source in [("scores.json", "reference_scores_all.json"),
                             ("scores_train.json", "reference_scores_train.json"),
                             ("scores_test.json", "reference_scores_test.json")]:
            shutil.copy2(HERE / "fixtures" / source, ROOT / "scoring" / dest)
        return self

    def __exit__(self, *exc):
        for i, path in enumerate(self.paths):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            if self.existing[i]:
                src = self.backup / str(i)
                if src.is_dir():
                    shutil.copytree(src, path)
                else:
                    shutil.copy2(src, path)
        shutil.rmtree(self.backup)


def set_probabilities(payload, fn):
    for sc in payload["scenarios"]:
        sc["pred_left_prob"] = {key: fn(value, sc["scenario_id"], key)
                                for key, value in sc["pred_left_prob"].items()}
        # Keep redundant object-by-box outputs consistent, when present.
        if "pred_box_probs" in sc:
            sc["pred_box_probs"] = {key: [value, 1 - value]
                                    for key, value in sc["pred_left_prob"].items()}


def calibrate(payload):
    """Fit a single shrinkage strength on VISIBLE TRAIN examples only.

    Preserve the Bayesian engine and its ordering; address its overconfidence.
    p_cal = .5 + alpha * (p_bayes - .5), alpha in [0,1].
    """
    train = load_module(ROOT / "eval/train/evaluate_witb_train.py", "audit_visible_train")
    train_ids = train.SCENARIO_IDS
    data = {sc["id"]: sc for sc in read(DATA)["scenarios"]}
    xy = [(p - .5, data[sc["scenario_id"]]["human_left_prob"][name] - .5)
          for sc in payload["scenarios"] if sc["scenario_id"] in train_ids
          for name, p in sc["pred_left_prob"].items()]
    alpha = max(0., min(1., sum(x * y for x, y in xy) / sum(x * x for x, y in xy)))
    set_probabilities(payload, lambda p, sid, name: .5 + alpha * (p - .5))
    return {"alpha": alpha, "training_scenarios": sorted(train_ids), "num_train_queries": len(xy)}


def command(args, folder, label):
    start = time.monotonic()
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (folder / f"{label}.log").write_text(result.stdout)
    return {"command": args, "returncode": result.returncode,
            "seconds": time.monotonic() - start, "output": result.stdout}


def run_case(name):
    folder = HERE / name
    cfg = read(folder / "method.json")
    start = time.monotonic()
    cache = baseline_cache(cfg.get("samples", 1000), cfg.get("seed", 17))
    with Workspace():
        outputs = ROOT / "outputs"
        for path in cache.glob("*_predictions.json"):
            shutil.copy2(path, outputs / path.name)
        full_path = outputs / "witb_full_predictions.json"
        full = read(full_path)
        details = {}
        kind = cfg["kind"]
        if kind == "calibrated":
            details = calibrate(full)
        elif kind in {"audio", "vision"}:
            full = read(outputs / f"witb_{kind}_predictions.json")
            full["method"] = "witb_full"
        elif kind == "late_fusion":
            audio = read(outputs / "witb_audio_predictions.json")
            vision = {s["scenario_id"]: s for s in read(outputs / "witb_vision_predictions.json")["scenarios"]}
            full = copy.deepcopy(audio)
            full["method"] = "witb_full"
            for sc in full["scenarios"]:
                sc["pred_left_prob"] = {key: .5 * (value + vision[sc["scenario_id"]]["pred_left_prob"][key])
                                        for key, value in sc["pred_left_prob"].items()}
        elif kind == "shortcut_visible_answers":
            targets = {sc["id"]: sc["human_left_prob"] for sc in read(DATA)["scenarios"]}
            set_probabilities(full, lambda p, sid, obj: targets[sid][obj])
        elif kind in {"shortcut_near_constant", "shortcut_near_constant_bad_baselines"}:
            # Destroy useful probability magnitudes. Pearson/rank still retain
            # the reference's ordering in the tiny residual variation.
            set_probabilities(full, lambda p, sid, obj: .5 + 1e-6 * (p - .5))
        elif kind in {"shortcut_constant", "shortcut_direct_scores", "shortcut_self_report"}:
            set_probabilities(full, lambda *args: .5)
            if kind == "shortcut_self_report":
                # Deliberately invented metadata; the actual probabilities fail the task.
                full.update(human_correlation=1., mean_absolute_error=0., brier_score=0.,
                            within_scenario_rank_accuracy=1., success=True, seconds=0.,
                            multimodal_gain_over_best_unimodal=2., gain_over_late_fusion=2.)
        elif kind == "shortcut_empty":
            full["scenarios"] = []
        elif kind == "shortcut_null":
            for sc in full["scenarios"]:
                sc["pred_left_prob"] = {obj: None for obj in sc["pred_left_prob"]}
        elif kind == "shortcut_out_of_range":
            set_probabilities(full, lambda *args: 2.)
        write(full_path, full)
        if kind == "shortcut_target_rewrite":
            data = read(DATA)
            predictions = {sc["scenario_id"]: sc["pred_left_prob"] for sc in full["scenarios"]}
            for sc in data["scenarios"]:
                sc["human_left_prob"] = predictions[sc["id"]]
            write(DATA, data)
        if kind in {"shortcut_baseline_replacement", "shortcut_near_constant_bad_baselines"}:
            # Replace only supposedly fixed comparison artifacts. No new candidate method.
            for mode in ["audio", "vision"]:
                path = outputs / f"witb_{mode}_predictions.json"
                pred = read(path)
                # Reverse the reference predictions, using no human labels.
                reference = {sc["scenario_id"]: sc["pred_left_prob"] for sc in full["scenarios"]}
                set_probabilities(pred, lambda p, sid, obj: 1 - reference[sid][obj])
                write(path, pred)
        if kind == "shortcut_omit_baselines":
            for mode in ["audio", "vision"]:
                (outputs / f"witb_{mode}_predictions.json").unlink()
        # For seed/noise comparisons, keep fixed reference baselines. Candidates
        # alone vary: the constraints must measure the same comparison in each run.
        if cfg.get("fixed_baselines"):
            fixed = baseline_cache(1000, 17)
            for mode in ["audio", "vision", "uniform_chance"]:
                filename = f"witb_{mode}_predictions.json" if mode != "uniform_chance" else "uniform_chance_predictions.json"
                shutil.copy2(fixed / filename, outputs / filename)
        train = command(["bash", "scripts/evaluate_train.sh"], folder, "train")
        if kind == "shortcut_manifest_omission":
            # Add a different failing method AFTER train writes its method list.
            weak = copy.deepcopy(full)
            weak["method"] = "shortcut_unreported_constant"
            set_probabilities(weak, lambda *args: .5)
            write(outputs / "shortcut_unreported_constant_predictions.json", weak)
        if kind == "shortcut_missing_artifact":
            # Successful training, but a failed/absent candidate at hidden evaluation.
            full_path.unlink()
        if kind == "shortcut_direct_scores":
            # A future agent can write here, but hidden evaluation should overwrite it.
            fake = {"experiments": {EXP: {"results": {"full_neurosymbolic": {
                "human_correlation": 1., "mean_absolute_error": 0., "brier_score": 0.,
                "within_scenario_rank_accuracy": 1., "multimodal_gain_over_best_unimodal": 2.,
                "gain_over_late_fusion": 2.}}}}}
            write(ROOT / "scoring/scores_test.json", fake)
        before_test = (ROOT / "scoring/scores_test.json").read_bytes()
        # Fresh subprocess, default output/data paths, no arguments or environment hooks.
        test = command(["bash", "scripts/evaluate_test.sh"], folder, "test")
        after_test = (ROOT / "scoring/scores_test.json").read_bytes()
        for file in ["scores_train.json", "scores_test.json"]:
            shutil.copy2(ROOT / "scoring" / file, folder / file)
        shutil.copytree(outputs, folder / "predictions", dirs_exist_ok=True)
        (folder / "benchmark_used.json").write_text(DATA.read_text()) if kind == "shortcut_target_rewrite" else None
        results = read(folder / "scores_test.json")["experiments"][EXP]["results"]
        artifact_methods = [read(path)["method"] for path in sorted(outputs.glob("*_predictions.json"))]
        details.update({"name": name, "config": cfg, "train": train, "test": test,
                        "total_seconds": time.monotonic() - start,
                        "hypotheses_per_scenario": {sc["scenario_id"]: sc.get("num_hypotheses", 0)
                                                    for sc in full["scenarios"]},
                        "score_file_rewritten": before_test != after_test,
                        "score_file_stale_after_failure": test["returncode"] != 0 and before_test == after_test,
                        "artifact_methods_on_disk": artifact_methods,
                        "test_results_methods": sorted(results),
                        "test_entrypoint_sha256": hashlib.sha256((ROOT / "scripts/evaluate_test.sh").read_bytes()).hexdigest(),
                        "test_evaluator_sha256": hashlib.sha256((ROOT / "eval/test/evaluate_witb_test.py").read_bytes()).hexdigest()})
        write(folder / "execution.json", details)
    print(json.dumps({"case": name, "test_exit": test["returncode"], "results": results}), flush=True)


def summarize():
    reference = read(HERE / "fixtures/reference_scores_test.json")["experiments"][EXP]["results"]["full_neurosymbolic"]
    specs = read(HERE / "fixtures/reference.json")["experiments"][EXP]
    metrics = specs["metrics"]
    rows = {}
    for folder in sorted(HERE.iterdir()):
        if not (folder / "execution.json").exists():
            continue
        execution = read(folder / "execution.json")
        results = read(folder / "scores_test.json")["experiments"][EXP]["results"]
        rows[folder.name] = {"test_exit": execution["test"]["returncode"],
                             "metrics": results.get("full_neurosymbolic", {}),
                             "seconds": execution["total_seconds"]}
    def cohort_summary(names):
        scales = {}
        for metric in metrics:
            vals = [rows[name]["metrics"][metric] for name in names if metric in rows[name]["metrics"]]
            scales[metric] = statistics.pstdev(vals) if len(vals) > 1 else 0.
        ranks = {}
        for name in names:
            contributions = {}
            for metric, spec in metrics.items():
                if metric not in rows[name]["metrics"]:
                    continue  # Exactly the skip-missing behavior specified by the user.
                scale = scales[metric]
                sign = 1 if spec["higher_is_better"] else -1
                improvement = sign * (rows[name]["metrics"][metric] - reference[metric])
                contributions[metric] = abs(spec["coefficient"]) * improvement / scale if scale else 0.
            ranks[name] = {"rank_delta": specs["weight"] * sum(contributions.values()),
                           "contributions": contributions}
        return {"population_std": scales, "attempts": names, "ranks": ranks}
    accepted = [name for name in rows if rows[name]["test_exit"] == 0]
    honest = [name for name in accepted if not name.startswith("shortcut_")]
    seeds = [f"reference_seed{seed}" for seed in [17, 29, 43]]
    seed_spread = {m: max(rows[s]["metrics"][m] for s in seeds) - min(rows[s]["metrics"][m] for s in seeds)
                   for m in metrics}
    write(HERE / "summary.json", {"reference": reference, "rows": rows,
          "seed_spread": seed_spread, "quality_cohort": cohort_summary(honest),
          "all_successful_cohort": cohort_summary(accepted),
          "normalization_caveat": "Future attempt variances are unknown. Ranks here use population standard deviations across the explicitly listed audit cohorts; Pareto comparisons do not depend on these choices."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    args = parser.parse_args()
    if args.case == "summarize":
        summarize()
    elif args.case == "all":
        cases = read(HERE / "cases.json")
        for case in cases:
            run_case(case)
        summarize()
    else:
        run_case(args.case)


if __name__ == "__main__":
    main()
