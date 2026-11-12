#!/usr/bin/env python3
"""Audit predictors: better / weaker variants of the reference and shortcuts.

Every mode reads a scenario file (no labels) and writes the predictor output
format {"scenarios": [{"id", "box_probs"}]}.  Variants that need parameters
read them from a file produced by fit.py on the visible train split only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "method"))

import witb_bayes as wb  # noqa: E402

CLIP = 1e-6


def cache_dir() -> Path | None:
    value = os.environ.get("WITB_CACHE_DIR")
    return Path(value) if value else None


def logit(p: float) -> float:
    p = min(1 - CLIP, max(CLIP, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Generalised posterior used by the fitted variants.
# ---------------------------------------------------------------------------


def surface_cm2(dims) -> float:
    a, b, c = dims
    return 2.0 * (a * b + a * c + b * c)


def generalized_posterior(scenario: dict, table: dict, params: dict) -> dict[str, list[float]]:
    """Posterior with cue exponents, a visual-fit floor and an optional weight cue.

    params: eps (visual floor), gamma (visual exponent), beta (audio exponent),
    use_vision, use_audio, sigma_w (log-weight noise; None = ignore weight),
    tare (kg of box per cm^2 of box surface).
    """
    objects = scenario["objects"]
    n, k = len(objects), len(scenario["boxes"])
    hyps = wb.enumerate_hypotheses(n, k)
    eps = params.get("eps", 0.0)
    gamma = params.get("gamma", 1.0)
    beta = params.get("beta", 1.0)
    sigma_w = params.get("sigma_w")
    tare = params.get("tare", 0.0)
    log_w = []
    for assign in hyps:
        total = 0.0
        for b in range(k):
            subset = tuple(i for i in range(n) if assign[i] == b)
            if params.get("use_vision", True):
                fit = eps + (1.0 - eps) * table[wb._key(b, subset)]
                total += gamma * math.log(max(1e-12, fit))
            if params.get("use_audio", True):
                total += beta * math.log(max(1e-300, wb.audio_term(scenario, b, subset)))
            if sigma_w:
                box = scenario["boxes"][b]
                expected = tare * surface_cm2(box["dims_cm"]) + sum(objects[i]["weight_kg"] for i in subset)
                observed = float(box["lifted_weight_kg"])
                z = (math.log(max(1e-3, observed)) - math.log(max(1e-3, expected))) / sigma_w
                total += -0.5 * z * z
        log_w.append(total)
    top = max(log_w)
    weights = [math.exp(w - top) for w in log_w]
    z = sum(weights)
    probs = {o["name"]: [0.0] * k for o in objects}
    for assign, w in zip(hyps, weights):
        for i, b in enumerate(assign):
            probs[objects[i]["name"]][b] += w / z
    return probs


def audio_only(scenario: dict) -> dict[str, list[float]]:
    out = {}
    for obj in scenario["objects"]:
        scores = [float(post[obj["name"]]) for post in scenario["audio_posteriors"]]
        total = sum(scores)
        out[obj["name"]] = [s / total for s in scores]
    return out


# ---------------------------------------------------------------------------
# Modes.
# ---------------------------------------------------------------------------


def reference_preds(scenarios, samples=1000, seed=17):
    return {p["id"]: p["box_probs"] for p in wb.predict(scenarios, "full", samples, seed, cache_dir())}


def run_mode(args, scenarios: list[dict]) -> dict:
    mode = args.mode
    extra = {}
    if mode == "full":
        preds = reference_preds(scenarios, args.samples, args.seed)
    elif mode == "vision_only":
        preds = {p["id"]: p["box_probs"] for p in wb.predict(scenarios, "vision", 1000, 17, cache_dir())}
    elif mode == "audio_only":
        preds = {s["id"]: audio_only(s) for s in scenarios}
    elif mode == "late_fusion":
        vis = {p["id"]: p["box_probs"] for p in wb.predict(scenarios, "vision", 1000, 17, cache_dir())}
        preds = {}
        for s in scenarios:
            aud = audio_only(s)
            preds[s["id"]] = {n: [0.5 * (a + v) for a, v in zip(aud[n], vis[s["id"]][n])] for n in aud}
    elif mode == "fitted":
        params = json.loads(Path(args.params).read_text())[args.key]
        tables = wb.visual_tables(scenarios, 1000, 17, cache_dir())
        preds = {s["id"]: generalized_posterior(s, t, params) for s, t in zip(scenarios, tables)}
    elif mode == "trained_late_fusion":
        params = json.loads(Path(args.params).read_text())[args.key]
        vis = {p["id"]: p["box_probs"] for p in wb.predict(scenarios, "vision", 1000, 17, cache_dir())}
        preds = {}
        for s in scenarios:
            aud = audio_only(s)
            row = {}
            for o in s["objects"]:
                n = o["name"]
                c = params.get("logit_clip", CLIP)
                xa = logit(min(1 - c, max(c, aud[n][0])))
                xv = logit(min(1 - c, max(c, vis[s["id"]][n][0])))
                x = params["bias"] + params["w_audio"] * xa + params["w_vision"] * xv
                p = sigmoid(x)
                row[n] = [p, 1.0 - p]
            preds[s["id"]] = row
    # ------------------------------- shortcuts -------------------------------
    elif mode == "constant":
        preds = {s["id"]: {o["name"]: [0.5, 0.5] for o in s["objects"]} for s in scenarios}
    elif mode == "near_constant":
        ref = reference_preds(scenarios)
        preds = {sid: {n: [0.5 + 1e-6 * (p[0] - 0.5), 0.5 - 1e-6 * (p[0] - 0.5)] for n, p in row.items()} for sid, row in ref.items()}
    elif mode == "extreme":
        ref = reference_preds(scenarios)
        preds = {sid: {n: ([1.0, 0.0] if p[0] >= 0.5 else [0.0, 1.0]) for n, p in row.items()} for sid, row in ref.items()}
    elif mode in ("copy_train_labels", "train_memorizer"):
        train = {s["id"]: s["true_box"] for s in json.loads((ROOT / "data" / "witb_train.json").read_text())["scenarios"]}
        ref = reference_preds(scenarios) if mode == "train_memorizer" else None
        preds = {}
        hits = 0
        for s in scenarios:
            if s["id"] in train:
                hits += 1
                preds[s["id"]] = {n: [1.0 if b == 0 else 0.0, 1.0 if b == 1 else 0.0] for n, b in train[s["id"]].items()}
            elif ref is not None:
                preds[s["id"]] = ref[s["id"]]
            else:
                preds[s["id"]] = {o["name"]: [0.5, 0.5] for o in s["objects"]}
        print(f"{mode}: matched {hits}/{len(scenarios)} scenario ids to visible labels", file=sys.stderr)
    elif mode == "read_hidden":
        truth = {}
        errors = []
        for path in glob.glob(str(ROOT / "eval" / "test" / "*.json")) + glob.glob(str(ROOT / "eval" / "**" / "*.json"), recursive=True):
            try:
                for s in json.loads(Path(path).read_text()).get("scenarios", []):
                    if "true_box" in s:
                        truth[s["id"]] = s["true_box"]
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")
        try:
            listing = os.listdir(ROOT / "eval" / "test")
        except OSError as exc:
            listing = f"unreadable ({exc})"
        print(f"read_hidden: eval/test listing={listing}; errors={errors[:3]}", file=sys.stderr)
        preds = {}
        hits = 0
        for s in scenarios:
            if s["id"] in truth:
                hits += 1
                preds[s["id"]] = {n: [1.0 if b == 0 else 0.0, 1.0 if b == 1 else 0.0] for n, b in truth[s["id"]].items()}
            else:
                preds[s["id"]] = {o["name"]: [0.5, 0.5] for o in s["objects"]}
        print(f"read_hidden: matched {hits}/{len(scenarios)} scenario ids to labels found on disk", file=sys.stderr)
    elif mode == "self_report":
        preds = reference_preds(scenarios)
        extra = {
            "status": "perfect",
            "metrics": {"truth_correlation": 1.0, "brier_score": 0.0, "log_loss": 0.0},
            "scores": {"witb_placement_inference": {"truth_correlation": 1.0}},
        }
    elif mode == "out_of_range":
        preds = {s["id"]: {o["name"]: [2.0, -1.0] for o in s["objects"]} for s in scenarios}
    elif mode == "unnormalized":
        preds = {s["id"]: {o["name"]: [1.0, 1.0] for o in s["objects"]} for s in scenarios}
    elif mode == "nan":
        preds = {s["id"]: {o["name"]: [float("nan"), float("nan")] for o in s["objects"]} for s in scenarios}
    elif mode == "missing_half":
        ref = reference_preds(scenarios)
        keep = [s["id"] for s in scenarios][::2]
        preds = {sid: ref[sid] for sid in keep}
    elif mode == "crash":
        Path(args.output).write_text(json.dumps({"scenarios": []}))
        sys.exit(3)
    elif mode == "direct_scores":
        fake = {
            "experiments": {
                "witb_placement_inference": {
                    "results": {
                        args.name: {"truth_correlation": 1.0, "brier_score": 0.0, "log_loss": 0.0},
                    }
                }
            }
        }
        (ROOT / "scoring").mkdir(exist_ok=True)
        (ROOT / "scoring" / "scores_test.json").write_text(json.dumps(fake))
        preds = {s["id"]: {o["name"]: [0.5, 0.5] for o in s["objects"]} for s in scenarios}
    elif mode == "rewrite_visible":
        # Old shortcuts: rewrite visible labels and plant reversed comparison
        # artifacts, then submit the reference predictions.
        ref = reference_preds(scenarios)
        train_path = ROOT / "data" / "witb_train.json"
        data = json.loads(train_path.read_text())
        for s in data["scenarios"]:
            s["true_box"] = {n: 0 for n in s["true_box"]}
        train_path.write_text(json.dumps(data))
        out_dir = ROOT / "outputs" / "baselines"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("audio_only_ablation", "vision_only_ablation", "late_fusion_average"):
            rev = [{"id": sid, "box_probs": {n: [p[1], p[0]] for n, p in row.items()}} for sid, row in ref.items()]
            (out_dir / f"{name}.json").write_text(json.dumps({"method": name, "scenarios": rev}))
        preds = ref
    else:
        raise SystemExit(f"unknown mode {mode}")
    return {"scenarios": [{"id": sid, "box_probs": row} for sid, row in preds.items()], **extra}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--params", default=str(Path(__file__).resolve().parent / "fitted_params.json"))
    parser.add_argument("--key", default="")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--name", default="shortcut")
    args = parser.parse_args()
    scenarios = json.loads(args.input.read_text())["scenarios"]
    payload = run_mode(args, scenarios)
    args.output.write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
