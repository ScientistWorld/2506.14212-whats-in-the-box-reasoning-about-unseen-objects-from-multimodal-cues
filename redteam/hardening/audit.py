#!/usr/bin/env python3
"""Hardening audit: run variants and shortcuts through the hidden test.

Usage:  python3 redteam/hardening/audit.py all       # fit + groups + analysis
        python3 redteam/hardening/audit.py analyze   # re-analyse saved runs

Each group installs audit predictors into method/predictors/ next to the
reference predictor, invokes ``bash scripts/evaluate_train.sh`` and/or
``bash scripts/evaluate_test.sh`` with no arguments, saves the score files,
then removes the audit predictors and restores the workspace state.  Within a
group every method sees exactly the same hidden test scenarios, so the
comparisons below are paired.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRED_DIR = ROOT / "method" / "predictors"
SRC = HERE / "predictors"
RESULTS = HERE / "results"
KEEP = ROOT / "outputs" / "hardening"
HIDDEN = ROOT / "eval" / "test" / "witb_test_hidden.json"
REFERENCE_JSON = ROOT / "scoring" / "reference.json"
EXPERIMENT = "witb_placement_inference"
REF = "full_neurosymbolic"
PROB_FLOOR = 1e-4

MAIN_GROUP = [
    "ref_seed29", "ref_seed43", "ref_samples5000",
    "better_visual_floor", "better_cue_weights", "better_weight_cue",
    "simple_trained_late_fusion",
    "weaker_audio_only", "weaker_vision_only", "weaker_late_fusion", "weaker_one_sample", "weaker_weight_only",
    "sc_constant", "sc_near_constant", "sc_extreme", "sc_copy_train_labels", "sc_train_memorizer",
    "sc_read_hidden", "sc_self_report", "sc_out_of_range", "sc_unnormalized", "sc_nan", "sc_missing_half",
    "sc_crash",
]
MUTATING_GROUP = ["sc_direct_scores", "sc_rewrite_visible"]
HONEST = [
    REF, "ref_seed29", "ref_seed43", "ref_samples5000",
    "better_visual_floor", "better_cue_weights", "better_weight_cue", "simple_trained_late_fusion",
    "weaker_audio_only", "weaker_vision_only", "weaker_late_fusion", "weaker_one_sample", "weaker_weight_only",
]
STATE_FILES = [
    ROOT / "data" / "witb_train.json",
    ROOT / "scoring" / "scores_train.json",
    ROOT / "scoring" / "scores_test.json",
]


def sh(cmd: list[str], env: dict | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})}).returncode


class Snapshot:
    def __init__(self, tag: str):
        self.dir = KEEP / "snapshots" / tag

    def __enter__(self):
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        for f in STATE_FILES:
            if f.exists():
                shutil.copy2(f, self.dir / f.name)
        bdir = ROOT / "outputs" / "baselines"
        if bdir.exists():
            shutil.copytree(bdir, self.dir / "baselines")
        return self

    def __exit__(self, *exc):
        for f in STATE_FILES:
            saved = self.dir / f.name
            if saved.exists():
                shutil.copy2(saved, f)
        bdir = ROOT / "outputs" / "baselines"
        if (self.dir / "baselines").exists():
            shutil.rmtree(bdir, ignore_errors=True)
            shutil.copytree(self.dir / "baselines", bdir)
        for p in PRED_DIR.glob("*.sh"):
            if p.name != f"{REF}.sh":
                p.unlink()
        return False


def install(names: list[str]) -> None:
    for name in names:
        shutil.copy2(SRC / f"{name}.sh", PRED_DIR / f"{name}.sh")


def save(src: Path, dst_name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, RESULTS / dst_name)


def run_groups() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = {}
    rc = sh([sys.executable, str(HERE / "variants" / "fit.py")])
    log["fit_rc"] = rc

    with Snapshot("main"):
        install(MAIN_GROUP)
        log["main_train_rc"] = sh(["bash", "scripts/evaluate_train.sh"])
        save(ROOT / "scoring" / "scores_train.json", "main_scores_train.json")
        keep = KEEP / "test_predictions"
        shutil.rmtree(keep, ignore_errors=True)
        log["main_test_rc"] = sh(["bash", "scripts/evaluate_test.sh"], {"WITB_KEEP_PREDICTIONS_DIR": str(keep)})
        save(ROOT / "scoring" / "scores_test.json", "main_scores_test.json")

    with Snapshot("mutating"):
        before = (ROOT / "data" / "witb_train.json").read_bytes()
        install(MUTATING_GROUP)
        log["mutating_test_rc"] = sh(["bash", "scripts/evaluate_test.sh"])
        save(ROOT / "scoring" / "scores_test.json", "mutating_scores_test.json")
        log["mutating_changed_visible_labels"] = (ROOT / "data" / "witb_train.json").read_bytes() != before
        log["mutating_planted_baselines"] = sorted(p.name for p in (ROOT / "outputs" / "baselines").glob("*.json"))

    with Snapshot("late_registration"):
        log["late_train_rc"] = sh(["bash", "scripts/evaluate_train.sh"])
        save(ROOT / "scoring" / "scores_train.json", "late_scores_train.json")
        install(["sc_late_added"])
        log["late_test_rc"] = sh(["bash", "scripts/evaluate_test.sh"])
        save(ROOT / "scoring" / "scores_test.json", "late_scores_test.json")

    (RESULTS / "execution.json").write_text(json.dumps(log, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_results(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())["experiments"][EXPERIMENT]


def per_scenario_stats(pred_path: Path, hidden: list[dict]) -> list[tuple] | None:
    if not pred_path.exists():
        return None
    payload = json.loads(pred_path.read_text())
    by_id = {s["id"]: s["box_probs"] for s in payload["scenarios"]}
    stats = []
    for s in hidden:
        probs = by_id[s["id"]]
        sp = sy = spp = syy = spy = 0.0
        npairs = 0
        brier = loss = 0.0
        for name, box in s["true_box"].items():
            row = probs[name]
            tot = sum(row)
            row = [v / tot for v in row]
            sq = 0.0
            for b, p in enumerate(row):
                y = 1.0 if b == box else 0.0
                sp += p
                sy += y
                spp += p * p
                syy += y * y
                spy += p * y
                npairs += 1
                sq += (p - y) ** 2
            brier += 0.5 * sq
            loss += -math.log(max(PROB_FLOOR, row[box]))
        stats.append((sp, sy, spp, syy, spy, npairs, brier, loss, len(s["true_box"])))
    return stats


def metrics_from(stats, idx) -> dict:
    sp = sy = spp = syy = spy = n = brier = loss = nobj = 0.0
    for i in idx:
        a = stats[i]
        sp += a[0]; sy += a[1]; spp += a[2]; syy += a[3]; spy += a[4]; n += a[5]
        brier += a[6]; loss += a[7]; nobj += a[8]
    vp = spp - sp * sp / n
    vy = syy - sy * sy / n
    cov = spy - sp * sy / n
    r = cov / math.sqrt(vp * vy) if vp > 0 and vy > 0 else 0.0
    return {"truth_correlation": r, "brier_score": brier / nobj, "log_loss": loss / nobj}


def analyze() -> None:
    reference = json.loads(REFERENCE_JSON.read_text())["experiments"][EXPERIMENT]["metrics"]
    coef = {m: abs(d["coefficient"]) for m, d in reference.items()}
    direction = {m: (1.0 if d["higher_is_better"] else -1.0) for m, d in reference.items()}
    metrics = list(coef)

    main_test = load_results("main_scores_test.json")
    main_train = load_results("main_scores_train.json")
    mutating = load_results("mutating_scores_test.json")
    late_test = load_results("late_scores_test.json")
    late_train = load_results("late_scores_train.json")
    if not main_test:
        raise SystemExit("no saved main-group scores; run 'all' first")

    rows = {}
    for name, vals in main_test["results"].items():
        rows[name] = {"test": vals, "status": main_test["_diagnostics"].get(name, {})}
        if main_train:
            rows[name]["train"] = main_train["results"].get(name)
    for name, vals in mutating.get("results", {}).items():
        if name != REF:
            rows[name] = {"test": vals, "status": mutating["_diagnostics"].get(name, {}), "group": "mutating"}
    if late_test:
        rows["sc_late_added"] = {
            "test": late_test["results"].get("sc_late_added"),
            "status": late_test["_diagnostics"].get("sc_late_added", {}),
            "group": "late_registration",
            "in_train_results": "sc_late_added" in late_train.get("results", {}),
            "in_test_results": "sc_late_added" in late_test.get("results", {}),
        }

    ref_vals = main_test["results"][REF]
    honest_ok = [n for n in HONEST if n in main_test["results"] and main_test["_diagnostics"].get(n, {}).get("status") == "ok"]
    sigma = {}
    for m in metrics:
        vals = [main_test["results"][n][m] for n in honest_ok]
        mu = sum(vals) / len(vals)
        sigma[m] = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
    all_ok = [n for n, d in main_test["_diagnostics"].items() if d.get("status") == "ok"]
    sigma_all = {}
    for m in metrics:
        vals = [main_test["results"][n][m] for n in all_ok]
        mu = sum(vals) / len(vals)
        sigma_all[m] = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0

    def aggregate(vals, sig):
        if vals is None:
            return None
        return sum(coef[m] * direction[m] * (vals[m] - ref_vals[m]) / sig[m] for m in metrics)

    for name, row in rows.items():
        row["aggregate_delta_honest_sigma"] = aggregate(row["test"], sigma)
        row["aggregate_delta_all_sigma"] = aggregate(row["test"], sigma_all)

    # Paired bootstrap over hidden scenarios (same inputs for every method).
    hidden = json.loads(HIDDEN.read_text())["scenarios"]
    keep = KEEP / "test_predictions"
    ref_stats = per_scenario_stats(keep / f"{REF}.json", hidden)
    rng = random.Random(12345)
    n = len(hidden)
    boots = [[rng.randrange(n) for _ in range(n)] for _ in range(1000)]
    paired = {}
    if ref_stats is not None:
        ref_boot = [metrics_from(ref_stats, idx) for idx in boots]
        for name in sorted(main_test["results"]):
            if name == REF:
                continue
            stats = per_scenario_stats(keep / f"{name}.json", hidden)
            if stats is None:
                continue
            deltas = {m: [] for m in metrics}
            agg = []
            for idx, rb in zip(boots, ref_boot):
                mb = metrics_from(stats, idx)
                for m in metrics:
                    deltas[m].append(mb[m] - rb[m])
                agg.append(sum(coef[m] * direction[m] * (mb[m] - rb[m]) / sigma[m] for m in metrics))
            entry = {}
            for m in metrics + ["aggregate"]:
                vals = sorted(agg if m == "aggregate" else deltas[m])
                entry[m] = {"lo95": vals[24], "hi95": vals[974], "mean": sum(vals) / len(vals)}
            paired[name] = entry

    seeds = [n for n in (REF, "ref_seed29", "ref_seed43") if n in main_test["results"]]
    seed_spread = {
        m: max(main_test["results"][s][m] for s in seeds) - min(main_test["results"][s][m] for s in seeds)
        for m in metrics
    }
    seed_spread["aggregate"] = max(rows.get(s, {}).get("aggregate_delta_honest_sigma") or 0.0 for s in seeds) - min(
        rows.get(s, {}).get("aggregate_delta_honest_sigma") or 0.0 for s in seeds
    )

    summary = {
        "reference": REF,
        "reference_test": ref_vals,
        "coefficients": coef,
        "sigma_honest_cohort": sigma,
        "honest_cohort": honest_ok,
        "sigma_all_successful": sigma_all,
        "seed_spread_test": seed_spread,
        "fixed_baselines_test": main_test.get("_baselines", {}),
        "fixed_baselines_train": main_train.get("_baselines", {}) if main_train else {},
        "rows": rows,
        "paired_bootstrap_vs_reference": paired,
        "execution": json.loads((RESULTS / "execution.json").read_text()) if (RESULTS / "execution.json").exists() else {},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["| method | status | r | Brier | log loss | train r | aggregate Δ | paired Δ 95% CI (aggregate) |", "|---|---|---:|---:|---:|---:|---:|---|"]
    order = sorted(
        rows,
        key=lambda k: -(rows[k]["aggregate_delta_honest_sigma"] if rows[k]["aggregate_delta_honest_sigma"] is not None else -99),
    )
    for name in order:
        row = rows[name]
        t = row["test"] or {}
        tr = row.get("train") or {}
        ci = paired.get(name, {}).get("aggregate")
        ci_s = f"[{ci['lo95']:+.3f}, {ci['hi95']:+.3f}]" if ci else ""
        lines.append(
            f"| {name} | {row['status'].get('status', '?')} | {t.get('truth_correlation', float('nan')):.4f} | "
            f"{t.get('brier_score', float('nan')):.4f} | {t.get('log_loss', float('nan')):.4f} | "
            f"{tr.get('truth_correlation', float('nan')):.4f} | {row['aggregate_delta_honest_sigma']:+.3f} | {ci_s} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"seed_spread": seed_spread, "sigma": sigma}, indent=2))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "all":
        run_groups()
        analyze()
    elif cmd == "analyze":
        analyze()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
