#!/usr/bin/env python3
"""Fit the audit variants' parameters on the visible train split only.

Every fitted variant keeps the paper's hypothesis enumeration, rejection-
sampled visual fit and audio evidence; it only adds cue reliability exponents,
a floor on the visual fit probability and (optionally) a likelihood for the
lifted-weight cue.  Parameters minimise train log loss on true placements.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "method"))
sys.path.insert(0, str(HERE))

import witb_bayes as wb  # noqa: E402
from witb_variants import audio_only, cache_dir, generalized_posterior, logit  # noqa: E402

FLOOR = 1e-4
STACK_CLIP = 0.005  # probabilities are clipped to [clip, 1 - clip] before the logit


def clipped_logit(p: float) -> float:
    return logit(min(1.0 - STACK_CLIP, max(STACK_CLIP, p)))


def train_log_loss(scenarios, tables, params) -> float:
    total, count = 0.0, 0
    for s, t in zip(scenarios, tables):
        probs = generalized_posterior(s, t, params)
        for name, box in s["true_box"].items():
            total += -math.log(max(FLOOR, probs[name][box]))
            count += 1
    return total / count


def grid(scenarios, tables, base, axes):
    best = None
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        params = {**base, **dict(zip(keys, values))}
        loss = train_log_loss(scenarios, tables, params)
        if best is None or loss < best[0]:
            best = (loss, params)
    return best


def fit_logistic(rows, l2: float = 1.0):
    """rows: (x_audio, x_vision, y). L2-regularised Newton-Raphson logistic regression."""
    w = [0.0, 1.0, 1.0]
    for _ in range(100):
        g = [l2 * wi if i else 0.0 for i, wi in enumerate(w)]
        h = [[(l2 if (i == j and i) else 0.0) + (1e-9 if i == j else 0.0) for j in range(3)] for i in range(3)]
        for xa, xv, y in rows:
            x = (1.0, xa, xv)
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            for i in range(3):
                g[i] += (p - y) * x[i]
                for j in range(3):
                    h[i][j] += p * (1 - p) * x[i] * x[j]
        a = [row[:] + [gi] for row, gi in zip(h, g)]
        for c in range(3):
            piv = max(range(c, 3), key=lambda r: abs(a[r][c]))
            a[c], a[piv] = a[piv], a[c]
            for r in range(3):
                if r != c:
                    f = a[r][c] / a[c][c]
                    for cc in range(c, 4):
                        a[r][cc] -= f * a[c][cc]
        d = [a[i][3] / a[i][i] for i in range(3)]
        step = max(abs(di) for di in d)
        scale = min(1.0, 2.0 / step) if step > 0 else 1.0
        w = [wi - scale * di for wi, di in zip(w, d)]
        if step < 1e-9:
            break
    return {"bias": w[0], "w_audio": w[1], "w_vision": w[2], "l2": l2, "logit_clip": STACK_CLIP}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "witb_train.json")
    parser.add_argument("--out", type=Path, default=HERE / "fitted_params.json")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    scenarios = data["scenarios"]
    tables = wb.visual_tables(scenarios, 1000, 17, cache_dir())
    out = {}
    ref_loss = train_log_loss(scenarios, tables, {})
    out["_train_log_loss_reference"] = ref_loss

    eps_axis = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    gamma_axis = [0.4, 0.6, 0.8, 1.0]
    beta_axis = [0.5, 0.7, 0.85, 1.0, 1.2]

    loss, params = grid(scenarios, tables, {"beta": 1.0}, {"eps": eps_axis, "gamma": gamma_axis})
    out["visual_floor"] = {**params, "_train_log_loss": loss}

    loss, params = grid(scenarios, tables, {}, {"eps": eps_axis, "gamma": gamma_axis, "beta": beta_axis})
    out["cue_weights"] = {**params, "_train_log_loss": loss}

    base = {k: params[k] for k in ("eps", "gamma", "beta")}
    loss, wparams = grid(
        scenarios,
        tables,
        base,
        {"sigma_w": [0.15, 0.25, 0.35, 0.5, 0.7], "tare": [0.0, 0.00003, 0.00006, 0.0001]},
    )
    loss, wparams = grid(
        scenarios,
        tables,
        {"sigma_w": wparams["sigma_w"], "tare": wparams["tare"]},
        {
            "eps": sorted({max(0.0, base["eps"] / 2), base["eps"], min(0.2, base["eps"] * 2 + 0.002)}),
            "gamma": sorted({max(0.2, base["gamma"] - 0.2), base["gamma"], min(1.2, base["gamma"] + 0.2)}),
            "beta": sorted({max(0.3, base["beta"] - 0.15), base["beta"], base["beta"] + 0.15}),
        },
    )
    out["weight_cue"] = {**wparams, "_train_log_loss": loss}

    loss, oparams = grid(
        scenarios,
        tables,
        {"use_vision": False, "use_audio": False},
        {"sigma_w": [0.15, 0.25, 0.35, 0.5, 0.7], "tare": [0.0, 0.00003, 0.00006, 0.0001]},
    )
    out["weight_only"] = {**oparams, "_train_log_loss": loss}

    vision = {p["id"]: p["box_probs"] for p in wb.predict(scenarios, "vision", 1000, 17, cache_dir())}
    rows = []
    for s in scenarios:
        aud = audio_only(s)
        for name, box in s["true_box"].items():
            rows.append((clipped_logit(aud[name][0]), clipped_logit(vision[s["id"]][name][0]), 1.0 if box == 0 else 0.0))
    out["trained_late_fusion"] = fit_logistic(rows)

    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
