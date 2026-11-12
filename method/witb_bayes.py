#!/usr/bin/env python3
"""Bayesian hypothesis engine for the What's-in-the-Box task (paper method).

Implements the symbolic inference stage of Ying, Xu et al. (2025):

1. enumerate every placement of the N objects into the K boxes with no empty
   box (|H| = K! S(N, K) hypotheses), with a uniform prior;
2. visual evidence P(O_i | H_i): rejection sampling -- sample object and box
   dimensions from the parsed normal distributions 1000 times and count how
   often the hypothesised set of objects physically fits in box i;
3. audio evidence P(H_i | A_i) = prod_{o in H_i} P(o | A_i) from the per-box
   zero-shot audio-classifier posteriors (paper Eq. 2);
4. posterior P(H | O, A) ∝ prod_i P(O_i | H_i) P(H_i | A_i) (paper Eq. 1),
   normalised and marginalised into per-object box probabilities.

The parsed weight, rigidity, material and lifted-weight fields are not used by
the paper's equations and are ignored here.

Input:  {"scenarios": [{"id", "objects", "boxes", "audio_posteriors", ...}]}
Output: {"method", "scenarios": [{"id", "box_probs": {object: [p_box0, ...]}}]}
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

EPS = 1e-12
MIN_DIM_CM = 0.1
CACHE_VERSION = "witb-visual-v2"


# ---------------------------------------------------------------------------
# Physical fit predicate used inside rejection sampling.
# ---------------------------------------------------------------------------


def _orientations(dims: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    return sorted(set(itertools.permutations(dims, 3)))


def _fits_single(item_sorted: tuple[float, float, float], box_sorted: tuple[float, float, float]) -> bool:
    return (
        item_sorted[0] <= box_sorted[0] + 1e-9
        and item_sorted[1] <= box_sorted[1] + 1e-9
        and item_sorted[2] <= box_sorted[2] + 1e-9
    )


def can_pack_axis_aligned(items: list[tuple[float, float, float]], box: tuple[float, float, float]) -> bool:
    """Backtracking axis-aligned guillotine packer for a handful of objects.

    Every object may be rotated into any of its six axis-aligned orientations.
    After placing an object in a free region the remaining space is split into
    three guillotine sub-regions; at most eight free regions are tracked.
    """
    if not items:
        return True
    if sum(math.prod(x) for x in items) > math.prod(box) + 1e-9:
        return False
    box_sorted = tuple(sorted(box))
    if any(not _fits_single(tuple(sorted(x)), box_sorted) for x in items):
        return False
    if len(items) == 1:
        return True
    ordered = sorted(items, key=math.prod, reverse=True)
    orients = [_orientations(tuple(x)) for x in ordered]

    def rec(k: int, free_boxes: list[tuple[float, float, float]]) -> bool:
        if k == len(ordered):
            return True
        for o in orients[k]:
            for idx, free in enumerate(free_boxes):
                if o[0] <= free[0] + 1e-9 and o[1] <= free[1] + 1e-9 and o[2] <= free[2] + 1e-9:
                    new_free = free_boxes[:idx] + free_boxes[idx + 1:]
                    fw, fh, fd = free
                    for split in ((fw - o[0], fh, fd), (o[0], fh - o[1], fd), (o[0], o[1], fd - o[2])):
                        if min(split) > 1e-9:
                            new_free.append(split)
                    new_free = sorted(new_free, key=math.prod, reverse=True)[:8]
                    if rec(k + 1, new_free):
                        return True
        return False

    return rec(0, [tuple(box)])


# ---------------------------------------------------------------------------
# Hypotheses and evidence.
# ---------------------------------------------------------------------------


def enumerate_hypotheses(num_objects: int, num_boxes: int) -> list[tuple[int, ...]]:
    """All assignments object -> box that leave no box empty (K! S(N, K))."""
    return [
        assign
        for assign in itertools.product(range(num_boxes), repeat=num_objects)
        if len(set(assign)) == num_boxes
    ]


def _sample_dims(rng: random.Random, mean: list[float], std: list[float]) -> tuple[float, float, float]:
    return tuple(max(MIN_DIM_CM, rng.gauss(float(m), float(s))) for m, s in zip(mean, std))


def visual_fit_table(scenario: dict, samples: int, seed: int) -> dict[str, float]:
    """P(objects in subset fit in box b) for every (box, subset) a hypothesis needs.

    Dimensions are drawn ``samples`` times from the parsed normal
    distributions; each draw checks whether the subset fits (rejection
    sampling).  All subsets of one scenario share the same draws.
    """
    rng = random.Random(f"{seed}:{scenario['id']}")
    objects = scenario["objects"]
    boxes = scenario["boxes"]
    n, k = len(objects), len(boxes)
    draws = []
    for _ in range(samples):
        obj_dims = [_sample_dims(rng, o["dims_cm"], o["dims_std_cm"]) for o in objects]
        box_dims = [_sample_dims(rng, b["dims_cm"], b["dims_std_cm"]) for b in boxes]
        draws.append((obj_dims, box_dims))
    needed = set()
    for assign in enumerate_hypotheses(n, k):
        for b in range(k):
            needed.add((b, tuple(i for i in range(n) if assign[i] == b)))
    table = {}
    for b, subset in sorted(needed):
        hits = 0
        for obj_dims, box_dims in draws:
            hits += can_pack_axis_aligned([obj_dims[i] for i in subset], box_dims[b])
        table[_key(b, subset)] = hits / max(1, samples)
    return table


def _key(box_idx: int, subset: tuple[int, ...]) -> str:
    return f"{box_idx}|" + ",".join(str(i) for i in subset)


def audio_term(scenario: dict, box_idx: int, subset: tuple[int, ...]) -> float:
    """Paper Eq. 2: P(H_i | A_i) = prod_{o in H_i} P(o | A_i)."""
    post = scenario["audio_posteriors"][box_idx]
    value = 1.0
    for i in subset:
        value *= max(EPS, float(post[scenario["objects"][i]["name"]]))
    return value


def posterior_box_probs(scenario: dict, visual: dict[str, float], mode: str = "full") -> dict[str, list[float]]:
    objects = scenario["objects"]
    n, k = len(objects), len(scenario["boxes"])
    hyps = enumerate_hypotheses(n, k)
    log_w = []
    for assign in hyps:
        total = 0.0
        for b in range(k):
            subset = tuple(i for i in range(n) if assign[i] == b)
            evidence = 1.0
            if mode in ("full", "vision"):
                evidence *= visual[_key(b, subset)]
            if mode == "full":
                evidence *= audio_term(scenario, b, subset)
            total += math.log(max(EPS, evidence))
        log_w.append(total)
    top = max(log_w)
    weights = [math.exp(w - top) for w in log_w]
    z = sum(weights)
    probs = {o["name"]: [0.0] * k for o in objects}
    for assign, w in zip(hyps, weights):
        for i, b in enumerate(assign):
            probs[objects[i]["name"]][b] += w / z
    return probs


# ---------------------------------------------------------------------------
# Batch driver with an optional on-disk cache of visual tables.
# ---------------------------------------------------------------------------


def _scenario_hash(scenario: dict, samples: int, seed: int) -> str:
    payload = {
        "v": CACHE_VERSION,
        "samples": samples,
        "seed": seed,
        "id": scenario["id"],
        "objects": [[o["name"], o["dims_cm"], o["dims_std_cm"]] for o in scenario["objects"]],
        "boxes": [[b["dims_cm"], b["dims_std_cm"]] for b in scenario["boxes"]],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _visual_job(args):
    scenario, samples, seed = args
    return visual_fit_table(scenario, samples, seed)


def _workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover
        return max(1, os.cpu_count() or 1)


def visual_tables(scenarios: list[dict], samples: int, seed: int, cache_dir: Path | None) -> list[dict[str, float]]:
    cache = {}
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"visual_s{samples}_seed{seed}.json"
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                cache = {}
    hashes = [_scenario_hash(s, samples, seed) for s in scenarios]
    todo = [i for i, h in enumerate(hashes) if h not in cache]
    if todo:
        jobs = [(scenarios[i], samples, seed) for i in todo]
        workers = _workers()
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                tables = list(pool.map(_visual_job, jobs, chunksize=4))
        else:
            tables = [_visual_job(j) for j in jobs]
        for i, table in zip(todo, tables):
            cache[hashes[i]] = table
        if cache_path is not None:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(cache_path)
    return [cache[h] for h in hashes]


def predict(scenarios: list[dict], mode: str, samples: int, seed: int, cache_dir: Path | None) -> list[dict]:
    if mode in ("full", "vision"):
        tables = visual_tables(scenarios, samples, seed, cache_dir)
    else:
        tables = [{} for _ in scenarios]
    out = []
    for scenario, table in zip(scenarios, tables):
        out.append({"id": scenario["id"], "box_probs": posterior_box_probs(scenario, table, mode)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="scenario JSON (labels, if present, are ignored)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "vision"], default="full")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    scenarios = data["scenarios"]
    preds = predict(scenarios, args.mode, args.samples, args.seed, args.cache_dir)
    payload = {
        "method": f"witb_{args.mode}",
        "samples": args.samples,
        "seed": args.seed,
        "scenarios": preds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
