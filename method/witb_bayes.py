#!/usr/bin/env python3
"""Bayesian hypothesis engine for the What-is-in-the-Box task.

This implements the paper's symbolic inference step: enumerate all placements
of objects into boxes, score each hypothesis by visual fit and audio evidence,
normalize the posterior, and marginalize to object-in-box probabilities.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path


EPS = 1e-12


def positive_normal(rng: random.Random, mean: float, std: float) -> float:
    return max(0.1, rng.gauss(mean, std))


def orientations(dims: list[float]) -> list[tuple[float, float, float]]:
    return sorted(set(itertools.permutations(tuple(dims), 3)))


def fits_single(item: tuple[float, float, float], box: tuple[float, float, float]) -> bool:
    return any(all(o[i] <= box[i] + 1e-9 for i in range(3)) for o in orientations(list(item)))


def split_free_box(free_box: tuple[float, float, float], item: tuple[float, float, float]):
    fw, fh, fd = free_box
    iw, ih, id_ = item
    return [
        (fw - iw, fh, fd),
        (iw, fh - ih, fd),
        (iw, ih, fd - id_),
    ]


def can_pack_axis_aligned(items: list[tuple[float, float, float]], box: tuple[float, float, float]) -> bool:
    """Small exact-enough backtracking packer for 2-5 household objects.

    The paper describes rejection sampling over whether sampled objects fit in a
    sampled box. This predicate supplies that fit check while allowing all
    object orientations and simple recursive free-space splitting.
    """
    if not items:
        return True
    if sum(math.prod(x) for x in items) > math.prod(box) + 1e-9:
        return False
    if any(not fits_single(x, box) for x in items):
        return False
    ordered = sorted(items, key=math.prod, reverse=True)

    def rec(remaining, free_boxes):
        if not remaining:
            return True
        item = remaining[0]
        for item_o in orientations(list(item)):
            for idx, free in enumerate(free_boxes):
                if all(item_o[d] <= free[d] + 1e-9 for d in range(3)):
                    new_free = free_boxes[:idx] + free_boxes[idx + 1 :]
                    new_free.extend(x for x in split_free_box(free, item_o) if min(x) > 1e-9)
                    new_free = sorted(new_free, key=math.prod, reverse=True)[:8]
                    if rec(remaining[1:], new_free):
                        return True
        return False

    return rec(ordered, [box])


def enumerate_hypotheses(object_names: list[str], num_boxes: int) -> list[list[list[str]]]:
    hypotheses = []
    for assignment in itertools.product(range(num_boxes), repeat=len(object_names)):
        if len(set(assignment)) != num_boxes:
            continue
        boxes = [[] for _ in range(num_boxes)]
        for obj, box_idx in zip(object_names, assignment):
            boxes[box_idx].append(obj)
        hypotheses.append(boxes)
    return hypotheses


def visual_fit_probability(scenario: dict, box_idx: int, object_names: list[str], samples: int, rng: random.Random) -> float:
    if not object_names:
        return 1.0
    obj_by_name = {obj["name"]: obj for obj in scenario["objects"]}
    box_info = scenario["boxes"][box_idx]
    hits = 0
    for _ in range(samples):
        box_dims = tuple(
            positive_normal(rng, m, s)
            for m, s in zip(box_info["dims_cm"], box_info["dims_std_cm"])
        )
        item_dims = []
        for name in object_names:
            obj = obj_by_name[name]
            rigidity = float(obj.get("rigidity", 1.0))
            compress = 0.75 + 0.25 * rigidity
            sampled = [
                positive_normal(rng, m, s) * compress
                for m, s in zip(obj["dims_cm"], obj["dims_std_cm"])
            ]
            item_dims.append(tuple(sampled))
        hits += int(can_pack_axis_aligned(item_dims, box_dims))
    return (hits + 1.0) / (samples + 2.0)


def score_scenario(scenario: dict, samples: int, seed: int, mode: str) -> dict:
    rng = random.Random(seed)
    object_names = [obj["name"] for obj in scenario["objects"]]
    hypotheses = enumerate_hypotheses(object_names, len(scenario["boxes"]))
    visual_cache = {}
    log_weights = []
    for hyp in hypotheses:
        log_w = 0.0
        for box_idx, names in enumerate(hyp):
            key = (box_idx, tuple(sorted(names)))
            if key not in visual_cache:
                visual_cache[key] = visual_fit_probability(scenario, box_idx, list(names), samples, rng)
            visual = visual_cache[key]
            audio = 1.0
            for name in names:
                audio *= max(EPS, scenario["audio_posteriors"][box_idx][name])
            if mode == "full":
                evidence = visual * audio
            elif mode == "vision":
                evidence = visual
            elif mode == "audio":
                evidence = audio
            else:
                raise ValueError(f"unknown mode: {mode}")
            log_w += math.log(max(EPS, evidence))
        log_weights.append(log_w)
    max_log = max(log_weights)
    weights = [math.exp(w - max_log) for w in log_weights]
    norm = sum(weights)
    weights = [w / norm for w in weights]
    marginals = {name: [0.0 for _ in scenario["boxes"]] for name in object_names}
    for hyp, weight in zip(hypotheses, weights):
        for box_idx, names in enumerate(hyp):
            for name in names:
                marginals[name][box_idx] += weight
    return {
        "scenario_id": scenario["id"],
        "mode": mode,
        "pred_left_prob": {name: probs[0] for name, probs in marginals.items()},
        "num_hypotheses": len(hypotheses),
    }


def run(data_path: Path, output_path: Path, mode: str, samples: int, seed: int) -> None:
    data = json.loads(data_path.read_text())
    predictions = {
        "method": f"witb_{mode}",
        "mode": mode,
        "data_path": str(data_path),
        "samples": samples,
        "seed": seed,
        "scenarios": [
            score_scenario(scenario, samples=samples, seed=seed + idx * 1009, mode=mode)
            for idx, scenario in enumerate(data["scenarios"])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/witb_parsed_benchmark.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "vision", "audio"], default="full")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run(args.data, args.output, args.mode, args.samples, args.seed)


if __name__ == "__main__":
    main()
