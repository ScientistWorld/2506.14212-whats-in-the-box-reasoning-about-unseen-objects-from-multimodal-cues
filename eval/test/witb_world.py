#!/usr/bin/env python3
"""Hidden What's-in-the-Box world simulator.

This module lives only in eval/test/.  It defines the latent world that
produces both the visible train scenarios (written once to data/) and the
hidden test scenarios (written to eval/test/).  A scenario has 2-5 household
objects hidden in two boxes, as in the paper.  The ground truth is the actual
placement of every object.  The observations mirror the paper's parsed neural
outputs:

* a language-model style parse of each object name (dimension means and
  standard deviations, weight, material, rigidity); the parse is fixed per
  object type and is systematically biased / mis-calibrated, as an LLM's
  guesses would be;
* a vision-model style parse of each box (inner dimensions with an
  uncertainty), plus an estimate of how heavy each box looks when it is lifted
  and shaken;
* a CLAP-style zero-shot audio posterior over the scenario's object labels for
  the sound of each box being shaken.  Sounds come from the box's actual
  contents: loud objects mask quiet ones, tightly packed boxes rattle less,
  soft objects are nearly silent, and the classifier has label-specific
  text/audio misalignment and bias.

Nothing in this file is imported by method code.  Only eval/test/ code may use
it.  Use ``python3 eval/test/witb_world.py --help`` to rebuild the splits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

WORLD_SEED = 61813
TRAIN_SEED = 7_000_001
TEST_SEED = 9_000_017

TIMBRES = [
    "metal_ring",
    "ceramic_clatter",
    "glass_clink",
    "wood_knock",
    "plastic_rattle",
    "crinkle",
    "slosh",
    "jingle",
    "electronic_rattle",
    "soft_thump",
    "rubber_bounce",
    "granular_rattle",
]
T_INDEX = {name: i for i, name in enumerate(TIMBRES)}

# name, typical dims (cm), size variability (cv), weight (kg), rigidity,
# material, timbre mixture, loudness when shaken
CATALOG = [
    ("yoga mat", (62, 13, 13), 0.08, 1.2, 0.55, "foam rubber", {"soft_thump": 0.7, "rubber_bounce": 0.5}, 0.35),
    ("boxed laptop", (38, 27, 6), 0.06, 2.2, 0.97, "cardboard and electronics", {"electronic_rattle": 0.6, "wood_knock": 0.3, "plastic_rattle": 0.3}, 0.6),
    ("pillow", (50, 35, 12), 0.15, 0.6, 0.25, "fabric and foam", {"soft_thump": 1.0}, 0.12),
    ("water jug", (30, 18, 18), 0.10, 3.5, 0.85, "plastic and water", {"slosh": 0.9, "plastic_rattle": 0.3}, 0.7),
    ("water bottle", (24, 7, 7), 0.08, 0.55, 0.8, "plastic and water", {"slosh": 0.8, "plastic_rattle": 0.5}, 0.55),
    ("coins", (8, 6, 3), 0.30, 0.15, 0.95, "metal", {"jingle": 1.0, "metal_ring": 0.5}, 1.0),
    ("ceramic mug", (12, 10, 9), 0.10, 0.35, 0.98, "ceramic", {"ceramic_clatter": 1.0, "glass_clink": 0.2}, 0.7),
    ("hardcover book", (24, 17, 3.5), 0.12, 0.8, 0.97, "paper", {"wood_knock": 0.5, "soft_thump": 0.4, "crinkle": 0.2}, 0.45),
    ("scarf", (30, 20, 5), 0.30, 0.2, 0.1, "fabric", {"soft_thump": 0.6, "crinkle": 0.1}, 0.06),
    ("dinner plate", (27, 27, 2.5), 0.06, 0.6, 0.99, "ceramic", {"ceramic_clatter": 0.9, "glass_clink": 0.4}, 0.75),
    ("metal spoon", (18, 4, 2), 0.10, 0.05, 0.99, "metal", {"metal_ring": 1.0, "jingle": 0.3}, 0.6),
    ("bath towel", (35, 25, 10), 0.20, 0.6, 0.2, "fabric", {"soft_thump": 0.8}, 0.1),
    ("flashlight", (20, 4, 4), 0.15, 0.25, 0.95, "metal and plastic", {"plastic_rattle": 0.5, "metal_ring": 0.4, "electronic_rattle": 0.3}, 0.5),
    ("keys", (9, 5, 2), 0.20, 0.08, 0.97, "metal", {"jingle": 0.9, "metal_ring": 0.6}, 0.85),
    ("sponge", (11, 7, 3), 0.10, 0.02, 0.3, "foam", {"soft_thump": 0.5}, 0.05),
    ("glass jar", (14, 9, 9), 0.10, 0.4, 0.99, "glass", {"glass_clink": 1.0, "metal_ring": 0.2}, 0.7),
    ("wooden spoon", (30, 6, 2), 0.08, 0.07, 0.99, "wood", {"wood_knock": 1.0}, 0.45),
    ("cutting board", (35, 25, 2), 0.08, 0.9, 0.99, "wood", {"wood_knock": 1.0}, 0.55),
    ("tennis ball can", (21, 7, 7), 0.05, 0.2, 0.9, "plastic and rubber", {"rubber_bounce": 0.8, "plastic_rattle": 0.4}, 0.55),
    ("basketball", (24, 24, 24), 0.03, 0.6, 0.7, "rubber", {"rubber_bounce": 1.0}, 0.5),
    ("bag of LEGO bricks", (20, 12, 8), 0.25, 0.4, 0.7, "plastic", {"granular_rattle": 0.9, "plastic_rattle": 0.6}, 0.9),
    ("pill bottle", (8, 4, 4), 0.10, 0.06, 0.95, "plastic", {"granular_rattle": 1.0, "plastic_rattle": 0.4}, 0.8),
    ("bag of rice", (20, 12, 6), 0.10, 1.0, 0.5, "paper and grain", {"granular_rattle": 0.7, "crinkle": 0.5}, 0.5),
    ("smartphone", (15, 7.5, 0.8), 0.05, 0.18, 0.99, "glass and electronics", {"electronic_rattle": 0.6, "plastic_rattle": 0.3, "glass_clink": 0.2}, 0.3),
    ("TV remote", (20, 5, 2.5), 0.10, 0.15, 0.97, "plastic", {"plastic_rattle": 0.7, "electronic_rattle": 0.5}, 0.5),
    ("headphone case", (19, 17, 8), 0.10, 0.3, 0.85, "plastic", {"plastic_rattle": 0.6, "soft_thump": 0.3}, 0.35),
    ("stuffed animal", (30, 20, 15), 0.25, 0.3, 0.2, "fabric", {"soft_thump": 0.9}, 0.08),
    ("pair of sneakers", (32, 20, 12), 0.08, 0.8, 0.45, "rubber and fabric", {"soft_thump": 0.6, "rubber_bounce": 0.5}, 0.3),
    ("frying pan", (45, 28, 6), 0.08, 1.4, 1.0, "metal", {"metal_ring": 1.0, "ceramic_clatter": 0.2}, 0.75),
    ("metal water bottle", (26, 7.5, 7.5), 0.06, 0.35, 0.98, "metal", {"metal_ring": 0.9, "slosh": 0.1}, 0.7),
    ("wine glass", (21, 9, 9), 0.06, 0.2, 0.99, "glass", {"glass_clink": 1.0}, 0.55),
    ("paperback novel", (18, 11, 3), 0.10, 0.3, 0.9, "paper", {"crinkle": 0.5, "soft_thump": 0.5, "wood_knock": 0.2}, 0.3),
    ("spiral notebook", (28, 21, 1.5), 0.08, 0.4, 0.8, "paper", {"crinkle": 0.7, "metal_ring": 0.1}, 0.25),
    ("cereal box", (30, 20, 7), 0.08, 0.5, 0.7, "cardboard and grain", {"granular_rattle": 0.6, "crinkle": 0.6}, 0.6),
    ("six-pack of soda cans", (20, 13, 12), 0.05, 2.2, 0.9, "aluminum and liquid", {"slosh": 0.6, "metal_ring": 0.6}, 0.7),
    ("hammer", (32, 12, 3), 0.07, 0.6, 1.0, "metal and wood", {"metal_ring": 0.6, "wood_knock": 0.6}, 0.7),
    ("screwdriver set", (20, 10, 4), 0.10, 0.35, 0.98, "metal and plastic", {"jingle": 0.4, "metal_ring": 0.5, "plastic_rattle": 0.5}, 0.75),
    ("folded umbrella", (28, 6, 6), 0.10, 0.35, 0.6, "fabric and metal", {"soft_thump": 0.4, "metal_ring": 0.3}, 0.2),
    ("candle in a jar", (10, 8, 8), 0.10, 0.45, 0.99, "glass and wax", {"glass_clink": 0.8, "wood_knock": 0.2}, 0.5),
    ("jar of marbles", (12, 9, 9), 0.10, 0.8, 0.99, "glass", {"glass_clink": 0.7, "granular_rattle": 0.7}, 0.95),
    ("board game", (40, 27, 6), 0.08, 1.2, 0.9, "cardboard, plastic and wood", {"granular_rattle": 0.4, "plastic_rattle": 0.4, "wood_knock": 0.3, "crinkle": 0.2}, 0.6),
    ("plastic food container", (18, 12, 7), 0.10, 0.12, 0.8, "plastic", {"plastic_rattle": 1.0}, 0.5),
    ("folded sweater", (32, 25, 7), 0.15, 0.5, 0.2, "fabric", {"soft_thump": 0.8}, 0.07),
    ("baseball", (7.5, 7.5, 7.5), 0.03, 0.15, 0.95, "leather", {"wood_knock": 0.3, "soft_thump": 0.5, "rubber_bounce": 0.3}, 0.45),
    ("computer mouse", (11, 6, 4), 0.08, 0.1, 0.98, "plastic", {"plastic_rattle": 0.6, "electronic_rattle": 0.5}, 0.4),
    ("bottle of olive oil", (28, 7, 7), 0.06, 0.9, 0.99, "glass and liquid", {"glass_clink": 0.6, "slosh": 0.6}, 0.6),
    ("desk lamp", (40, 15, 15), 0.12, 1.1, 0.9, "metal and plastic", {"metal_ring": 0.4, "plastic_rattle": 0.4, "electronic_rattle": 0.3}, 0.45),
    ("box of crayons", (12, 9, 2), 0.10, 0.1, 0.95, "wax and cardboard", {"granular_rattle": 0.5, "crinkle": 0.4}, 0.45),
    ("bag of chips", (30, 20, 8), 0.10, 0.2, 0.1, "plastic film", {"crinkle": 1.0}, 0.5),
    ("rubber duck", (9, 8, 8), 0.10, 0.05, 0.5, "rubber", {"rubber_bounce": 0.7, "soft_thump": 0.3}, 0.2),
]

# Box styles: inner dimensions in cm.  As in the paper's stimuli, a scenario
# usually pairs a box that can hold most objects with one that may only hold
# one or two small objects.
LARGE_BOX_STYLES = [
    (45, 35, 30),
    (75, 30, 15),
    (50, 40, 10),
    (40, 25, 20),
    (30, 30, 30),
    (55, 38, 28),
]
SMALL_BOX_STYLES = [
    (22, 16, 12),
    (18, 18, 18),
    (25, 18, 10),
    (30, 22, 8),
    (26, 26, 14),
    (33, 20, 12),
    (35, 25, 15),
]
P_LARGE_SMALL_PAIR = 0.7

CARDBOARD_KG_PER_CM2 = 0.00005
AUDIO_TEMPERATURE = 0.13
EPS = 1e-9


def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n > 0 else vec


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class World:
    """Latent object-type parameters shared by every scenario."""

    def __init__(self, seed: int = WORLD_SEED):
        rng = random.Random(seed)
        self.types = []
        for idx, (name, dims, cv, weight, rigidity, material, timbre, loud) in enumerate(CATALOG):
            base = [0.0] * len(TIMBRES)
            for key, val in timbre.items():
                base[T_INDEX[key]] = val
            base = _norm(base)
            idio = _norm([abs(rng.gauss(0.0, 1.0)) for _ in TIMBRES])
            sound = [b + 0.15 * i for b, i in zip(base, idio)]
            text_gap = [abs(rng.gauss(0.0, 0.30)) for _ in TIMBRES]
            text_emb = _norm([s + g for s, g in zip(sound, text_gap)])
            label_bias = rng.gauss(0.0, 0.35)
            # Language-model parse: fixed per type, biased and mis-calibrated.
            dim_bias = [math.exp(rng.gauss(0.0, 0.12)) for _ in dims]
            parse_dims = [round(d * b, 1) for d, b in zip(dims, dim_bias)]
            parse_cv = min(0.4, max(0.03, cv * math.exp(rng.gauss(0.0, 0.45)) + 0.03))
            parse_std = [round(max(0.3, d * parse_cv), 1) for d in parse_dims]
            parse_weight = weight * math.exp(rng.gauss(0.0, 0.25))
            parse_rigidity = min(1.0, max(0.05, rigidity + rng.gauss(0.0, 0.08)))
            self.types.append(
                {
                    "idx": idx,
                    "name": name,
                    "dims": list(dims),
                    "cv": cv,
                    "weight": weight,
                    "rigidity": rigidity,
                    "material": material,
                    "sound": sound,
                    "loudness": loud,
                    "text_emb": text_emb,
                    "label_bias": label_bias,
                    "parse": {
                        "name": name,
                        "dims_cm": parse_dims,
                        "dims_std_cm": parse_std,
                        "weight_kg": round(parse_weight, 3),
                        "weight_std_kg": round(0.3 * parse_weight, 3),
                        "material": material,
                        "rigidity": round(parse_rigidity, 2),
                    },
                }
            )


# ---------------------------------------------------------------------------
# Physical fit in the true world: rigid guillotine packing where deformable
# objects can be squeezed to a fraction of their volume set by rigidity.
# ---------------------------------------------------------------------------


def _orients(dims):
    return sorted(set(itertools.permutations(tuple(dims), 3)))


def _fits_one(item, box):
    a = sorted(item)
    b = sorted(box)
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]


def true_pack(items: list[tuple[float, float, float]], box: tuple[float, float, float]) -> bool:
    if not items:
        return True
    if sum(math.prod(x) for x in items) > math.prod(box):
        return False
    if any(not _fits_one(x, box) for x in items):
        return False
    ordered = sorted(items, key=math.prod, reverse=True)

    def rec(remaining, free):
        if not remaining:
            return True
        item = remaining[0]
        for o in _orients(item):
            for k, f in enumerate(free):
                if o[0] <= f[0] and o[1] <= f[1] and o[2] <= f[2]:
                    rest = free[:k] + free[k + 1:]
                    splits = [
                        (f[0] - o[0], f[1], f[2]),
                        (o[0], f[1] - o[1], f[2]),
                        (o[0], o[1], f[2] - o[2]),
                        (f[0], f[1] - o[1], f[2]),
                        (f[0], f[1], f[2] - o[2]),
                    ]
                    rest.extend(s for s in splits if min(s) > 1e-9)
                    rest = sorted(rest, key=math.prod, reverse=True)[:12]
                    if rec(remaining[1:], rest):
                        return True
        return False

    return rec(ordered, [tuple(box)])


def effective_dims(dims, rigidity):
    squeeze = rigidity ** (1.0 / 3.0)
    return tuple(d * squeeze for d in dims)


def sample_box(rng: random.Random, styles):
    style = rng.choice(styles)
    scale = math.exp(rng.gauss(0.0, 0.15))
    dims = [d * scale * math.exp(rng.gauss(0.0, 0.08)) for d in style]
    return dims


def box_tare(dims):
    a, b, c = dims
    return CARDBOARD_KG_PER_CM2 * 2.0 * (a * b + a * c + b * c)


def box_sound(rng, world: World, box_dims, contents):
    """Feature vector of the sound made by shaking one box."""
    vol = math.prod(box_dims)
    fill = sum(math.prod(c["eff_dims"]) for c in contents) / vol
    freedom = min(1.0, max(0.08, 1.0 - fill)) ** 0.8
    sound = [0.0] * len(TIMBRES)
    for c in contents:
        t = world.types[c["type"]]
        heft = min(2.0, max(0.5, (c["weight"] / t["weight"]) ** 0.3))
        energy = t["loudness"] * freedom * heft * math.exp(rng.gauss(0.0, 0.5))
        for k, v in enumerate(t["sound"]):
            sound[k] += energy * v
    sound[T_INDEX["soft_thump"]] += 0.10 * math.exp(rng.gauss(0.0, 0.3))
    sound[T_INDEX["wood_knock"]] += 0.05 * math.exp(rng.gauss(0.0, 0.3))
    for k in range(len(sound)):
        sound[k] += abs(rng.gauss(0.0, 0.07))
    return sound


def audio_posterior(world: World, sound, type_ids):
    logits = [
        _cos(sound, world.types[t]["text_emb"]) / AUDIO_TEMPERATURE + world.types[t]["label_bias"]
        for t in type_ids
    ]
    m = max(logits)
    ws = [math.exp(l - m) for l in logits]
    z = sum(ws)
    return [w / z for w in ws]


def sample_scenario(rng: random.Random, world: World, sid: str) -> dict:
    while True:
        n = rng.choice([2, 3, 4, 5])
        type_ids = rng.sample(range(len(world.types)), n)
        objs = []
        for t in type_ids:
            spec = world.types[t]
            scale = math.exp(rng.gauss(0.0, spec["cv"]))
            dims = [d * scale * math.exp(rng.gauss(0.0, spec["cv"] / 2.0)) for d in spec["dims"]]
            weight = spec["weight"] * scale ** 2 * math.exp(rng.gauss(0.0, 0.1))
            objs.append(
                {
                    "type": t,
                    "dims": dims,
                    "eff_dims": effective_dims(dims, spec["rigidity"]),
                    "weight": weight,
                }
            )
        for _ in range(40):
            if rng.random() < P_LARGE_SMALL_PAIR:
                boxes = [sample_box(rng, LARGE_BOX_STYLES), sample_box(rng, SMALL_BOX_STYLES)]
                rng.shuffle(boxes)
            else:
                pool = LARGE_BOX_STYLES + SMALL_BOX_STYLES
                boxes = [sample_box(rng, pool) for _ in range(2)]
            feasible = []
            for assign in itertools.product(range(2), repeat=n):
                if len(set(assign)) < 2:
                    continue
                ok = True
                for b in range(2):
                    items = [objs[i]["eff_dims"] for i in range(n) if assign[i] == b]
                    if not true_pack(items, tuple(boxes[b])):
                        ok = False
                        break
                if ok:
                    feasible.append(assign)
            if feasible:
                break
        else:
            continue
        assign = rng.choice(feasible)
        break

    names = [world.types[t]["name"] for t in type_ids]
    box_inputs = []
    audio = []
    for b in range(2):
        contents = [objs[i] for i in range(n) if assign[i] == b]
        sound = box_sound(rng, world, boxes[b], contents)
        post = audio_posterior(world, sound, type_ids)
        audio.append({name: round(p, 5) for name, p in zip(names, post)})
        parsed = [round(d * math.exp(rng.gauss(0.0, 0.07)), 1) for d in boxes[b]]
        total = box_tare(boxes[b]) + sum(c["weight"] for c in contents)
        box_inputs.append(
            {
                "name": "left" if b == 0 else "right",
                "dims_cm": parsed,
                "dims_std_cm": [round(max(0.5, 0.06 * d), 1) for d in parsed],
                "lifted_weight_kg": round(total * math.exp(rng.gauss(0.0, 0.25)), 3),
            }
        )
    return {
        "id": sid,
        "objects": [dict(world.types[t]["parse"]) for t in type_ids],
        "boxes": box_inputs,
        "audio_posteriors": audio,
        "true_box": {name: int(assign[i]) for i, name in enumerate(names)},
    }


def build_split(count: int, seed: int, world: World) -> list[dict]:
    rng = random.Random(seed)
    ids = random.Random(seed + 1)
    return [sample_scenario(rng, world, f"witb_{ids.getrandbits(40):010x}") for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", type=Path, default=Path("data/witb_train.json"))
    parser.add_argument("--test-out", type=Path, default=Path("eval/test/witb_test_hidden.json"))
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--test-count", type=int, default=2000)
    args = parser.parse_args()
    world = World()
    train = build_split(args.train_count, TRAIN_SEED, world)
    test = build_split(args.test_count, TEST_SEED, world)
    header = {
        "format": "witb-scenarios-v2",
        "description": (
            "What's-in-the-Box scenarios: parsed object, box and per-box audio cues. "
            "true_box gives the index of the box that actually holds each object."
        ),
    }
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_out.write_text(json.dumps({**header, "split": "train", "scenarios": train}, indent=1) + "\n")
    args.test_out.parent.mkdir(parents=True, exist_ok=True)
    args.test_out.write_text(json.dumps({**header, "split": "test", "scenarios": test}) + "\n")
    print(f"wrote {len(train)} train scenarios to {args.train_out}")
    print(f"wrote {len(test)} test scenarios to {args.test_out}")


if __name__ == "__main__":
    main()
