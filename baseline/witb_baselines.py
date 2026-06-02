#!/usr/bin/env python3
"""Unimodal ablation baselines from the WiTB paper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "method"))

from witb_bayes import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/witb_parsed_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--which", choices=["audio", "vision", "all"], default="all")
    args = parser.parse_args()
    modes = ["audio", "vision"] if args.which == "all" else [args.which]
    for mode in modes:
        run(args.data, args.output_dir / f"witb_{mode}_predictions.json", mode, args.samples, args.seed)


if __name__ == "__main__":
    main()
