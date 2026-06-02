# Parsed WiTB Benchmark

The paper's original 25 video stimuli, parsed box/object attributes, CLAP audio
outputs, and human ratings are not released with the paper. This repository
therefore includes a small WiTB-style parsed benchmark that preserves the
paper's computational problem: infer hidden object placements from uncertain
box geometry and per-box audio posteriors.

The benchmark stores only method-agnostic inputs and target human-like
judgments. The paper method in `method/` consumes these inputs, but evaluation
code in `eval/` does not import from `method/`.
