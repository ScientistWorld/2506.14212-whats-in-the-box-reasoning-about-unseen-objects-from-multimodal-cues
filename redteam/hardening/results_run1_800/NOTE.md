First hardening job (a9c565e4-b83), 800 hidden test scenarios. Kept for the
record: `better_weight_cue` was rejected here because one probability was
1.0000000000000002 (floating-point rounding); the evaluators now accept values
within 1e-6 of [0, 1] and clip them. The trained late-fusion stacker in this run
had a diverged (unregularised) fit. Both were fixed and the test was enlarged to
2000 scenarios before the final job; see ../results/.
