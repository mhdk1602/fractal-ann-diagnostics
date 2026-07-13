# Experiments

## Current executable study

`run_governance_pilot.py` runs the fixed-seed v0.2 synthetic mechanism pilot. It replays every
search action against exact authorized top-k truth and writes compact evidence to
`artifacts/pilot/`.

```bash
python experiments/run_governance_pilot.py --output artifacts/pilot
```

## Historical precursor

`calibrate_v0_1_0.py` and `calibration-v0.1.0.md` preserve the original index-recommendation study.
They remain because deleting a failed approach would erase the chain of correction. They are not
valid v0.2 evidence: angular metrics were mishandled, backend outcomes were not measured, and the
former MFDFA statistic depended on corpus row order.
