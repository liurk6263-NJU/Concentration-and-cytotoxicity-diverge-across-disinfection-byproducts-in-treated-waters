# Code Ocean capsule layout

`analysis/` is the exact internal snapshot from `nature_water_repro_package_1008_20261009_new`. It is split across the capsule's `code/analysis`, `data/analysis`, and `results/analysis` folders only at the outer layer; the runner reconstructs the original relative directory tree in a temporary workspace before executing the published checks. This preserves the package's SHA-256 manifest and relative imports.

Set `code/run.sh` as the capsule's run file. The run is deterministic and non-interactive:

1. Reconstruct the original package tree in a temporary directory.
2. Run `audit_protocol.py`, which records the declared independence unit, role/fold overlap checks, source-domain overlap, and the leave-identical-compound-out control.
3. Run `verify.py` (45 integrity, metric, isolation, screening and source-data checks).
4. Replay the Figure 5 selection and applicability-domain checks.
5. Recalculate the treated-water class-contribution table.
6. Write logs and machine-readable outputs under `/results/reproducible_run`.

The default run validates and replays frozen results. It does not retrain the ten-seed models; the full training and audit entry points remain under `analysis/01_models/code/experiments` and are documented in the original package README. The frozen predictions and source data needed by the checks are included in the capsule.

The independence unit is `canonical_smiles + exposure_time`: the same chemical at different exposure times is intentionally treated as a distinct toxicity observation because exposure time changes the endpoint. `random_cv_oof` is a prediction view over the `train` records, so the audit checks its key set against `train` and checks fold isolation separately. The package reports held-out records/time conditions, while the separate leave-identical-compound-out control tests transfer when the canonical structure is shared between NCI-60 and CHO-K1.

The environment files are in `../environment`. Install dependencies during environment construction, then select `run.sh` as the file to run in Code Ocean. No network access or interactive input is used by the run.
