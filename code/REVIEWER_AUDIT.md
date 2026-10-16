# Data partitioning and validation

Data partitions use canonical SMILES and exposure time (`canonical_smiles + exposure_time`) as the observation key.

- Within each seed, each key belongs to one of the train, test, validation or experimental roles.
- random_cv_oof contains out-of-fold predictions for the training keys. Each key occurs in one validation fold per seed.
- The 37-record test set evaluates held-out observations and exposure-time conditions. The four experimental compounds are excluded from CHO-K1 model development by canonical structure.
- The NCI-60 and CHO-K1 datasets share eight canonical structures. The leave-identical-compound-out control excludes these source records from direct transfer and source-teacher construction.
- Candidate selection, calibration and blend weights use training or other-fold data. Test, literature-validation and experimental labels are used for evaluation.

The capsule run records the partition and transfer-control checks in results/reproducible_run/protocol_audit.json.
