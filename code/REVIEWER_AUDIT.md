# Reviewer audit of data isolation

## Protocol decision

The primary independence unit is `canonical_smiles + exposure_time`. This is a scientific decision, not a convenience: the same DBP can have materially different cytotoxicity at different exposure times, so a record at 24 h and a record at 72 h are distinct endpoint observations. They may therefore be assigned to different roles only when their full protocol keys differ.

The stronger interpretation is recorded separately. A held-out record or held-out exposure-time condition is not a claim that the chemical structure was never present at another exposure time. Manuscript wording should use “held-out records/time conditions” unless the structure-level control is explicitly being discussed.

## Checks implemented

- Every frozen seed keeps each `canonical_smiles + exposure_time` key in one of the mutually exclusive `train`, `test`, `validation` or `experimental` roles.
- `random_cv_oof` is an out-of-fold result view of the `train` role, so its protocol-key set is expected to equal `train`; it is not treated as a fifth mutually exclusive role.
- Every random-CV key occurs in one fold per seed.
- The four independent experimental compounds are excluded from the CHO-K1 literature development frame by canonical structure.
- The NCI-60 source domain shares eight canonical structures with the CHO-K1 target domain. The released leave-identical-compound-out control removes those source rows from direct transfer and source-teacher construction; all ten seeds pass with zero leakage hard failures.
- Candidate selection, calibration and blend-weight selection use only development/train or other-fold data. Test, literature-validation and experimental labels are report-only.

## Recommended wording correction

Use “held-out records/time conditions” for the 37-record test set and reserve “unseen compounds” for the four independent experimental compounds, whose canonical structures were excluded from the literature development frame. This preserves the intended exposure-time protocol and avoids overstating structure-level independence.

The default capsule run writes the machine-readable evidence to `results/reproducible_run/protocol_audit.json` and retains the original frozen package unchanged.
