# Input data

The capsule stores the package's raw and processed inputs under `data/analysis`, preserving the analysis layout (for example `01_models/data_raw` and `04_screening/data_raw`). Raw files use English filenames, sheet names and labels for submission. Numeric values, formulas, SMILES and record counts are unchanged. The run reconstructs these paths beside the source code in a temporary workspace so that the data contract and recorded hash checks remain valid.

Before public release, confirm that every bundled dataset and external database file is licensed for redistribution. If a file cannot be redistributed, replace it with a documented Code Ocean Data Asset or access instruction and update the package hash manifest accordingly.
