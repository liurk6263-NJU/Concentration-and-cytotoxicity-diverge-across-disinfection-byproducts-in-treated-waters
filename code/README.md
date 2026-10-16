# Running the analyses

## GitHub repository

The numbered folders contain the analysis code, inputs and results. INDEX.csv lists the files for each figure and Supplementary Information item. Model-training scripts are in 01_models/code/experiments.

## Code Ocean capsule

The capsule separates the same analysis files into code/analysis, data/analysis and results/analysis. The runner combines these folders in a temporary workspace before executing the analyses.

Select code/run.sh as the Code Ocean run file. From the capsule root, the equivalent command is:

```text
python code/run_capsule.py
```

The default run checks the saved predictions and data partitions, repeats the screening, treated-water and cellular-state calculations, and saves outputs in results/reproducible_run. Model training is separate from this run; capsule training scripts are in code/analysis/01_models/code/experiments.

Dependencies are listed in environment/requirements.txt. Data partitioning and validation are described in REVIEWER_AUDIT.md.
