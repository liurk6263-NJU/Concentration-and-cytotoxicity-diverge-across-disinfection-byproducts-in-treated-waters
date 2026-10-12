# Nature Water reproducibility data

Source data, code, fixed parameters and frozen results for NW-Manuscript-1008 and NW-Supporting Information-1008. INDEX.csv follows the manuscript and SI. Shared inputs are stored once. figure_versions.csv identifies the author-updated SVG/PNG files. Raw workbooks use English filenames, sheet names and labels; numeric values, formulas, SMILES and record counts are unchanged from the supplied source files.

| Folder | Content |
|---|---|
| 00_workflow | Figure 1 workflow |
| 01_models | Figure 2 and model inputs, parameters and predictions |
| 02_treated_water | Figure 3, 63 samples and 614 concentration records |
| 03_mechanistic | Figure 4, SHAP, pathway controls and structure series |
| 04_screening | Figure 5, catalogue predictions and selected records |
| 05_supplementary | Figures S1-S5 and Tables S1-S9 |
| 06_source_data | `Source_Data_Figures_Tables_1009.xlsx`, verification summaries and figure-to-data index |

Model metrics are means of ten seed-specific metrics. LC50 is inverse-transformed per seed before averaging. Splits use canonical SMILES plus exposure time. Figure 2d retains the registered within-role interference predictions.

Figure 3 display S1-S12 maps to original sample IDs 4,5,1,2,3,6,8,9,10,11,12,7. O3 is the author-confirmed process label for display S12. Concentration inputs are original measurements. Category CAT components are retained with the frozen sample totals; Figure 3 c/d is verified by final-model prediction and back-calculation under the author-defined small-difference criterion.

Figure S3 is mixture toxicity and Figure S4 is cellular state. The mixture design uses equitoxic contributions. Original assay worksheets and Origin projects are retained. BCAT and experimental D can be recalculated from original fitted LC50 values. The frozen model CAT, CI/p and prediction-D chain are tracked in 06_source_data/source_status.csv and FigS3_verification.

Figure S5 uses the exact median and upper quartile of 369 labeled observations, displayed as 0.158/0.228. Counts are 185/92/92. Candidate-domain SD uses a separate development-population cutoff, displayed as 0.165.

Table S3 values are 0.865 and 0.053. Table S6 reports 1,120 unique structures with 25/5 potency-threshold counts; 2,857 pair records have 27/6 counts. The two units are documented beside Table S6. Figure 5c retains six local SMILES/structure identities; recorded PubChem fields keep their original evidence status.

```text
python verify.py
python 04_screening/code/selection.py
python 04_screening/code/domain.py
```

Model and analysis entries are listed in INDEX.csv. verify.py checks the recorded raw hashes, frozen-prediction metrics, role/fold isolation, selected records and package integrity. The source workbook adds explicit Figure 3 and Figure S3 verification summaries. The raw-hash records were regenerated after the English-only workbook and filename cleanup.
