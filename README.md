# Concentration and cytotoxicity diverge across disinfection byproducts in treated waters

Data and code accompanying the study. [INDEX.csv](INDEX.csv) links each figure and Supplementary Information item to its data, scripts and results.

| Folder | Content |
|---|---|
| 00_workflow | Figure 1 workflow |
| 01_models | Figure 2, model inputs, training and evaluation |
| 02_treated_water | Figure 3, water concentrations and CAT calculations |
| 03_mechanistic | Figure 4, SHAP and pathway and structure analyses |
| 04_screening | Figure 5, predictions and candidate selection |
| 05_supplementary | Figures S1-S5 and Tables S1-S9 |
| 06_source_data | Source workbook and figure-to-data index |

## Analysis

Dependencies are listed in [requirements.txt](requirements.txt). Analysis scripts are indexed in INDEX.csv; model-training scripts are in [01_models/code/experiments](01_models/code/experiments).

The Code Ocean run checks saved predictions and repeats the screening, treated-water and cellular-state calculations. [Capsule instructions](https://github.com/liurk6263-NJU/Concentration-and-cytotoxicity-diverge-across-disinfection-byproducts-in-treated-waters/blob/main/code/README.md) describe the run and its outputs.

## Source data

[Source_Data_Figures_Tables_1009.xlsx](06_source_data/Source_Data_Figures_Tables_1009.xlsx) contains the figure and table data. [source_status.csv](06_source_data/source_status.csv) identifies measured, predicted and back-calculated values. Sample mappings and reporting units are documented with the corresponding analyses.
