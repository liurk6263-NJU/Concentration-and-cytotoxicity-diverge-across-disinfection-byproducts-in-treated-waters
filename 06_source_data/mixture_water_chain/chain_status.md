# CAT calculations and sources

## Treated water

The inputs contain 614 concentration records, 329 displayed occurrence rows and 63 process records. CAT = sum(Ci_mol_L / final_model_LC50_mol_L). The category components sum to the plotted sample totals within floating-point rounding.

Fig3_verification in the source workbook compares the model predictions and back-calculated values for Figure 3c,d. The S12 shares reported in the Supplementary Information are 22.8% and 2.8%.

## Mixtures

Figure S3 uses an equitoxic mixture design. The files include the mixture measurements, four single-compound inputs, forty monomer-seed predictions, thirty mixture-seed predictions and saved panel values.

BCAT and experimental D are calculated from the fitted experimental LC50 values. CAT, MDR, CI/p and D are checked by reverse calculation against the Supplementary Information in FigS3_verification. Mixture Ci values are derived from the equitoxic design rather than measured concentrations.
