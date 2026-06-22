# Northern Lights Line-Source Pressure Screening

This is a standalone screening result for the infinite-acting line-source formula. The same formula is exposed through simulator well and reservoir snapshots.

## Inputs

- Data file: `data\northern_lights_line_source_parameters.json`
- Active wells used for rate split: 1
- Elapsed time: 365.25 days
- Reservoir pressure means point pressure at radii: 100 m, 1000 m
- Initial pressure: 260 bar
- Effective thickness: 173 m
- Porosity: 0.22
- CO2 viscosity assumption: 6.00e-05 Pa s
- CO2 density assumption: 630 kg/m3

## Parameter Audit

| Parameter | Current value | Source | Confidence | Note |
|---|---:|---|---|---|
| initial_pressure_bar | 260 | Local docs/northern_lights_data_summary.md, summarizing reported Aurora pressure range 200-300 bar | medium | The pressure range is sourced, but 260 bar is a selected representative value rather than a directly reported initial pressure at the modelled well. |
| permeability_md | 100 | Local docs/northern_lights_data_summary.md, summarizing Marashi Aurora model horizontal permeability range 0.1-500 mD | medium | The range is model-based; the selected 100 mD is a moderate screening value rather than the optimistic upper bound or a calibrated effective well-test permeability. |
| thickness_m | 173 | Local docs/northern_lights_data_summary.md, summarizing Marashi Aurora model Johansen/Cook thickness | medium | Reported as a model value, but effective injection thickness near the well may differ. |
| porosity_fraction | 0.22 | Local docs/northern_lights_data_summary.md, summarizing Marashi Aurora model porosity range 0.073-0.314 | medium | The range is model-based; 0.22 is a representative selected value. |
| total_compressibility_1_pa | 7e-10 | Engineering assumption for brine/rock/CO2 system compressibility | low | No Northern Lights-specific compressibility value was found in the reviewed sources. |
| viscosity_pa_s | 6e-05 | Engineering assumption for dense/supercritical CO2 near 260 bar and 98-100 C | low | Should be recomputed from an equation of state or property package for the actual CO2 composition and pressure-temperature path. |
| co2_density_kg_m3 | 630 | Engineering assumption for dense/supercritical CO2 near 260 bar and 98-100 C | low | Used only to convert mass rate to reservoir volumetric rate; should be replaced with EOS/property-package output. |
| well_radius_m | 0.10795 | Northern-Lights-Project-Concept-report.pdf, p.40-p.41; derived from 8.5 inch open hole diameter in text and well schematic | high | The diameter is directly shown in the concept report; the radius is a direct unit conversion. |
| skin | 0 | Engineering assumption | low | No well-test skin or completion skin value was found; skin can strongly affect bottomhole pressure. |
| active_wells | 1 | Northern-Lights-Project-Concept-report.pdf, p.42; single well satellite phase 1 injection | high | Appropriate for the Phase 1 screening case; future expansion scenarios require an updated well count. |

Confidence convention: high = directly reported or directly derived from a project PDF; medium = sourced model/report range but selected representative value; low = engineering assumption pending replacement.

## Results

| Case | k (mD) | Rate (Mt/y) | Bottomhole pressure (bar) | dBHP (bar) | Reservoir pressure @100 m (bar) | dP @100 m (bar) | Reservoir pressure @1000 m (bar) | dP @1000 m (bar) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_medium_k | 100 | 0.4 | 261.40 | 1.40 | 260.63 | 0.63 | 260.37 | 0.37 |
| baseline_medium_k | 100 | 0.8 | 262.80 | 2.80 | 261.26 | 1.26 | 260.75 | 0.75 |
| baseline_medium_k | 100 | 1.5 | 265.25 | 5.25 | 262.37 | 2.37 | 261.40 | 1.40 |
| baseline_medium_k | 100 | 2.5 | 268.75 | 8.75 | 263.95 | 3.95 | 262.33 | 2.33 |
| baseline_medium_k | 100 | 5.0 | 277.51 | 17.51 | 267.90 | 7.90 | 264.66 | 4.66 |
| low_k | 50 | 0.4 | 262.72 | 2.72 | 261.19 | 1.19 | 260.67 | 0.67 |
| low_k | 50 | 0.8 | 265.45 | 5.45 | 262.37 | 2.37 | 261.34 | 1.34 |
| low_k | 50 | 1.5 | 270.21 | 10.21 | 264.45 | 4.45 | 262.51 | 2.51 |
| low_k | 50 | 2.5 | 277.02 | 17.02 | 267.41 | 7.41 | 264.18 | 4.18 |
| low_k | 50 | 5.0 | 294.04 | 34.04 | 274.83 | 14.83 | 268.35 | 8.35 |
| high_k_sensitivity | 500 | 0.4 | 260.30 | 0.30 | 260.14 | 0.14 | 260.09 | 0.09 |
| high_k_sensitivity | 500 | 0.8 | 260.60 | 0.60 | 260.29 | 0.29 | 260.19 | 0.19 |
| high_k_sensitivity | 500 | 1.5 | 261.12 | 1.12 | 260.54 | 0.54 | 260.35 | 0.35 |
| high_k_sensitivity | 500 | 2.5 | 261.86 | 1.86 | 260.90 | 0.90 | 260.58 | 0.58 |
| high_k_sensitivity | 500 | 5.0 | 263.73 | 3.73 | 261.81 | 1.81 | 261.16 | 1.16 |

## Interpretation Notes

- Pressure response is linear in injection rate for fixed time and fixed parameters.
- Lower effective permeability produces much larger pressure buildup because the coefficient scales approximately with 1/(k h).
- These are point-pressure estimates. They should not be read as whole-reservoir average pressure.
- Reservoir pressure columns are line-source point pressures p(r,t) at 100 m and 1000 m from the injection well.
- The baseline uses a moderate selected permeability; the high-k case is retained only as an optimistic sensitivity.
