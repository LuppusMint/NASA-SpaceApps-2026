# Known Limitations and Scope

This file consolidates limitations that are documented but not fully resolved in the product. It is part of the deliverable and should be read with `docs/01`, `docs/02`, and the validation JSON files.

## Scientific scope

- **Bridge target.** The cross-sensor bridge is trained on the **S-NPP-scale** event/FRP target, not on the full VIIRS union. NOAA-20 and NOAA-21 are combined after the bridge with `_dedup_ratios`. The N21 marginal ratio is assumed equal to N20 because N21 is absent from the 2012-2020 overlap; the 2023 relay step depends on that assumption.
- **Backcast assumption.** The 2000-2011 record is a backcast. No VIIRS exists before 2012, so the stability of the VIIRS/MODIS relationship before 2012 is untestable. The ratio-drift check inside 2012-2020 is evidence about stability, not proof.
- **Admin-1.** Admin-1 is a **proportional downscaling** of the country total by raw overpass/FRP shares, not a bottom-up sum of 50 km cells. Country = sum of Admin-1 is exact; the 50 km -> Admin-1 step is an allocation.
- **Panel.** The country/month cube has **4,014** rows, not 14 x 313 = 4,382. Missing country-months mean no detections; they are not missing data. FLK has 8 months (2024+), GUF 268, SUR 299, GUY 311, ECU/URY 312.
- **Borders.** 485 50 km cells straddle more than one country; the harmonized cube assigns each cell to one country. Country totals reflect that assignment.
- **Static sources.** The static mask is an OR of persistence (153 cells) and type-1/2 dominance (304 cells; 99 overlap), removing 2.87M detections. `type` is absent in VIIRS NRT, so the type rule is asymmetric.
- **Overpass consolidation.** The 2-hour overpass bin is a design choice; no 1 h/3 h sensitivity is reported. One event can contain several pixels, and one fire observed by two satellites counts as two events.
- **SAMA.** The suspect nocturnal flag never activates after the nominal+ confidence filter; it removes no rows. In the committed cubes, `sama_suspect_det_count` is the night-detection count in SAMA, not a suspect count. The API exposes `sama_night_det_count` as an alias; a future rebuild will rename the column.
- **Climatology and severity.** `n_years` ranges from 1 to 20; 48.0% of Admin-1 cells have fewer than 20 years, 7.7% fewer than 10, and 1.6% at most 3. Tiers with `n_years < 10` are `LOW_SAMPLE`; p90 and z are not used as critical thresholds. The UI suppresses the p90 ring below 10 years.
- **Trends.** Trends use annual totals from 2001-2025 and include the 2001-2011 backcast; reconstruction uncertainty is not propagated. `low_sample` marks n<10. The BH FDR family is global across SREHI/FRP and Admin-1/country.
- **M3 blend.** The causal blend orders spatial folds by median time, not by a strict calendar cut. The fitted M1/M2 weights are [0, 1], which limits the practical effect.
- **GEV.** The GEV is fitted by L-moments/PWM with a parametric bootstrap and a sanity bound. In the current run 475/510 series pass and 25/26 country series have CIs; 31 series are rejected by the sanity bound and 4 by insufficient active years. It is a screening diagnostic, not an operational return-period product. Admin-1 CIs are available where the bootstrap converges.
- **Interval coverage.** The nominal 90% interval is not calibrated for the largest fires in the Terra-only regime: measured coverage is 56.5% for `y > 100` and 90.1% overall; Terra+Aqua is 84.2% and 96.4%. The product publishes the nominal band and this limitation; per-size recalibration (S4-A) is future work.
- **Partial month.** November 2000 is a complete month and is not flagged as partial in the country/Admin-1/calendar artifacts; the year 2000 is a two-month start. The large 10 km/50 km Release cubes may still carry the pre-fix flag until a full rebuild.
- **Large assets.** The 10 km/50 km cubes, `harmonized_cell50km`, `oof_m0_m3`, and `splits.json` are Release assets, not Git-tracked. The repository without them serves country/Admin-1/calendar and CSV but not the map layers.

## Not implemented in this delivery

- Bottom-up Admin-1 aggregation (S2-A).
- Per-size-bin sigma recalibration (S4-A).
- Independent N21 marginal-ratio estimation.
- Strict calendar-time M3 blend.
- Quantitative sensitivity for 1 h/3 h overpass, border splitting, and static-mask OR.
