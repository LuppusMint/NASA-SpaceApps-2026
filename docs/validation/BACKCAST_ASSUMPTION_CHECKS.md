# Backcast assumption checks

The 2000–2011 record is a backcast: MODIS observations converted to the VIIRS scale with a bridge fitted on 2012–2020. Two questions decide how far that record can be trusted. This document gives the available evidence on each. Script: `scripts/backcast_assumption_checks.py`. Output: `backcast_assumption_checks.json`. Every input is committed, and their SHA-256 hashes are recorded in the JSON.

## 1. Was the VIIRS/MODIS relationship stable?

Every MODIS-to-VIIRS backcast assumes that the relationship measured in 2012–2020 also held before 2012. That cannot be observed, because there is no VIIRS before 2012. The best available evidence is whether the relationship **drifted inside** 2012–2020. For each year we compute the ratio `sum(VIIRS-scale target) / sum(MODIS)` over all 50 km cell-months of the overlap table and fit a Theil-Sen trend to its logarithm. The last column extrapolates that trend to 2006, the middle of the backcast, relative to the 2012–2020 centre. A positive value means the true 2006 ratio would be higher than the fitted one, so the backcast would be too low.

| Series | Annual ratio range | Year-to-year CV | Trend %/yr (95% CI) | Implied ratio change at 2006 (95% CI) |
|---|---|---:|---|---|
| South America, Terra+Aqua, SREHI | 2.28–2.59 | 4.3% | −0.73 (−2.39 to +1.06) | +7.6% (−10.0 to +27.4) |
| South America, Terra+Aqua, FRP | 0.81–0.92 | 4.4% | −0.45 (−1.87 to +1.28) | +4.6% (−11.9 to +20.8) |
| South America, Terra-only, SREHI | 5.75–6.91 | 6.6% | −1.44 (−2.61 to +0.97) | +15.6% (−9.2 to +30.3) |
| South America, Terra-only, FRP | 2.47–3.03 | 5.7% | −0.56 (−2.05 to +1.60) | +5.8% (−14.7 to +23.0) |

Six largest fire countries (Terra+Aqua, SREHI):

| Country | CV | Trend %/yr (95% CI) | Implied ratio change at 2006 (95% CI) |
|---|---:|---|---|
| Brazil | 5.5% | −0.81 (−3.09 to +0.93) | +8.5% (−8.8 to +36.9) |
| Venezuela | 4.0% | −0.25 (−1.68 to +1.27) | +2.5% (−11.9 to +18.5) |
| Argentina | 4.2% | +0.12 (−1.75 to +1.40) | −1.2% (−12.9 to +19.3) |
| **Bolivia** | 7.2% | **−1.98 (−4.72 to −0.10)** | **+22.1% (+1.0 to +62.2)** |
| Paraguay | 3.6% | −0.52 (−1.97 to +0.44) | +5.4% (−4.3 to +22.1) |
| **Colombia** | 3.9% | **−1.05 (−1.91 to −0.38)** | **+11.1% (+3.9 to +21.2)** |

**Reading.**
- At the South America level the relationship is stable: annual variation is about 4%, and no trend is distinguishable from zero.
- **Bolivia and Colombia show significant drifts.** If those drifts extended back to 2006, the reconstructed Bolivian and Colombian totals for the mid-2000s would be too low by roughly 10–20%.
- These trends cover nine years and are extrapolated ten years back, so the intervals are wide. They describe a plausible size of error, not a correction.
- The production model (M2) is not a constant ratio: it conditions on MODIS counts, FRP, season, latitude and other features, so it can absorb part of a change in fire-size mix. The raw ratio drift is therefore a conservative indicator.
- The ratio is `target_srehi / modis_events` = 1 + (VIIRS/MODIS), not a pure VIIRS/MODIS ratio. The trend sign and size are informative, but do not read the percentage as a pure instrument ratio.
- **The Theil-Sen confidence intervals are unadjusted.** They assume independent annual residuals and there is no correction for autocorrelation or for the ten series compared (four South America regimes plus six countries). "Significant" here means only that the unadjusted 95% interval excludes zero. Bolivia and Colombia should be treated as exploratory until a block bootstrap and an FDR correction are applied; the wide Bolivia interval (+1.0% to +62.2% at 2006) already shows the uncertainty.

## 2. Are the relay steps within natural variability?

The relay guard in `benchmarks_m0_m3.json` passes whenever the harmonized 12-month step at a sensor transition is smaller than the raw step. That bar is low. Here the same statistic, the mean of the 12 months after divided by the mean of the 12 months before, is computed on the harmonized South America series at every start month whose 24-month window contains no sensor transition and no partial month. There are 172 such start months. They define how large a 12-month change is when no sensor changes.

Natural 12-month step, as a ratio (|log| quantiles mapped back to a ratio):

| Quantity | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|
| SREHI | 1.17× | 1.27× | 1.38× | 1.48× |
| FRP | 1.32× | 1.55× | 1.90× | 2.38× |

Relay steps placed in that distribution (percentile of the step size):

| Transition | SREHI raw → harmonized | Harmonized percentile | FRP raw → harmonized | Harmonized percentile |
|---|---|---:|---|---:|
| 2002-07 (Aqua joins Terra) | 3.41× → **1.52×** | **95th** | 5.02× → 1.60× | 79th |
| 2012-01 (VIIRS S-NPP) | 3.94× → 1.18× | 51st | 2.26× → 1.19× | 35th |
| 2018-04 (NOAA-20) | 1.41× → 0.84× | 55th | 0.98× → 0.70× | 64th |
| 2021-01 (MODIS drift era) | 0.74× → 0.75× | 83rd | 0.73× → 0.74× | 56th |
| 2023-01 (NOAA-21) | 1.38× → 1.00× | 0th | 1.18× → 0.92× | 18th |

**Reading.**
- The raw 2002 and 2012 steps lie at the 100th percentile: they are sensor artefacts.
- After harmonization, every step except one is inside the 90th percentile of natural variability. The 2012 VIIRS transition sits at the median.
- **The exception is SREHI at the 2002 Aqua join (1.52×, 95th percentile).** Part of that step may be real fire activity in 2002–2003, but part may be residual sensor inflation. Mid-2002 values should be read with that in mind.
- The 2021 step is a decline that harmonization leaves unchanged. It is consistent with natural variability (83rd percentile) and with the MODIS orbital-drift era, which the relay replaces with VIIRS.

Caveats: overlapping 24-month windows are autocorrelated, so the 172 start months are not independent samples. Percentiles are descriptive, not significance tests. The natural distribution comes from the harmonized product itself, so it includes both real variability and residual model noise.

## Reproduce

```bash
python scripts/backcast_assumption_checks.py --out docs/validation/backcast_assumption_checks.json
```

The run takes a few seconds and uses only committed files.
