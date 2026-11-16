# Temporal hold-out validation of the cross-sensor bridge

**Question.** The published M0-M3 benchmark uses 5-fold spatial-block cross-validation inside the 2012-2020 overlap. The product is used mainly to reconstruct 2000-2011, i.e. years the model never sees. Does the model transfer across time as well as across space?

**Answer.** Yes, within the overlap period, but the result must be read per regime. The overlap has **930,100 rows = 465,050 cell-months x 2 counterfactual replicas**: every cell-month appears once as `TERRA_AQUA` and once as `TERRA_ONLY` (Aqua signals masked, same S-NPP target). Pooling them mixes two predictor regimes and double-counts every cell-month. The corrected results are:

- **Terra-only (the MODIS-only backcast regime):** M2 fitted on 2016-2020 predicts 2012-2015 with SREHI R² **0.808** (RMSE 13.11) and FRP R² **0.748** (RMSE 764.7). Its spatial-block score on the same regime is R² **0.793** (SREHI) and **0.745** (FRP). Temporal transfer is therefore not worse than spatial transfer for the backcast regime.
- **Terra+Aqua:** M2 predicts 2012-2015 with SREHI R² **0.920** (RMSE 8.47) and FRP R² **0.890** (RMSE 505.2), against spatial R² **0.913** and **0.875**.

Annual South America totals for the held-out years are less accurate than the pooled table suggested. Terra-only relative errors are **-8.2% to -0.7%** for SREHI (2020: **-8.6%**) and **-15.2% to -2.8%** for FRP (2020: **-15.2%**). Terra+Aqua errors are **-0.0% to +8.6%** for SREHI and **-6.0% to +5.2%** for FRP.

## Method

`scripts/temporal_holdout_validation.py` re-runs the **unchanged** model code (`_oof_predictions` and `_metric_bundle`) on the committed overlap table. All 930,100 rows are used for evaluation; up to 400,000 training rows are sampled per fold (seed 42). The fold assignment is by 50 km cell, so both replicas of a cell-month stay in the same fold.

| Scheme | Folds | What it tests |
|---|---|---|
| `spatial_block` | frozen `fold_id` (200 km blocks, 5 folds) | published scheme, re-run |
| `leave_one_year_out` | 9 folds (2012-2020) | transfer to an unseen year at known cells |
| `backward_block` | 2012-2015 vs 2016-2020 | the direction of the real backcast |

The M2 covariate-shift weights toward 2021-2026 need a Release asset, so all schemes use uniform weights (`future=None`). The spatial re-run reproduces the published M3 SREHI score closely (RMSE 13.07 / R² 0.818 vs 12.88 / 0.817). Compare schemes with each other and **within a regime**, not the old pooled numbers.

## M2 per regime (SREHI / FRP)

| Scheme | Terra-only RMSE / R² | Terra+Aqua RMSE / R² |
|---|---|---|
| spatial block | 13.943 / 0.792 | 9.052 / 0.913 |
| leave-one-year-out | 12.858 / 0.824 | 8.518 / 0.923 |
| backward 2012-2015 | 13.110 / 0.808 | 8.473 / 0.920 |

Per-regime scores for M0, M1, M2 and M3 are in `temporal_holdout_validation.json`. Two facts the pooled table hid:

- **M1 fails on Terra+Aqua** (SREHI R² -0.03 spatial, -0.21 backward); its pooled R² 0.24 was an average of a mediocre Terra-only fit and a worse Terra+Aqua fit.
- **M3 as scored is pulled down by the causal warm-up fold.** In production the fitted M1/M2 simplex weights are `[0, 1]`; the production backcast point is M2 for MODIS-only cells, while overlap cells with VIIRS blend the observation and the bridge.

## Coverage of the published 90% interval

The interval is `[max(p - 1.645σ, 0), p + 1.645σ]`. The table reports the share of held-out observations inside it for 2012-2015, using the production-equivalent construction **for the MODIS-only backcast** (M2 point + M3 sigma); it is not the overlap production interval, where observation and bridge are blended.

| Observed value | Terra-only n | Terra-only coverage | Terra+Aqua n | Terra+Aqua coverage |
|---|---:|---:|---:|---:|
| all | 197,377 | **90.1%** | 197,377 | **96.4%** |
| 0 | 9,057 | 98.8% | 9,057 | 99.1% |
| 0 < y ≤ 10 | 126,376 | 98.4% | 126,376 | 99.3% |
| 10 < y ≤ 100 | 56,980 | **73.3%** | 56,980 | 90.8% |
| y > 100 | 4,964 | **56.5%** | 4,964 | 84.2% |

**Reading.** Average Terra-only coverage is close to nominal, but the interval is **too narrow for large fires**: 56.5% coverage for `y > 100`, which is exactly where critical-period calls matter. Terra+Aqua is better (84.2%) but still below nominal for the largest fires. A per-size-bin recalibration is the recommended next step; it is **not applied** here because it changes the published product. Any recalibration must use the deduplicated cell-month sample, not the 930,100-row replica table.

## Limitations

- The core assumption is untestable: 2012-2015 is still inside the MODIS/VIIRS overlap; it cannot prove the relationship before 2012 matched the overlap years.
- The overlap rows are counterfactual replicas, not independent observations. Metrics are per regime; CIs should be clustered by `(year_month, cell_id_50km)`.
- M2 was trained without covariate-shift weights.
- Cells in the same year are not independent; pooled R² describes cell-month skill, not a number of independent tests.
- Terra-only annual totals carry larger errors (up to -8.6% SREHI, -15.2% FRP in 2020) than the old pooled table implied.

## Reproduce

```bash
python scripts/temporal_holdout_validation.py --out docs/validation/temporal_holdout_validation.json
```

The JSON records the input SHA-256, the replication counts, every per-regime metric and coverage table.
