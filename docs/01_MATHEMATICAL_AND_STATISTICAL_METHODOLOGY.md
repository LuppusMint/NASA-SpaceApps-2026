# Mathematical and Statistical Methodology

**Fire Relay South America - cross-sensor harmonization of MODIS and VIIRS active-fire detections, 2000-11 to 2026-11.**

This document describes the implemented mathematics, not an aspirational design. Every equation below maps to code in `src/core/`, `src/data/`, or `src/models/`. Numerical values are the committed benchmark outputs.

---

## 1. Notation and scope

Let \(i\) index a 50 km equal-area cell, \(t\) a calendar month, and \(s\) a sensor slot. The raw record is a set of active-fire detections with latitude, longitude, acquisition time, instrument, confidence, brightness temperature, and fire radiative power (FRP). The target quantities are:

- \(E_{i,t}\): consolidated active-fire overpass events, the basis of **SREHI**;
- \(F_{i,t}\): integrated FRP in megawatts;
- \(A_{i,t}\): active 10 km cells inside the 50 km cell.

The record spans 26 years and 313 months. The five sensor slots are `MODIS_Terra`, `MODIS_Aqua`, `VIIRS_SNPP`, `VIIRS_N20`, and `VIIRS_N21`.

---

## 2. Physical sensor biases and the relay problem

### 2.1 Footprint and sensitivity

MODIS has a **1 km** nadir footprint; VIIRS has a **375 m** nadir footprint. A single MODIS pixel can contain several VIIRS pixels. Consequently, the same fire can appear as one MODIS detection and several VIIRS detections. The count ratio is not constant: it depends on fire size, fragmentation, biome, cloud cover, view angle, and confidence threshold.

VIIRS also detects cooler and smaller fires, including nocturnal fires that MODIS may miss. Its additional overpasses increase the probability of observing a short-lived fire. FRP is less sensitive to pixel fragmentation than a raw detection count, but it is still affected by saturation, cloud screening, and the sensor's radiometric response.

### 2.2 Constellation expansion, 2000-2026

The effective constellation changes repeatedly:

| Period | Dominant constellation | Consequence |
|---|---|---|
| 2000-11 to 2002-06 | Terra only | Sparse early record |
| 2002-07 to 2011-12 | Terra + Aqua | More daily overpasses |
| 2012-01 to 2018-03 | Terra + Aqua + VIIRS S-NPP | First MODIS/VIIRS overlap |
| 2018-04 to 2020-12 | Adds NOAA-20/J1 | Denser VIIRS backbone |
| 2021-01 to 2022-12 | Orbital drift and sampling change | Non-stationary swath coverage |
| 2023-01 to 2026-11 | Adds NOAA-21/J2 | Deduplicated VIIRS backbone |

A naive sum over sensors turns these additions into artificial upward trends. The distortion is concentrated at the relay boundaries **2002, 2012, 2018, 2021, and 2023**.

### 2.3 Orbital drift and the LDOPE HC_MOD_26008 note

Terra and Aqua orbital drift changes the local overpass time and the swath geometry after 2021. The NASA LDOPE quality note **HC_MOD_26008** documents the MODIS orbital-drift condition. The implementation responds in two ways:

1. the calibration overlap is restricted to the pre-drift window **2012-02 to 2020-12**;
2. the relay continuity diagnostic is evaluated separately at 2021 and 2023, so a drift-induced step cannot be hidden inside a single global error metric.

The model does not claim to correct the orbital mechanics. It learns the sensor-to-sensor mapping on the stable overlap and then applies a regime-aware bridge, a Kalman-smoothed relay, and hierarchical reconciliation to the full record.

### 2.4 South Atlantic Magnetic Anomaly (SAMA)

The SAMA region is defined as **7°N–55°S, 110°W–11°E** (the implementation uses latitude between -55.0 and 7.0 and longitude between -110.0 and 11.0). Energetic particles in this region can produce false or degraded nocturnal VIIRS detections. The pipeline:

- flags every detection inside the SAMA box;
- marks nocturnal low-confidence NRT VIIRS detections as suspect;
- flags suspect nocturnal detections; in the committed run the flag never activates because the nominal+ confidence filter already excludes VIIRS low confidence, so this rule removes no rows;
- retains the SAMA containment flag on every aggregate so downstream users can condition on it.

The clean record contains zero suspect rows because the class is empty after the confidence filter. This is a structural zero, not evidence that a removal step was effective. In the harmonized cubes, `sama_suspect_det_count` is the night-detection count for SAMA rows, not a count of suspect rows; the next full rebuild will rename it to `sama_night_det_count`.

---

## 3. Equal-area gridding

### 3.1 Albers Equal-Area Conic

All spatial aggregation uses an Albers Equal-Area Conic projection in the **EPSG:102033 South America** family. The implementation builds the PROJ string from the South America bounding box:

```text
+proj=aea +lat_0=<centre> +lon_0=<centre>
+lat_1=<inner parallel> +lat_2=<outer parallel>
+datum=WGS84 +units=m
```

The standard parallels are placed inside the latitudinal span (`lat_min + span/6` and `lat_max - span/6`), which preserves area across the continent. Equal-area cells are essential: a longitude-latitude grid would shrink high-latitude cells and inflate equatorial cells, making a "cell count" incomparable across the continent.

### 3.2 Two resolutions

- **10 km grid**: the physical detection and static-source layer. Cell indices are `floor(x_m / 10_000)` and `floor(y_m / 10_000)`.
- **50 km grid**: the calibration and public map layer. Cell indices are `floor(x_m / 50_000)` and `floor(y_m / 50_000)`.

The 10 km layer preserves the native detection footprint. The 50 km layer reduces zero-inflation and makes the monthly cell-month panel stable enough for cross-sensor learning. Country and continental totals are sums of the assigned 50 km cells. Admin-1 is a proportional downscaling of the country total by raw overpass/FRP shares; it is not a bottom-up sum of 50 km cells.

### 3.3 Boundary assignment and epistemic precision

Country and Admin-1 labels are assigned by point-in-polygon against Natural Earth 1:10m boundaries. A 10 km cell has a half-width of about **5 km**; a 50 km cell has a half-width of about **25 km**. When a cell centre lies close to a boundary, the label is an epistemic approximation, not a survey-grade geolocation. The application states this limit explicitly rather than presenting municipal precision that the grid cannot support.

---

## 4. SREHI: the Sensor-Relay-Ensemble Harmonized Index

### 4.1 Overpass consolidation

Raw detections are first collapsed into **overpass events**. The event key is:

```text
satellite | acquisition date | day/night flag | floor(acquisition hour / 2)
```

Several pixels from the same satellite in the same two-hour overpass bin count once. This reduces pixel fragmentation without erasing genuine multiple overpasses. The monthly 10 km count is

\[
E^{(10)}_{c,t} = \#\{\text{distinct overpass buckets in cell } c \text{ and month } t\}.
\]

The 50 km SREHI is the sum of the 10 km event counts over the 25 constituent 10 km cells:

\[
E_{i,t} = \sum_{c \in i} E^{(10)}_{c,t}.
\]

In the public CSV, `raw_srehi` is exactly this `overpass_event_count`. The M0-M3 models estimate the harmonized SREHI, which corrects the sensor mix.

### 4.2 Why events rather than raw detections

A raw detection count conflates fire activity with pixel size and sensor sensitivity. The event count is closer to "how many times a fire was observed" and is more stable across the MODIS/VIIRS footprint difference. It is still not a physical fire count: one large fire can generate several overpass events, and one event can contain many pixels.

### 4.3 FRP

FRP is summed per cell-month after quality control and negative-value repair:

\[
F_{i,t} = \sum_{d \in (i,t)} \mathrm{FRP}_d.
\]

FRP is expressed in megawatts. The integrated continental total is **1,453,803,616.63 MW** before harmonization; conservation is checked after every aggregation step.

---

## 5. Quality control before modelling

The deterministic cleaning layer applies:

1. **Confidence**: MODIS confidence at least 30%, VIIRS nominal or high; a high-severity track retains MODIS at least 75% and VIIRS high.
2. **Type removal**: rows with `type IN (1,2,3)` are removed where the flag exists (**2,703,746** detections).
3. **Static thermal mask**: **358** persistent 10 km cells are removed. A cell is persistent when it has at least 9 active months per year over at least 5 years, or at least 5 type-1/2 rows with a type-1/2 share above 0.50. This removes **2,872,511** detections.
4. **Repairs**: **138** negative FRP values are clipped and **39** zero-brightness rows are repaired.
5. **SAMA**: suspect nocturnal detections are flagged; the flag is redundant with the nominal+ confidence filter, so zero rows are removed. The output column is a night-detection count, not a suspect count.
6. **Overpass consolidation**: the event key above is applied before gridding.

The clean count is **75,556,207** out of **85,002,288** raw detections.

---

## 6. The M0-M3 model family

All models are evaluated on the same frozen folds and the same pre-drift overlap. The calibration window is **2012-02 to 2020-12**, with **930,100** rows = **465,050** cell-months x **2** counterfactual replicas (`TERRA_AQUA` and `TERRA_ONLY`, same S-NPP target) and **7,184** unique 50 km cells.

### 6.1 M0 - raw unharmonized

M0 is the observational baseline. For each cell-month it predicts the MODIS-only event count and MODIS-only FRP:

\[
\hat{E}^{M0}_{i,t} = E^{\text{MODIS}}_{i,t}, \qquad
\hat{F}^{M0}_{i,t} = F^{\text{MODIS}}_{i,t}.
\]

M0 is not a model of the full product. It is the "what the long record would look like without harmonization" reference.

### 6.2 M1 - physical biweight bridge

M1 learns a robust mapping from the MODIS predictors to the **S-NPP-scale event-count target** on the pre-drift overlap. N20/N21 are combined later with the `_dedup_ratios` factors; the bridge target is not the full VIIRS union. Two ingredients matter:

**Tukey biweight location.** For observations \(x_1,\dots,x_n\), with median \(m\) and MAD \(s = \mathrm{median}|x_j-m|\), the biweight M-estimator iterates

\[
u_j = \frac{x_j-m}{c s}, \qquad
m \leftarrow m + \frac{\sum_{|u_j|<1}(1-u_j^2)^2(x_j-m)}{\sum_{|u_j|<1}(1-u_j^2)^2},
\]

with \(c=6\). The implementation returns the median when the sample is too small, constant, or has zero MAD.

**Cross-sensor bridge.** The bridge is fitted separately for SREHI (Tweedie log-link) and FRP (hurdle: a positive/zero classifier and a positive-intensity regressor). Predictors include Terra/Aqua event counts and FRP, active 10 km cells, MODIS clean detections, night fraction, high-confidence fraction, SAMA containment, absolute latitude, and month sine/cosine. Features are clipped to the 0.1%-99.9% training quantiles before the log link, and a grouped robust correction is applied afterwards. The predictive standard deviation is heteroscedastic:

\[
\sigma_{i,t} = \sigma_0 \sqrt{\frac{\max(\hat{E}_{i,t},1)}{\bar{E}}}.
\]

M1 is physically interpretable but does not capture the full nonlinearity of the relay. Its SREHI RMSE (**26.7017**) is slightly worse than M0, which is an honest result, not a hidden failure.

### 6.3 M2 - hurdle Tweedie with covariate-shift weights

M2 is a LightGBM hurdle model:

\[
P(E>0 \mid x) = p(x), \qquad
\mathbb{E}[E \mid E>0, x] = \mu(x),
\]

with a Tweedie objective (variance power 1.5) for the positive intensity and a binary classifier for the hurdle. The point prediction is

\[
\hat{E}^{M2}(x) =
\begin{cases}
p(x)\mu(x) & \text{soft expectation},\\
\mathbf{1}[p(x)\ge \tau_k]\mu(x) & \text{hard hurdle},
\end{cases}
\]

where the threshold \(\tau_k\) is fitted on the outer-training folds and applied to the held-out fold. The hard threshold is used only when its out-of-fold RMSE is no worse than the soft expectation; otherwise the soft expectation is retained. The per-fold sigma scale is calibrated on outer-training residuals, never on the validation fold.

**Covariate-shift weights.** The training overlap (2012-2020) and the future features (2021-2026) do not have the same distribution. Density-ratio weights are estimated with a classifier that distinguishes training from future rows:

\[
w(x) = \frac{\hat{p}_{\text{train}}(x)}{1-\hat{p}_{\text{train}}(x)},
\]

clipped for stability. These weights are passed to both LightGBM learners. This is a importance-weighting correction, not a claim that the future regime is fully observed.

M2 achieves the best point error: SREHI RMSE **11.8025** (R-squared **0.8512**) and FRP RMSE **790.2797** (R-squared **0.8018**).

### 6.4 M3 - simplex blend, Kalman relay, hierarchical reconciliation

M3 is the operational product. It combines M1 and M2 with a constrained simplex blend, smooths the relay in time, and reconciles the hierarchy.

**Simplex blend.** For each target, the blend weights \(w\) satisfy

\[
w \ge 0, \qquad \sum_m w_m = 1, \qquad
\hat{y}^{M3} = \sum_m w_m \hat{y}_m.
\]

The weights are optimized by SLSQP from a uniform start plus the best single-model vertices, with optional pruning of weights below 0.02. The optimization is **fold-wise temporal**: the first fold is a uniform warm-up, and each later fold is fitted only on strictly earlier folds. This prevents future information from entering the blend. If the blended score is worse than the best single model, the code falls back to that single model rather than accepting a dominated blend.

**One-dimensional Kalman relay.** The marginal NOAA-20 / S-NPP ratio that links the S-NPP-only and S-NPP + NOAA-20 regimes is smoothed with a scalar Kalman filter. For a latent ratio \(r_t\),

\[
r_t = r_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,Q),
\]
\[
z_t = r_t + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0,R_t).
\]

Here \(z_t\) is the monthly mean of country-level NOAA-20 / S-NPP event ratios over the overlap window `2018-04..2020-12` (country-months with more than 20 S-NPP events), \(R_t\) is estimated from robust first differences, and the process variance is \(Q=10^{-3}\) in the committed run. The smoothed series is then collapsed to one marginal ratio with the Tukey biweight location, and that ratio sets the S-NPP + NOAA-20 de-duplication factor; the FRP ratio uses the biweight location without smoothing. The filter therefore damps noise from months with few overpasses; it does not model a seasonal cycle.

The step-change diagnostics at the relay boundaries (`2002, 2012, 2018, 2021, 2023`) use the unsmoothed 12-month mean ratios only (`raw_step_change_ratio`, `harmonized_step_change_ratio`). An earlier smoothed-ratio diagnostic applied the same filter with an absolute \(Q=10^{-2}\) to monthly totals of magnitude \(10^4\)–\(10^6\); the smoothed series was almost flat, so its ratio was approximately 1 for any input. It has been removed from the pipeline. The stale `transition_smoothing` block was removed from the committed `data/processed/benchmarks_m0_m3.json`; the code no longer emits it.

**Non-negative reconciliation and Admin-1 downscaling.** Country totals are sums of the harmonized 50 km cells; Admin-1 is a proportional downscaling of the country total by raw overpass/FRP shares, not a bottom-up sum of cells. For a parent total \(T\) and child predictions \(c_1,\dots,c_m\), the reconciled children are

\[
\tilde{c}_j = \frac{c_j}{\sum_k c_k} T
\]

when the child sum is non-zero, with a weighted uniform allocation when the sum is zero, followed by a residual correction on the largest child. Negative children are clipped to zero before allocation. This is a proportional, non-negative, top-down reconciliation in the MinT family of constrained projections; it is exact by construction rather than a statistical estimate of the covariance structure.

The country total is therefore exactly the sum of its Admin-1 units. The measured maximum differences are **4.37e-11** (SREHI) and **3.73e-09 MW** (FRP) for Admin-1 -> country, and **5.82e-11** / **1.86e-09 MW** for country -> South America. The minimum reconciled value is **>= 0.0**.

M3 trades a small amount of point error for coherence: SREHI RMSE **12.8838** (R-squared **0.8171**), FRP RMSE **833.9594** (R-squared **0.7779**). M2 remains the point-error champion and is reported alongside M3.

---

## 7. Validation design

### 7.1 Frozen 200 km spatial blocks

The overlap table is partitioned into **5 folds** using **200 km spatial blocks**. A block is assigned to exactly one fold; every cell-month in that block inherits the block's fold. This guarantees that no 50 km cell and no spatial block appears in both training and validation. The split manifest stores the fold assignment, block assignment, fold sizes, and SHA-256 fingerprints.

### 7.2 Temporal splits

In addition to the spatial folds, the pipeline records temporal folds over the pre-drift window. The temporal ordering is used by the M3 simplex warm-up and by the trend/onset analytics. The validation code checks that every training timestamp is less than or equal to the corresponding validation timestamp for the temporal folds.

### 7.3 Leakage controls

- The climatology is frozen to **2001-2020**. A test perturbs the 2024 values by a factor of 3 and confirms that the climatology file is bit-identical.
- The M2 threshold and sigma scale are fitted on outer-training folds only.
- The M3 blend is temporal: no future fold contributes weights to an earlier fold.
- The static-cell mask is computed globally, but it is a deterministic quality-control rule, not a target-derived feature.
- The overlap hash is checked against the frozen split manifest before training.

---

## 8. Relay continuity diagnostic

For each relay year, the pipeline computes a 12-month mean before and after the boundary and forms the ratio

\[
\rho = \frac{\bar{y}_{\text{after}}}{\bar{y}_{\text{before}}}.
\]

The reduction is measured in log space:

\[
R = 1 - \frac{|\log \rho_{\text{harmonized}}|}{|\log \rho_{\text{raw}}|}.
\]

A value near 100% means the relay step almost disappears. The committed SREHI reductions are **65.7%** (2002), **88.1%** (2012), **47.7%** (2018), **7.2%** (2021), and **99.9%** (2023). The aggregate absolute step discontinuity `|delta_step|` falls **92.84%**, from **2.1892** to **0.1567**.

The 2021 reduction is intentionally the weakest. Orbital drift is a physical change in sampling, not a sensor addition; a model that forced a 99% reduction there would be overfitting the diagnostic. The 2023 reduction is the strongest because the VIIRS backbone is deduplicated and the relay is well observed.

### 8.1 Cross-regime guard

The guard compares RMSE in `TERRA_AQUA` and `TERRA_ONLY` regimes. All checks report `passed: true`. For M3 SREHI the RMSE ratio is **1.2572** (**11.5281** vs **14.4926**); for FRP it is **1.3718** (**699.5017** vs **959.5740**). The guard is a bias check: it ensures the reconstruction does not collapse or explode when the sensor mix changes. It does not assert that the two regimes are physically identical.

---

## 9. Seasonal calendar mathematics

### 9.1 Frozen climatology, 2001-2020

For each geography and calendar month \(m\), the climatology stores the count (`n_years`), mean, standard deviation, and quantiles \(p_{25}, p_{50}, p_{75}, p_{90}, p_{95}\) of the harmonized SREHI and FRP over the years **2001-2020**. The year 2000 is excluded because it is a two-month start; 2021-2026 are excluded to prevent the post-drift regime from contaminating the baseline. The sample is unbalanced: `n_years` ranges from 1 to 20; 48.0% of Admin-1 cells have fewer than 20 years, 7.7% fewer than 10, and 1.6% at most 3.

The standardized anomaly is

\[
z_{i,t} = \frac{y_{i,t} - \mu_{i,m}}{\sigma_{i,m}},
\]

with \(z=0\) when the standard deviation is missing or zero. The percent of median is

\[
\mathrm{pct}_{i,t} = 100 \frac{y_{i,t}}{p_{50,i,m}},
\]

with zero when the median is missing or zero.

### 9.2 Severity tiers

The monthly severity tier is a deterministic rule:

- `CRITICAL_EXTREME` if \(y > p_{90}\) or \(z > 2\);
- `HIGH_UNUSUAL` if \(y > p_{75}\);
- `MODERATE` if \(y > p_{25}\);
- `LOW_NORMAL` otherwise.

The tiers are descriptive, not probabilistic. They are calibrated against the frozen baseline, so a "critical" month in 2024 is critical relative to 2001-2020, not relative to 2024 itself. If `n_years < 10`, the tier is `LOW_SAMPLE`; `p90` and `z` are not used as a critical threshold, and the UI suppresses the p90 ring.

### 9.3 Theil-Sen trend

For a series \((t_j, y_j)\), the Theil-Sen slope is

\[
\hat{\beta} = \mathrm{median}_{j<k}\left(\frac{y_k-y_j}{t_k-t_j}\right).
\]

The intercept is the median of \(y_j - \hat{\beta}t_j\). The confidence interval is the standard Sen interval at the requested level. The implementation uses decimal years for \(t\).

### 9.4 Hamed-Rao modified Mann-Kendall

The classical Mann-Kendall statistic is

\[
S = \sum_{j<k}\mathrm{sign}(y_k-y_j),
\qquad
\mathrm{Var}(S) = \frac{n(n-1)(2n+5)-\sum_p t_p(t_p-1)(2t_p+5)}{18},
\]

where \(t_p\) are tie-group sizes. The Hamed-Rao correction replaces \(\mathrm{Var}(S)\) with

\[
\mathrm{Var}^{*}(S) = \mathrm{Var}(S)\left(1 + \frac{2}{n(n-1)(n-2)}\sum_{k=1}^{n-1}(n-k)(n-k-1)(n-k-2)\rho_k\right),
\]

where \(\rho_k\) is the lag-\(k\) autocorrelation of the detrended series. The default implementation models \(\rho_k=\rho_1^k\) (geometric AR(1), Yue-Wang 2004); the optional exact-lag mode uses the sample autocorrelation of the ranks. The standardized statistic is \(Z=(S-\mathrm{sign}(S))/\sqrt{\mathrm{Var}^{*}(S)}\), and the two-sided p-value is \(2(1-\Phi(|Z|))\). The effective sample size is reported with the test.

### 9.5 Benjamini-Hochberg FDR

For \(M\) p-values ordered \(p_{(1)}\le\cdots\le p_{(M)}\), the BH procedure finds the largest \(k\) such that

\[
p_{(k)} \le \frac{k}{M}q,
\]

and rejects the first \(k\) hypotheses. The implementation reports adjusted q-values and a boolean `significant_fdr` column at **q < 0.05**.

Committed results: **45 of 242** Admin-1 units are significant in SREHI (**34 increasing**, **11 decreasing**). **Peru** is the only significant country (slope **+903.23/yr**, z **3.34**, q **0.0134**).

### 9.6 GEV return levels

For each geography and year, the block maximum is the largest monthly SREHI/FRP value; the GEV is fitted to those annual maxima by **L-moments (probability-weighted moments)** with a parametric bootstrap. If the L-moment fit is invalid, the code falls back to MLE. SciPy uses the shape convention \(c=-\xi\), so the implemented return level for a return period \(T\) is

\[
x_T =
\begin{cases}
\mu + \frac{\sigma}{\xi}\left[1-\left(-\log(1-1/T)\right)^{-\xi}\right], & \xi\ne 0,\\
\mu - \sigma\log\left(-\log(1-1/T)\right), & \xi=0.
\end{cases}
\]

The fit is accepted only when it converges, passes a Kolmogorov-Smirnov check, and has at least 10 active (non-zero) annual maxima and satisfies a sanity bound: the 10-year return level must be \(\le 10\times\) the observed annual maximum (and the 50-year level must be finite and \(\le 10\times\) that maximum). Parametric bootstrap CIs (500 replicates) are reported for every published fit (Admin-1 and country) where the bootstrap converges. In this run **475 of 510** series pass; **31** are rejected by `return_level_sanity` and **4** by `insufficient_active_years`; all 510 have the complete 2001-2025 grid (zero-fire years included). **25 of 26 country series** pass and all 475 published fits have bootstrap CIs (`gev_estimator="lmoments"`). The GEV is a screening diagnostic, not an operational return-period product; the 25/26 country coverage is the relevant stability check. The published rows satisfy `0 < rl10 <= rl20 <= rl50`, `rl10 >= median(annual maxima)`, `rl10 <= 10*max_observed`, `rl50 <= 10*max_observed`, and `-0.5 <= xi <= 0.5`. With 25 annual blocks, the observed maximum sits around the 26-year empirical return period, so `rl10 < max_observed < rl50` is expected and is not a fit failure. The committed `calendar_trends.parquet` reports 10-, 20-, and 50-year return levels, the `gev_input`/`gev_max_observed`/`gev_active_years`/`gev_reject_reason` fields, country-level bootstrap CIs where available, and a `gev_fit_ok` flag. Return levels are conditional on the short 26-year record and are not a substitute for a regional frequency analysis.

### 9.7 Ninety-day early onset

The daily series is scaled by the monthly harmonized-to-raw ratio,

\[
\tilde{y}_{i,d} = y^{\text{daily}}_{i,d}\frac{\hat{E}_{i,m(d)}}{\max(E_{i,m(d)},1)},
\]

so the daily curve inherits the constellation scale of the month. For each day \(d\), the 90-day trailing sum is

\[
C_{i,d} = \sum_{k=0}^{89}\tilde{y}_{i,d-k}.
\]

The window crosses year boundaries and is a true trailing 90-day sum. The first 89 days of the record are `INSUFFICIENT_HISTORY` and are excluded from threshold estimation. The thresholds \(p_{75}(d)\) and \(p_{90}(d)\) are frozen from the same calendar day in **2001-2020**, using only complete windows and at least 10 years per calendar day; otherwise the flag is `NO_THRESHOLD`. The onset flag is

\[
\text{flag}_{i,d} =
\begin{cases}
\text{EARLY\_ONSET\_CRITICAL} & C_{i,d} > p_{90}(d),\\
\text{EARLY\_ONSET\_HIGH} & C_{i,d} > p_{75}(d),\\
\text{NORMAL} & \text{otherwise}.
\end{cases}
\]

The curve is computed on a 5-day step for the API. The 2026 record is a partial year through November and is labelled `PARTIAL_MONTH` / NRT where appropriate.

### 9.8 Spatial colour normalization, pagination and scientific states

The spatial observatory must answer two different visual questions without letting an extreme outlier wash out the continent: *where is activity highest in this month?* and *how does this month compare with the rest of the selected range?* The API therefore publishes two independent scale objects.

For a list of cell values \(v_1,\dots,v_n\), the empirical quantiles are computed with linear interpolation on the sorted sample. The scale object is

\[
\text{scale} = \{\max, p_{99}, p_{95}, \text{ceiling}\},
\qquad
\text{ceiling} =
\begin{cases}
p_{99} & p_{99} > 0 \ \text{and}\ \max > 1.5\,p_{99},\\
\max & \text{otherwise}.
\end{cases}
\]

All four values are rounded to four decimals; an empty or all-zero sample returns \(\max=p_{99}=p_{95}=0\) and \(\text{ceiling}=1\). The cap rule keeps a single enormous cell from flattening every other cell to the first colour ramp stop.

- `summary.month_scale` is computed from **every** cell of the requested month, before the `offset`/`limit` slice. A small page therefore never changes the scale.
- `reference_scale` is computed once over the effective interval `[range_start, range_end]` (defaulting to the catalogue bounds) and cached under `(resolution, iso_a3, start, end)`. Every animation frame in the same interval reuses it, so the `fixed` scale is stable during playback.
- The `50 km` layer always uses the M3-harmonized `harmonized_srehi` / `harmonized_frp_mw` columns. The `10 km` point-radius layer has no M3 calibration and is explicitly labelled `RAW_OVERPASS_COUNT`; the two value types are never mixed in one scale.

The spatial response also carries an exact pagination contract. For a single month the service fetches all matching rows ordered by `srehi DESC, cell_id ASC`, computes `total_available_cells`, the month scale, and then slices `cells[offset:offset+limit]`. `summary.truncated` is true when `offset + returned_cells < total_available_cells`. The default `limit` is `10,000` and the maximum is `20,000`; the month layer is bounded by the 50 km grid, so this is sufficient for every frame.

Scientific states are resolved in a fixed priority order so the interface never confuses "no data" with "no fire":

| State | Condition |
|---|---|
| `ASSET_MISSING` | the requested cube is not installed locally |
| `OUT_OF_CATALOG` | the month lies outside `2000-11..2026-11` and no rows exist |
| `ZERO_DETECTIONS` | an in-catalogue month with a valid AOI but no matching cells |
| `PARTIAL_MONTH` | the month or its maturity metadata is partial |
| `NRT_PRELIMINARY` | near-real-time rows that may still be revised |
| `DETECTIONS_PRESENT` | at least one cell and standard-science maturity |

Sensor participation is read from the monthly cube's `sensor_slot` column when present and otherwise inferred from the documented constellation epochs `C0_TERRA_ONLY` through `C4_MODIS_VIIRS_TRIO`. The inferred epoch boundaries are `2002-07`, `2012-01`, `2018-01` and `2024-01`.

---

## 10. Rejected alternatives

The following options were considered and are not used in the final product:

1. **A single global MODIS-to-VIIRS multiplier.** Rejected because the ratio varies by biome, fuel load, cloud cover, diurnal cycle, and confidence. A global scalar reduces one boundary and amplifies another.
2. **Raw detection counts as the target.** Rejected because pixel fragmentation makes the count a function of sensor footprint. Overpass events are more stable.
3. **Random cross-validation.** Rejected because neighbouring cell-months share fires and spatial blocks. A random split leaks and inflates R-squared.
4. **Fitting the climatology on the full record.** Rejected because 2021-2026 contains the drift era and would contaminate the baseline. The frozen 2001-2020 baseline is tested by perturbation.
5. **A hard hurdle threshold fixed globally.** Rejected because the optimal threshold varies by fold and target. The fold-wise threshold is used only when it beats the soft expectation.
6. **Unconstrained stacking.** Rejected because negative weights would make the blend uninterpretable and could produce negative fire activity. The simplex constraint and non-negativity are enforced.
7. **Post-hoc calibration on the validation fold.** Rejected as leakage. All sigma scales and thresholds are fitted on outer-training data.
8. **Point markers or kernel density as the primary map.** Rejected because they imply a precision the 10 km and 50 km grids cannot support, and because KDE bandwidth is an arbitrary visual choice. Equal-area cells are the honest cartographic unit.
9. **Municipal choropleths.** Rejected because the detection footprint is larger than most municipalities. A municipal polygon would imply attribution accuracy that the sensor does not provide.
10. **A universal GEV fit with no convergence check.** Rejected because short annual series can produce unstable shape parameters. The fit is gated by convergence and a KS check.

---

## 11. Uncertainty and limitations

- FIRMS detections are **not burned area**. A detection is a thermal anomaly at the overpass time; clouds, smoke, and sensor saturation can hide or distort it.
- The event count is a **sampling proxy**, not a physical fire count. One large fire can produce several events; one event can contain many pixels.
- The 10 km and 50 km cells are analysis units, not jurisdictional boundaries. Admin-1 labels near a border carry an epistemic uncertainty of roughly half a cell width (**about 5 km** at 10 km, **about 25 km** at 50 km).
- The M3 90% interval is a predictive band from the ensemble and the fold-wise sigma calibration. It is not a full Bayesian posterior and does not include model-selection uncertainty.
- The trend test controls FDR across the tested units, but it does not establish causality. A significant slope can reflect land-use change, climate variability, or sampling change.
- The GEV return levels are based on a short record and are sensitive to the shape parameter. Only series passing the sanity bound are published; country-level CIs are available where the bootstrap converges, and they are wide. Read them as a screening tool, not as a risk probability.
- The 2026 record is partial through November. NRT and SP flags and the data-maturity column distinguish preliminary from standard-science data.
- The calibration overlap is 2012-2020. The pre-2012 record is reconstructed by the bridge and carries larger uncertainty than the overlap period.
- The SAMA suspect flag never activates after the nominal+ confidence filter; it removes no rows. The `sama_suspect_det_count` column in the committed cubes is a night-detection count. The SAMA mask does not recover the true fire signal in the anomaly region.
- The static-source mask is an OR of persistence and type-1/2 dominance: 358 cells (153 persistent, 304 type-dominant, 99 both) and 2.87M detections are removed. `type` is absent in VIIRS NRT, so the type rule is asymmetric.
- The 50 km cube assigns each cell to one country; 485 cells straddle more than one country. Country totals reflect that assignment.
- N21 de-duplication is assumed equal to N20; there is no independent N21/MODIS overlap inside 2012-2020. The 2023 step depends on that assumption.
- The M3 blend orders spatial folds by median time, not by a strict calendar cut. The fitted M1/M2 weights are [0, 1], which limits the practical effect.
- Trends include 2001-2011 reconstructed values; their uncertainty is not propagated. `low_sample` marks n<10, and the BH FDR family is global across SREHI/FRP and Admin-1/country.
- The country/month panel has 4,014 rows, not 14x313=4,382. Missing country-months mean no detections, not missing data; FLK contributes 8 months (2024+), GUF 268, SUR 299, GUY 311, ECU/URY 312.

These limits are stated in the application, the API metadata, and this documentation. No claim in this project depends on treating a proxy as a direct physical measurement.

---

## Scientific References

The methods above are based on, or informed by, the following publications.
Citations support scientific attribution and reproducibility; the project does
not reproduce the copyrighted text of these works.

1. Giglio, L., Schroeder, W., & Justice, C. O. (2016). *The collection 6 MODIS active fire detection algorithm and fire products*. Remote Sensing of Environment, 178, 31–41. <https://doi.org/10.1016/j.rse.2016.02.054>

2. Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. A. (2014). *The New VIIRS 375m active fire detection data product: Algorithm description and initial assessment*. Remote Sensing of Environment, 143, 85–96. <https://doi.org/10.1016/j.rse.2013.12.008>

3. Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). *Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization*. Journal of the American Statistical Association, 114(526), 804–819. <https://doi.org/10.1080/01621459.2018.1448825>

4. Wickramasuriya, S. L., Turlach, B. A., & Hyndman, R. J. (2020). *Optimal non-negative forecast reconciliation*. Statistics and Computing, 30(5), 1167–1182. <https://doi.org/10.1007/s11222-020-09930-0>

5. Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering, 82(1), 35–45. <https://doi.org/10.1115/1.3662552>

6. Hamed, K. H., & Rao, A. R. (1998). *A modified Mann-Kendall trend test for autocorrelated data*. Journal of Hydrology, 204(1–4), 182–196. <https://doi.org/10.1016/S0022-1694(97)00125-X>

7. Sen, P. K. (1968). *Estimates of the Regression Coefficient Based on Kendall's Tau*. Journal of the American Statistical Association, 63(324), 1379–1389. <https://doi.org/10.1080/01621459.1968.10480934>

8. Theil, H. (1950). *A rank-invariant method of linear and polynomial regression analysis*. Indagationes Mathematicae, 12, 85–91. (No DOI was assigned to the original 1950 record; cited as published.)

9. Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society: Series B, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>

10. Hosking, J. R. M. (1990). *L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics*. Journal of the Royal Statistical Society: Series B, 52(1), 105–124. <https://doi.org/10.1111/j.2517-6161.1990.tb01775.x>

11. Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography, 40(8), 913–929. <https://doi.org/10.1111/ecog.02881>

12. Pinto, O., et al. (2003). *South Atlantic Magnetic Anomaly: energetic particle precipitation and satellite detector effects*. Journal of Atmospheric and Solar-Terrestrial Physics, 65(16–18), 1393–1404. (No DOI was located for this record; cited as published.)
