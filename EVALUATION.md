# Transaction-only control evaluation: chronological half/half

**User clarification:** external contextual inputs are permitted; external SALES targets are not requested. The no-external-feature restriction was an assistant misunderstanding. This document preserves the original transaction-only CONTROL. The active extension is documented in [EXTERNAL_CONTEXT.md](EXTERNAL_CONTEXT.md): it keeps Stories-only targets and January-March model fitting, but permits externally supplied features. Issued daily weather forecasts imply a day-ahead information set, unlike this control's complete March31-origin forecast. Actual weather, air archives and forecast errors remain retrospective diagnostics.

## Boundary

The forecast unit is a branch/category/day, then allocated to products. Keep complete days and orders together. Split the 181-day timeline into the earlier 90 dates and later 91 dates:

| Role | Dates | Days | Product lines |
|---|---|---:|---:|
| Training | Jan 1-Mar 31, 2026 | 90 | 22,395 |
| Testing | Apr 1-Jun 30, 2026 | 91 | 28,782 |

Transaction volume grows during the sample, so equal halves of time have unequal row counts. Randomly splitting product lines could place parts of the same order or future dates in training. A 50/50 split is a requested experiment, not a universal machine-learning rule.

## Training and selection

Within the 90 training dates, fit on the earlier 69 dates and validate on the final 21. Compare Poisson regularization values 0, 0.001, 0.01, 0.1 and 1. Select by mean category Poisson deviance. Compare seasonal baselines using the same training-only validation period. The trained Poisson model with alpha 0.001 wins this selection.

Refit its coefficients and baselines on the full 90 training dates. Freeze them before evaluating April-June. Item shares are smoothed from the last 28 training days; selling prices, ingredient costs and intraday mix use only the training period. Product and branch catalogs come from training. Unseen test products raise an error rather than disappearing silently.

## Allowed predictive inputs

The sole raw input file is `data/stories_transactions.csv`. Model features are branch/category identifiers, date-derived weekday, elapsed-time trend and category-specific trend. Product forecasts use training-only item shares. No weather, holidays, conflict labels, news, university calendars or other external files enter this model.

The entire April-June forecast is produced as of March 31. There are no test sales lags, test-period updates or refits. The prediction command loads the saved artifact and does not read the test CSV. Actual test quantities enter only evaluation and hypothetical inventory replay.

The operational replenishment demonstration can accept observed remaining stock at 16:00. This is a decision input for a separate two-stage scenario, not a hidden feature in the frozen test forecast. The primary finance table uses a simpler one-batch comparison so the inventory assumptions are explicit.

## Results

| Model | Category WAPE | Perishable item WAPE | Contribution scenario | Fulfillment |
|---|---:|---:|---:|---:|
| Trained Poisson | 25.2077% | 52.0318% | $10,934.29 | 92.0921% |
| Seasonal plus training-only growth | 26.6435% | 53.5775% | $9,915.15 | 92.1939% |
| Frozen seasonal | 28.2181% | 53.7521% | $13,409.52 | 88.2164% |

WAPE = sum of absolute forecast errors divided by total actual units. The model reduces category WAPE by 1.44 percentage points versus the growth baseline. In the one-batch scenario it improves contribution by $1,019.13, reduces unsold stock by 240 units and fulfills 13 fewer units. Fulfillment differs by about 0.10 percentage points.

A circular seven-day block resampling of test-day contribution differences gives an approximate 95% interval of **$770-$1,276** for the difference versus the growth baseline. This describes uncertainty conditional on fixed forecasts and hypothetical costs, not uncertainty in missing stockouts, real labor costs, model selection or future company ROI. It does not turn the previously explored test period into independent confirmation.

The frozen seasonal baseline earns higher contribution while fulfilling only 88.22% of recorded demand. That tradeoff must remain visible; it cannot be presented as equally good service. The common inventory constraint is 90% forecast expected unit fill, not guaranteed actual fulfillment or a 90% probability of no stockout.

All finance assumes recorded sales equal demand, no salvage, training-period economics, one preparation batch and no capacity, substitution or additional labor expense. Actual stockouts, preparation, discard quantities and current company policy are unavailable. These results do not show measured savings, net profit or superiority over current company operations.

## Reproduction and artifact

```bash
pip install -r requirements.txt
python src/strict_model.py
python src/predict_strict.py --date 2026-06-30 --remaining 0
python src/verify_strict_artifacts.py
python src/strict_visuals.py
python -m unittest discover -s tests -v
```

- `models/strict_half_model.joblib`: fitted pipeline, training cutoff, item shares/economics, baseline parameters and source hash.
- `reports/strict_half_results.json`: protocol, training-only selection, metrics and assumptions.
- `reports/strict_half_predictions.csv`: all branch-item test predictions and the one-batch scenario.
- `reports/strict_half_metrics.csv` and `reports/strict_half_finance.csv`: comparisons.
- `reports/strict_half_manifest.json`: package versions, artifact/input/source hashes and reconciliation checks.
- `reports/figures/strict_half_profit_service.png`: presentation figure showing accuracy and the service/profit tradeoff.

These implementation artifacts are in the Codex working copy; Markdown publication precedes separate code publication. Tests verify that changing test labels does not change the frozen model/baseline predictions and check economics, mathematical inference and inventory accounting. Independent verification reconciles all **36,855** item-day-method records, category-to-item forecast sums, stock/demand/cash accounting, fixed test-period economics and predictions freshly produced from the saved model.

## Interpretation limits

The whole dataset had been explored before this new protocol was requested. We can enforce a clean computational training/test boundary, but cannot claim the test data was newly unseen by the team. Further feature or policy choices based on these test results would require another future evaluation period.

The earlier rolling experiment used Jan-Mar initial training, Apr-May selection and June rolling testing. The Bayesian extension used that earlier setup. Those results are supplemental, with different horizons and information sets. Do not compare their June scores directly with the frozen April-June metrics or combine their profit numbers.

External research stays in a separate explanatory evidence layer. A sourced event may suggest an operational question, but it does not become a predictive feature or causal explanation without a permitted future experiment and verified exposure.
