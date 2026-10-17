# Reports index

Generated results and team documents. Most files are written by a script in `src/`; rerun the script rather than editing them by hand. `python src/verify_submission.py` checks that the main outputs reconcile.

## Team documents (Markdown)

| File | What it is | Owner |
|---|---|---|
| `company_information_request.md` | Questions for Stories (branch locations, costs, Tuesdays, shared IDs, counter layout) | Omar (teammate 1) |
| `metric_reconciliation.md` | One agreed definition for each disputed number | Marwa, agreed with Omar |
| `source_reconciliation.md` | Official sources for event dates | Charbel |
| `handoff_teammate2_to_teammate1.md` | Tasks Marwa handed to Omar | Marwa |
| `recommendation_evaluation.md` | Basket recommender evaluation | Charbel |
| `methodology_audit.txt` | Corrections to earlier claims | Charbel |
| `deeper_research_and_build_plan.txt` | Research and pilot specification | Charbel |

## Preparation model (src/strict_model.py)

`strict_half_results.json` (protocol, tuning, v2 details, metrics), `strict_half_metrics.csv`, `strict_half_predictions.csv`, `strict_half_finance.csv`, `strict_half_manifest.json` (verification). Earlier rolling/Bayesian experiments: `model_*`, `preparation_*`, `decision_cards.json`, `intraday_*`, `decision_run_manifest.json`.

## Basket recommender, layouts and growth plan

`recommendation_*` (src/basket_recommender.py), `wejha_*` (src/wejha_layout.py), `growth_demo.json` and `growth_preparation_cards.json` (src/growth_plan.py).

## Outside factors and events

| Files | Script |
|---|---|
| `external_factor_test.csv/.json` | src/external_factor_test.py (held-out test, 0 of 39 pass) |
| `external_analysis_results.json`, `external_associations.csv`, `external_feature_correlations.csv`, `external_sensitivity.csv`, `external_evidence.json` | src/external_analysis.py |
| `holiday_robustness.*`, `ramadan_daypart.csv`, `monthly_iced_mix.csv`, `season_category_summary.csv` | src/holiday_robustness.py and related |
| `outlier_dossiers.json`, `outlier_fingerprints.csv`, `daily_anomaly_audit.csv` | src/outlier_forensics.py |
| `context_v2_*` | src/context_model.py, src/context_mix_model.py, src/context_associations.py, src/context_robustness.py |

## Explanations and diagnostics

`why_drivers.json` (src/why_drivers.py), `correlation_scan*` (src/correlation_scan.py), `day_ledger.csv` (src/day_ledger.py), `daily_why_ledger.*` (src/day_explanations.py), `metric_reconciliation.*` (src/metric_reconciliation.py), `product_economics.csv`.

## Manifests

`*_manifest.json` and `run_manifest.json` record versions and SHA-256 hashes for reproducibility.

## Figures

`figures/` holds every saved chart: `1_` to `9_` from src/eda_visuals.py, `pitch_*` from src/pitch_visuals.py, model and evidence charts from their scripts, and `figures/ui/` demo screenshots.
