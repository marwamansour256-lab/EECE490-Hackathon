# Shared instructions for the three-person Stories project

## Objective and ownership

Build the Stories Profit Planner described in README.md: actionable preparation decisions, external evidence with verified exposure, and prospective incremental contribution measurement. Read README.md and, once published, reports/deeper_research_and_build_plan.txt before changing the model or claims. Markdown is being published ahead of the local model artifacts; a missing pending artifact is not permission to invent its outputs.

**Latest protocol:** the primary model is `src/strict_model.py` and prediction CLI `src/predict_strict.py`: first 90 dates train, last 91 test; only transaction CSV inputs; inner validation within training; final parameters and all learned shares/economics frozen at March 31. No test lags or external predictors. Read EVALUATION.md. Earlier rolling/Bayesian results are supplemental and must not be described as strict half/half results. The model, artifact and 16 tests are complete locally; GitHub code publication is separate from the authorized ongoing Markdown pushes.

- Codex teammate owns src/strict_model.py, src/predict_strict.py and the saved artifact, plus src/demand_model.py, src/profit_policy.py, src/recommend.py, src/intraday_model.py, src/bayesian_policy.py, src/adaptive_recommend.py, tests and generated numerical results. Read MODEL_CARD.md for which results belong to each protocol.
- Claude Code teammate 1 (Omar Al-Aghbari, confirmed) owns external evidence, geographic exposure verification and event/forecast provenance.
- Claude Code teammate 2 owns the presentation/demo interface and pilot logging built on the established data contracts.
- **Open handoff for teammate 1:** read [reports/handoff_teammate2_to_teammate1.md](reports/handoff_teammate2_to_teammate1.md) (metric definitions to agree, ANALYSIS.md updates, running the event list through the held-out test, the Tuesday question for Stories).
- These are collaboration boundaries, not permission to overwrite another person's uncommitted work. Coordinate contract changes. Do not launch agents or services just because this file describes multiple teammates.

## Repository and data discipline

- Inspect git status and fetch/pull safely before integrating teammates' work. Preserve local changes; no reset --hard, automatic conflict resolution that discards work, or force push.
- The user authorized publishing completed Markdown as work progresses. For code/data publication, follow the user's subsequent instructions and the agreed team workflow. Do not contact company staff or teammates through external tools without authorization.
- Preserve data/stories_transactions.csv. The brief describes real transactions with replacement IDs. SYN identifiers do not establish synthetic data. Branch aliases do not identify real neighborhoods.
- Retain repeated raw lines unless a genuine line ID or documented deduplication rule resolves them. Keep sensitivity analysis separate.
- Do not infer customer religion, acquisition date, price elasticity or causal war effects from the existing fields.
- Keep notebooks 01-08 as the team's original work unless assigned to modify one. Correct old claims in the shared audit and README. The original PDF is first-stage analysis; do not present it as covering the new policy. Notebook 05 now uses Batroun with Broummana/Beirut sensitivity; treat the North-region assumption as requiring company provenance, not inferred from aliases. Notebook 08 and its CPI sources are contextual research, not inputs to the strict model.

## Validation and financial claims

- Use chronological splits, shifted features and known-at timestamps. No random train/test split for this time series; no realized future weather or retrospectively selected conflict labels in an operational forecast.
- Preserve the original Jan-Mar training, Apr-May selection, June rolling evaluation record. New policy/event choices made after examining June are retrospective exploratory work. A new future period is needed for confirmation.
- For the PRIMARY half/half experiment, all model selection occurs in the last 21 dates of Jan-Mar; refit once on Jan-Mar; April-June is forecast without any test outcomes. Preserve the earlier rolling record as a separate experiment. Never combine metrics from the two splits or call the previously explored dataset newly untouched.
- Compare forecasts under the same inventory policy and compare policies under the same forecast. Separate forecast value from the value of allowing replenishment.
- Strong baseline: preceding four matching weekdays at branch/category level, allocated using prior 28-day smoothed item shares. Do not replace it with a weaker baseline to improve the headline.
- Report contribution, fulfillment, unsold stock, unmet recorded demand, replenishment count and labor/cost assumptions together. Gross profit is not net profit; simulated savings are not measured company savings.
- Two-stage 90% expected unit-fill constraints apply separately to stages; one-batch uses the daily floor. These are not identical timing constraints or a 90% probability of no stockout. Never hide actual service differences.
- No-service-floor outputs are stress scenarios. June has already shown that a lower WAPE can yield lower service-constrained contribution.
- Real stockouts, preparation, waste, lead times, capacity and current operating policy are missing. Do not claim the replay beats the company's current policy or guarantees ROI.
- Event occurrence, exposure, association, predictive utility and causal effect are distinct. News found after selecting an outlier does not validate its cause. Preserve unsuccessful hypotheses and multiple-testing corrections.

## Data contracts

- data/external/outlier_event_registry.json: event date, known_at where verified, source URL, mechanism and countercheck. Unknown exposure stays unknown.
- reports/outlier_dossiers.json: rolling expected orders, observed residuals, branch/daypart fingerprints and candidate events. Profit gaps are not recoverable profits.
- reports/model_predictions.csv: original daily branch/category forecast comparison.
- reports/two_stage_replay.csv: scenario and forecast identifiers, branch/product/date, opening/replenished/sold/unsold/unmet units and both policy contributions.
- reports/decision_cards.json: historical date, forecast, branch, category, product, forecast means, prices/costs, opening units and remaining-stock-to-topup lookup. Do not expose replay actuals as decision inputs.
- src/recommend.py: stable CLI interface for historical cards or explicit custom scenario inputs. Any UI must display historical/scenario status and cost assumptions.
- data/pilot/pilot_log_template.csv (teammate 2): one row per branch/product/day with date, branch, product, arm (planner/current/shadow), recommended_opening, prepared_opening, remaining_at_16, recommended_topup, prepared_topup, sold, unsold_end, sold_out_time, unit_price, unit_cost, setup_usd_per_batch, extra_usd_per_topup_unit, card_source, override_reason, notes, logged_at. Written by app/planner.html, read by src/pilot_report.py. Unmet demand is not a column because a pilot cannot observe it.
- reports/external_factor_test.json (teammate 2): held-out half/half factor tests (both directions, circular-shift placebo, Holm correction, planted-effect positive control). Retrospective association evidence only; never a forecast input.

### Frontend contract while model files are pending publication

**Primary frontend command:** `python src/predict_strict.py --date 2026-06-30 --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0`. It reads only the saved model artifact. Output includes `status`, `date`, `branch`, `product`, `category`, `method`, `trained_through`, `external_model_inputs`, `test_sales_used_in_forecast`, `forecast_mean_units`, `opening_prepare_units`, `one_batch_comparator_units`, `at16_topup_by_remaining_stock`, expected contribution fields, assumptions and (if requested) `prepare_at16_units`. The checked primary example has forecast 4.03997 units, opening 4 units and topup 4 when remaining is zero. Do not mix these with the earlier rolling example below. Display strict-model results as the main model; offer Bayesian experiments separately.

Build against this illustrative structure, not invented model outcomes:

```json
{
  "status": "Historical replay; verify production inputs",
  "date": "2026-06-30",
  "branch": "Harbour",
  "product": "Turkey & Cheese Sandwich",
  "forecast": "strong_baseline",
  "forecast_mean_units": 4.933234421364985,
  "opening_prepare_units": 5,
  "one_batch_comparator_units": 6,
  "at16_topup_by_remaining_stock": {"0": 4, "1": 3, "2": 2, "3": 1, "4": 0, "5": 0},
  "assumptions": {"setup_usd_per_sku": 0.25, "extra_usd_per_topup_unit": 0.10, "expected_fill_floor": 0.90, "immediate_second_batch": true, "capacity_unconstrained": true},
  "event_adjustment": "None: exposure and predictive value unverified"
}
```

This example is the checked existing seasonal-planner card, not output of the Bayesian extension. Show cost settings, historical status, predicted demand, opening quantity and the remaining-stock input. Keep the calculation backend replaceable.

The intraday CLI is `python src/adaptive_recommend.py --candidate selected --remaining 0`. Candidate choices are `selected`, `best-adaptive`, and `seasonal-control`. Its output includes `candidate`, `adaptive_signal_enabled`, `opening_prepare_units`, `prior_late_mean_units`, `at16_topup_by_remaining_stock`, expected contributions, assumptions and comparison status. When remaining stock is supplied it adds `observed_early_sales_units`, `early_stock_exhausted`, `observation_interpretation`, `posterior_late_mean_units`, `posterior_p10_units`, `posterior_p90_units` and `prepare_at16_units`. All are historical/scenario outputs. Do not label the adaptive option as the selected winner; April-May selected the static NB candidate. See MODEL_CARD.md for results and limits.

External-evidence teammate: use a JSON list with event_id, event_start/end, published_at (timezone or null), source_url, affected_geography, branch_exposure (verified/unknown), mechanism, affected_daypart/category, and validation_status. Prioritize company-confirmed branch locations and store hours before more generic national event flags. Sources published after an event are retrospective evidence, never forecast-time features.

## Checks and documentation

Run python -m unittest discover -s tests -v after changes to inventory math or accounting. Run the affected analysis script and check its JSON/CSV outputs when model/features change. Avoid rerunning unrelated expensive analyses without a reason.

Keep README metrics consistent with generated outputs. Store sources and retrieval dates. Do not invent provenance, held-out results, team member names or missing operational data. Document unresolved requirements and provide runnable artifacts. Record package versions and input hashes in a run manifest for submission.
