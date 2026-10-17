# Shared instructions for the three-person Stories project

## Objective and ownership

**Immediate deadline sprint:** read [WEJHA_BUILD_PLAN.md](WEJHA_BUILD_PLAN.md) first. The user now requests a basket recommendation system and a front-display ("wejha") placement experiment, connected to the preparation/evidence tools. Codex owns the new backend, evaluation and generated recommendation/layout assets; Marwa owns new `app/wejha.html` and its planner link; Omar owns new `WEJHA_PILOT.md`, the separate basket-pilot schema and business/physical-feasibility review. The build plan specifies exact filenames, output contracts, evaluation rules and acceptance criteria. Those new interfaces are planned until their implementation commit is published. Do not duplicate another owner's files or invent pending results.

**Publishing authorization has changed:** the latest user explicitly requests ongoing pushes of **all task-related code, artifacts and documentation**, including README progress. Publish usable milestones immediately after appropriate checks. The earlier Markdown-only publication arrangement below is superseded. Preserve uncommitted team changes and exclude credentials, virtual environments and temporary files.

**Implemented integration handoff:** [GROWTH_INTEGRATION.md](GROWTH_INTEGRATION.md) documents `src/growth_plan.py`, optional complementary food/drink filtering, and generated `window.STORIES_PREPARATION` cards in `app/preparation_data.js`. Marwa retains ownership of `app/wejha.html`; optional UI controls can use this contract. The CLI already connects offers, perishable preparation, explicit profit scenarios and separately labeled retrospective evidence. 36 tests pass. Unrestricted recommendation metrics must not be relabeled as complementary-filter performance.

**Latest user clarification supersedes the earlier no-external-feature restriction:** use external context freely when sourced and time-aligned. Only sales labels/train/test outcomes must come from Stories. Do not use another retailer's sales as test data. Transaction-only strict_half is the CONTROL; external-context models and the every-day ledger are the active extension. Read EXTERNAL_CONTEXT.md. Retain chronology, source availability and all original baseline results.

Build the Stories Profit Planner described in README.md: actionable preparation decisions, external evidence with verified exposure, and prospective incremental contribution measurement. Read README.md and, once published, reports/deeper_research_and_build_plan.txt before changing the model or claims. Markdown is being published ahead of the local model artifacts; a missing pending artifact is not permission to invent its outputs.

**Control protocol:** `src/strict_model.py` and `src/predict_strict.py` use the first 90 dates for training and last 91 for testing, with transaction-only features by control design. The ACTIVE external heads retain Stories-only sales targets and training-only selection; externally supplied context may enter as described in EXTERNAL_CONTEXT.md. Archived issued forecasts imply day-ahead information, not a March31-origin forecast of future weather. Observed weather/air and retrospective event labels cannot be mislabeled as advance forecasts. Twenty-three tests pass locally. The user now authorizes publishing the checked code and artifacts as well as documentation.

- Codex teammate owns src/strict_model.py, src/predict_strict.py and the saved artifact, plus src/demand_model.py, src/profit_policy.py, src/recommend.py, src/intraday_model.py, src/bayesian_policy.py, src/adaptive_recommend.py, tests and generated numerical results. Read MODEL_CARD.md for which results belong to each protocol.
- Codex also owns src/context_features.py, src/context_model.py (units and orders heads), src/context_mix_model.py, src/context_associations.py, src/context_robustness.py, src/day_explanations.py, src/explain_day.py, src/internet_checks.py and corresponding new model artifacts. Omar maintains event/geographic provenance; coordinate changes rather than recreating the same registry. The new cached source folder is data/external/context_v2/. Review reports/source_reconciliation.md before repeating searches for the road closures, race or clock-change announcement.
- Claude Code teammate 1 (Omar Al-Aghbari, confirmed) owns external evidence, geographic exposure verification and event/forecast provenance.
- Claude Code teammate 2 owns the presentation/demo interface and pilot logging built on the established data contracts.
- **Open handoff for teammate 1:** read [reports/handoff_teammate2_to_teammate1.md](reports/handoff_teammate2_to_teammate1.md) (metric definitions to agree, ANALYSIS.md updates, running the event list through the held-out test, the Tuesday question for Stories).
- These are collaboration boundaries, not permission to overwrite another person's uncommitted work. Coordinate contract changes. Do not launch agents or services just because this file describes multiple teammates.

## Repository and data discipline

- Inspect git status and fetch/pull safely before integrating teammates' work. Preserve local changes; no reset --hard, automatic conflict resolution that discards work, or force push.
- The user authorized publishing all task-related work as it progresses, including code, generated artifacts and Markdown. Stage reviewed task files explicitly; do not publish secrets or another person's incomplete edits. Do not contact company staff or teammates through external messaging tools without authorization.
- Preserve data/stories_transactions.csv. The brief describes real transactions with replacement IDs. SYN identifiers do not establish synthetic data. Branch aliases do not identify real neighborhoods.
- Retain repeated raw lines unless a genuine line ID or documented deduplication rule resolves them. Keep sensitivity analysis separate.
- Do not infer customer religion, acquisition date, price elasticity or causal war effects from the existing fields.
- Keep notebooks 01-08 as the team's original work unless assigned to modify one. Correct old claims in the shared audit and README. The original PDF is first-stage analysis; do not present it as covering the new policy. Notebook 05 now uses Batroun with Broummana/Beirut sensitivity; treat the North-region assumption as requiring company provenance, not inferred from aliases. Notebook 08 and its CPI sources are contextual research, not inputs to the strict model.

## Validation and financial claims

- Use chronological splits, shifted features and known-at timestamps. No random train/test split for this time series; no realized future weather or retrospectively selected conflict labels in an operational forecast.
- Preserve the original Jan-Mar training, Apr-May selection, June rolling evaluation record. New policy/event choices made after examining June are retrospective exploratory work. A new future period is needed for confirmation.
- For the chronological half/half experiments, all model selection occurs in the last 21 dates of Jan-Mar; refit once on Jan-Mar; April-June predictions do not use test sales outcomes. For external heads, issued context available before each forecast may enter. Preserve the earlier rolling record as a separate experiment. Never combine metrics from different information sets or call the previously explored dataset newly untouched.
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
- app/day_data.js and reports/day_ledger.csv (teammate 2, interim): one record per date x {All branches, Garden, Harbour, Uptown} with orders, expected_orders (frozen Jan-Mar Poisson), units, revenue, gross_profit, reference_gross_profit, bridge {traffic, basket, margin_mix} (Shapley), unexplained_orders, z, daypart_share and daypart_share_ref; the All-branches record also carries weather (three candidate locations) and events (id, status, exposure, mechanism, known_before, source). Produced by src/day_ledger.py until the Codex daily dossiers are published; a Codex export mapped to these fields can replace it without UI changes.
- reports/external_factor_test.json (teammate 2): held-out half/half factor tests (both directions, circular-shift placebo, Holm correction, planted-effect positive control). Retrospective association evidence only; never a forecast input.
- reports/daily_why_ledger.json: 181 dates x4 scopes (network and three branches). `actual`, `accounting_baseline`, `exact_accounting_bridge_usd`, `traffic_forecast`, product-unit `forecast`, `context_primary_proxy`, `alternative_geography_proxies`, and `event_and_mechanism_evidence`. Never label the accounting bridge or model adjustment a causal external attribution.
- `python src/explain_day.py --date 2026-06-24 --branch "All branches"` is the every-day evidence interface. A UI should show observed traffic/basket/margin changes, source-linked hypotheses, counterchecks and the unexplained remainder together.
- Published forecast data must distinguish previous_day1 forecasts from actual weather, air-quality archives, forecast errors and retrospective news. The latter are explanatory/oracle inputs, not pre-opening predictions.
- The teammate2 two-direction/Holm factor test and the expanded context models use different scopes and pass rules. Preserve both. A Poisson arrival-noise benchmark is conditional on that model, not a mathematical upper bound proving external factors cannot matter. Reverse-direction fitting is a diagnostic, not a deployable forecast of the past. Missing Customer ID means unidentified, not proven walk-in status.

### Frontend contract while model files are pending publication

**Preparation-control command:** `python src/predict_strict.py --date 2026-06-30 --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0`. It reads only the saved model artifact. Output includes `status`, `date`, `branch`, `product`, `category`, `method`, `trained_through`, `external_model_inputs`, `test_sales_used_in_forecast`, `forecast_mean_units`, `opening_prepare_units`, `one_batch_comparator_units`, `at16_topup_by_remaining_stock`, expected contribution fields, assumptions and (if requested) `prepare_at16_units`. The checked control example has forecast 4.03997 units, opening 4 units and topup 4 when remaining is zero. Do not mix these with the earlier rolling example below. The main day-diagnostic interface is now `src/explain_day.py`: combine its ledger, traffic forecast and evidence with the preparation control, and label each model's information set. Bayesian experiments remain separate.

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
