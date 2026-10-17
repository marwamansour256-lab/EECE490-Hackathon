# Shared instructions for the three-person Stories project

## Objective and ownership

Build the Stories Profit Planner described in README.md: actionable preparation decisions, external evidence with verified exposure, and prospective incremental contribution measurement. Read README.md and, once published, reports/deeper_research_and_build_plan.txt before changing the model or claims. Markdown is being published ahead of the local model artifacts; a missing pending artifact is not permission to invent its outputs.

- Codex teammate owns src/demand_model.py, src/profit_policy.py, src/recommend.py, the new src/intraday_model.py and src/bayesian_policy.py, policy tests and generated numerical results. The intraday extension is in progress: posterior afternoon demand from observed earlier sales, with sold-out observations treated as censored. No improvement is claimed before evaluation.
- Claude Code teammate 1 (Omar Al-Aghbari, confirmed) owns external evidence, geographic exposure verification and event/forecast provenance.
- Claude Code teammate 2 owns the presentation/demo interface and pilot logging built on the established data contracts.
- These are collaboration boundaries, not permission to overwrite another person's uncommitted work. Coordinate contract changes. Do not launch agents or services just because this file describes multiple teammates.

## Repository and data discipline

- Inspect git status and fetch/pull safely before integrating teammates' work. Preserve local changes; no reset --hard, automatic conflict resolution that discards work, or force push.
- The user authorized publishing completed Markdown as work progresses. For code/data publication, follow the user's subsequent instructions and the agreed team workflow. Do not contact company staff or teammates through external tools without authorization.
- Preserve data/stories_transactions.csv. The brief describes real transactions with replacement IDs. SYN identifiers do not establish synthetic data. Branch aliases do not identify real neighborhoods.
- Retain repeated raw lines unless a genuine line ID or documented deduplication rule resolves them. Keep sensitivity analysis separate.
- Do not infer customer religion, acquisition date, price elasticity or causal war effects from the existing fields.
- Keep notebooks 01-07 as the team's original work unless assigned to modify one. Correct old claims in the shared audit and README. The original PDF is first-stage analysis; do not present it as covering the new policy. Notebook 05 now uses Batroun with Broummana/Beirut sensitivity; treat the North-region assumption as requiring company provenance, not inferred from aliases.

## Validation and financial claims

- Use chronological splits, shifted features and known-at timestamps. No random train/test split for this time series; no realized future weather or retrospectively selected conflict labels in an operational forecast.
- Preserve the original Jan-Mar training, Apr-May selection, June rolling evaluation record. New policy/event choices made after examining June are retrospective exploratory work. A new future period is needed for confirmation.
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

### Frontend contract while model files are pending publication

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

This example is the checked existing seasonal-planner card, not output of the new Bayesian extension. Show cost settings, historical status, predicted demand, opening quantity and the remaining-stock input. Keep the calculation backend replaceable. For the intraday extension, reserve fields for observed early sales, stockout flag, posterior mean/interval, chosen policy and comparison status; final schema will be documented after validation.

External-evidence teammate: use a JSON list with event_id, event_start/end, published_at (timezone or null), source_url, affected_geography, branch_exposure (verified/unknown), mechanism, affected_daypart/category, and validation_status. Prioritize company-confirmed branch locations and store hours before more generic national event flags. Sources published after an event are retrospective evidence, never forecast-time features.

## Checks and documentation

Run python -m unittest discover -s tests -v after changes to inventory math or accounting. Run the affected analysis script and check its JSON/CSV outputs when model/features change. Avoid rerunning unrelated expensive analyses without a reason.

Keep README metrics consistent with generated outputs. Store sources and retrieval dates. Do not invent provenance, held-out results, team member names or missing operational data. Document unresolved requirements and provide runnable artifacts. Record package versions and input hashes in a run manifest for submission.
