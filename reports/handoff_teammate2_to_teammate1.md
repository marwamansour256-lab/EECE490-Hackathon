# Handoff: teammate 2 (Marwa) → teammate 1 (Omar)

*Written 2026-10-17 by Marwa's Claude Code session, for Omar and his Claude Code session. Read [AGENTS.md](../AGENTS.md) first; these tasks follow its rules (raw lines retained, no "synthetic" claims, associations are not causes, external inputs never enter the strict model).*

## What teammate 2 added (already pushed)

| File | What it is |
|---|---|
| `src/metric_reconciliation.py` → `reports/metric_reconciliation.md/.json` | Every disputed number recomputed under each document's definition, with one proposed definition each. **Needs your agreement.** |
| `src/external_factor_test.py` → `reports/external_factor_test.json` | Held-out half/half test of outside factors, run in both directions, with circular-shift placebo, Holm correction and a planted-effect positive control. 0 of 33 tests pass. |
| `src/why_drivers.py` → `reports/why_drivers.json` | Breakdowns: profit growth drivers, Garden's margin gap, held-out explained variance against the Poisson noise ceiling (92.7% vs 93.4%), and the Tuesday profile. |
| `src/correlation_scan.py` → `reports/correlation_scan*.{csv,json}` | All 203 order-level pairs, measured in each half; pair lifts vs shuffles. |
| `app/planner.html`, `src/pilot_report.py`, `data/pilot/pilot_log_template.csv` | Demo interface and pilot logging (teammate 2's AGENTS.md role). |

## Your tasks, in priority order

### 1. Agree or amend the metric definitions

Read `reports/metric_reconciliation.md`. For each of the six metrics, accept the proposed definition or propose another.

- **Accept:** in `src/metric_reconciliation.py`, change the status sentence in `lines_md` from "proposed, pending agreement with Omar" to "agreed by Omar and Marwa, <date>". Rerun `python src/metric_reconciliation.py`, then commit.
- **Amend:** edit the `PROPOSALS` entry, rerun, and say why in the commit message.

**Done when:** the status line says agreed, and ANALYSIS.md uses only the agreed figures (task 2).

### 2. Update ANALYSIS.md (your file; teammate 2 did not edit it)

- **Item 6, "Grow the order at the till":**
  - Give the exact single-product share: 53.9% of orders contain one distinct product.
  - Replace "the most common pairing is a matcha or iced latte with a pastry" with the tested result. In orders with two or more categories, a drink + food combination appears 59% of the time vs 50% under label shuffles, in both halves. No specific category pair goes together beyond chance: Bakery + Matcha & Tea has a lift of 0.96–1.01. Source: `src/correlation_scan.py`.
  - So the test should be a generic "any drink + any food" prompt. A matcha + pastry bundle can be one arm, but it isn't supported as the best pairing.
- **Item 7, "Plan preparation and staff by the calendar":**
  - The Tuesday figure 1.66× is correct only as "vs the average of the other six days, including the weekend". The proposed figure is **1.53× a normal weekday** (per day, vs Mon and Wed–Fri). Use whichever is agreed in task 1, with its wording.
  - Add that the out-of-sample evidence is now explicit. Branch + growth + weekday explain 92.7% of daily orders on months the model never saw. The ceiling set by random arrivals is 93.4%, which leaves under 1% for outside events. Separately, 0 of 33 held-out factor tests pass. Sources: `reports/why_drivers.json` and `reports/external_factor_test.json`.
- **Team table:** mark "Omar, with Marwa: agree one definition..." as done once task 1 is agreed.

### 3. Run your event list through the held-out test

`data/external/external_events.json` has 15 events, but the held-out test only uses its own hand-picked strike dates.

- In `src/external_factor_test.py`, build one daily 0/1 series per testable event from `external_events.json` and add each to `factors`. Add each event's targets to `PLAN`, using `affected_daypart` and `affected_category` where they are set, otherwise `orders`.
- Suggested events:
  - `WAR-2026-04-08-STRIKES`
  - `CEASEFIRE-2026-04-17`
  - `AUB-COMMENCEMENT-2026-06-05`
  - `DST-2026-03-29`
  - `HOLIDAY-EID-AL-FITR-2026-GOV-CLOSURE`
- Skip events that cover most of a half (`DISPLACEMENT-...`, `HOLIDAYS-2026-H1-NATIONAL`, `PRICES-...`). A long window can't be tested with a half/half split because it is confounded with the trend.
- Skip the `not_found` events, or keep them labelled as alerts.
- Keep the existing factors so the Holm correction counts every test. Rerun with `python src/external_factor_test.py`; use `--quick` while editing.
- **Expected result:** single-day events will not pass. The positive control shows an 11-day flag is detected only at about 20% effects. Record that in each event's `association_in_data` and `validation_status` as "held-out test: not supported; detectable size ≈ X%". Never write "no effect".
- **Done when:** every testable event in `external_events.json` has a held-out result recorded, and `reports/external_factor_test.json` lists them.

### 4. Add the Tuesday question to the Stories request

Add the following to `reports/company_information_request.md`, section C or a new section F:

> **Tuesdays:** every Tuesday (26 of 26) has about 1.5× the orders of a normal weekday, at all three branches, at every hour, for every category, for loyalty customers and walk-ins alike, with the same order value. What happens on Tuesdays (a standing offer, a market or university day, delivery/stock day, how sales are recorded or batched)? Saturdays run at about 0.7×. Are branches open shorter hours or busy with something not in the till data?

If it fits, also add to the existing question 16:

> Can one person appear both with a Customer ID and as a walk-in?

### 5. When branch locations arrive

- Set `branch_exposure` in `external_events.json` to verified/unverified for each event.
- In `src/external_factor_test.py`, keep only the confirmed location's weather in `WEATHER_FILES`. Fewer locations means fewer tests and a smaller multiple-testing penalty.
- Rerun, then update notebook 05's README note and the "Held-out test of outside factors" section in README.md.

## Please don't

- Don't change `app/`, `src/pilot_report.py` or the pilot-log schema without telling Marwa. AGENTS.md lists them as teammate 2's contract.
- Don't quote these old pitch numbers: 51% single-item, 91% "retention", −23% promo profit, "demand is a habit", "the data is synthetic". The reconciliation and README replace them.
- Don't hand-edit `app/why_data.js` or `app/evidence_data.js`; they are generated by the scripts.
