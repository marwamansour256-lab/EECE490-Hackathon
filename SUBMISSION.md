# Stories: submission and demo quickstart

**Charbel Manih — 202502099 · Omar Al Aghbari — 202514349 · Marwa Mansour — 202505788.** [GitHub accounts and responsibilities](TEAM.md).

The repository includes model source, trained artifacts, the original Stories CSV, cached external features, generated evaluations and an offline application. No retraining or network service is needed to open the demo.

## Run

1. Pull `main`: `git pull --ff-only origin main` after preserving any local edits.
2. Open **`app/wejha.html`** for Basket Growth and display layouts; open **`app/planner.html`** for preparation, daily diagnostics, evidence and pilot logging.
3. For Python commands, use Python 3.11 and `pip install -r requirements-lock.txt` to match the tested saved-model environment, then run `python src/verify_submission.py`. `requirements.txt` retains broad dependencies for development; scikit-learn joblib artifacts are best loaded with the locked versions.

```bash
python src/basket_recommender.py recommend --branch Harbour --item "Butter Croissant" --daypart morning
python src/growth_plan.py --branch Harbour --item Espresso --daypart morning --output reports/growth_demo.json
python src/predict_strict.py --date 2026-06-30 --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0
python src/explain_day.py --date 2026-06-24 --branch "All branches"
```

## What each model does

| Component | Executable implementation | Saved/exported artifacts | Business use |
|---|---|---|---|
| Basket recommender | `src/basket_recommender.py` | `models/basket_recommender.json`, `app/recommendation_data.js` | Rank an available add-on using smoothed co-purchases, context and unit margin |
| Display-layout comparison | `src/wejha_layout.py` | `reports/wejha_layouts.json`, `app/wejha_data.js` | Compare adjacent/separated arrangements under a geometric exposure hypothesis |
| Frozen demand model | `src/strict_model.py`, `src/predict_strict.py` | `models/strict_half_model.joblib` | Forecast category/item demand using training-only fitted parameters |
| Preparation decision | `src/profit_policy.py` | `reports/decision_cards.json`, `app/preparation_data.js` | Choose opening quantity and top-up conditional on remaining stock, with cost/service assumptions |
| Integrated growth plan | `src/growth_plan.py` | `reports/growth_demo.json` | Connect food/drink offers to prep cards and explicit incremental-profit scenarios |
| External-context heads | `src/context_model.py`, `src/context_mix_model.py` | Three `models/context_v2*joblib` artifacts | Separate traffic, volume and menu-mix hypotheses; compare with internal controls |
| Every-day evidence | `src/day_explanations.py`, `src/explain_day.py` | `reports/daily_why_ledger.json` | Reconcile traffic/basket/margin changes and source-linked external hypotheses for all 181 dates |

## Ninety-second walkthrough

- Select Harbour and morning in Basket Growth, add Butter Croissant, and inspect the ranked add-ons, margins, support and smoothed probabilities. Mark the top item unavailable to see the recommendation change.
- In Wejha Lab, select a pastry pair and compare its distance in three arrangements. The separated layout is the proposed experiment, not a measured profit result.
- In the profit tab, keep incremental acceptance at zero first. Enter a clearly labeled assumption and costs to see the break-even requirement. Accuracy is not acceptance.
- In the planner, load a preparation card or use the CLI example. For Turkey & Cheese Sandwich, the frozen June 30 scenario (selected v2 method) forecasts 3.67 units, prepares 4 at opening and tops up 4 at 16:00 when none remain. The planner's built-in example card still shows the original method's 4.04.
- In Day Explorer, compare an unusual day's traffic, basket and margin changes; inspect external-event evidence and unresolved geography. Accounting changes are not causal event attribution.

## Evidence the submission can defend

The recommendation model is selected within Jan–Mar and frozen for Apr–Jun. In 8,801 multi-product test orders, recovered-item margin@3 is $0.5080/order versus $0.4956 for margin-popularity (+2.51%). Hit@3 is 13.29%; plain popularity is better at 13.74%. These are masked-basket relevance metrics, not measured incremental purchases.

The preparation control improves a conditional one-batch contribution simulation by $1,019.13 versus a seasonal-growth baseline at similar realized service. The selected v2 version (level calibration and full-training item shares, both chosen on the last 21 training dates) improves category WAPE from 25.21% to 24.74% and adds a further $1,010 (interval about $884-$1,126) with 528 fewer unsold units, while realized fulfillment stays above 90% (90.9%). Another frozen baseline earns more but serves fewer units. Inventory, waste, capacity and actual operating policy are unobserved; do not call the simulation achieved savings.

Display placement has no observed treatment labels. Its weighted distance scores compare the proposed mechanism; the profit effect must be measured against the company's current arrangement. Public holidays/weather/war/access evidence is sourced but branch aliases do not establish real exposure or customer religion.

## Checks and remaining company inputs

`python src/verify_submission.py` runs 39 unit tests and four artifact-reconciliation scripts. `node app/check_ui.js` checks the actual frontend ranking code, schemas and JavaScript syntax; the recorded run passed 72 query checks. Python/frontend parity also passes all **144 queries across both offer modes**. `python src/audit_item_share_selection.py` reproduces the v2 model's 20-candidate training-only share search. The committed UI screenshots in reports/figures/ui were visually inspected. This Codex session had no connected live browser; screenshot review and code-level checks are distinguished from live interaction testing.

Git LF/CRLF changes are tolerated for text-source hashes; changes to data values and binary artifacts are not. Exact dependencies are recorded in `requirements-lock.txt`; cached sources permit offline numerical reproduction.

Before operating in a real shop, confirm branch coordinates, storage/food-safety rules, current layout and service flow, product availability, preparation capacity/lead times, waste/labor costs and prospective pilot design. All three team members' names/student IDs have been confirmed in TEAM.md.
