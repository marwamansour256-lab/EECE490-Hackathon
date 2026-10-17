# Stories Coffee: basket growth, display experiments and preparation

**EECE490 Hackathon team:** Charbel Manih (202502099), Omar Al Aghbari (202514349), Marwa Mansour (202505788). Roles are in [TEAM.md](TEAM.md).

We analysed Stories' January–June 2026 transactions from three branches: Harbour, Garden and Uptown, with 33,887 orders and 51,177 product lines. The implemented system ranks an add-on for the current basket, compares front-display arrangements, and tells each branch **how many of each fresh item to prepare at opening and at 16:00**. Models are fitted on earlier dates and evaluated chronologically; previously explored findings remain exploratory.

**Conclusion.** Stories' recorded January-to-June gross-profit growth of 52.5% came almost entirely from more orders, while 54% of orders contained one product. We built tools to test two repeatable decisions: **how much fresh food to prepare** and **which add-on to offer**. Fitted on Jan–Mar, v2 achieved 24.74% category WAPE on Apr–Jun versus 26.64% for the seasonal-growth baseline. Under a common one-batch policy it simulated $11,944 contribution versus $9,915, with recorded-demand fulfilment of 90.92% versus 92.19%; these are assumptions-based results, not company savings. Separate opening/16:00 recommendations require cheap, feasible replenishment. Garden's recorded unit costs explain 84% of its margin gap with Harbour; matching Harbour's costs gives an approximately $16k six-month scenario requiring invoice verification. No factor passed the strict 39-test external-event screen; a separate monthly-calendar traffic model improved WAPE slightly, without establishing preparation-profit gains. The dataset was explored before this frozen evaluation, so a randomised prospective pilot must confirm value. **[Full report mapped to the brief](PROJECT_REPORT.md) · [Final submission audit](SUBMISSION_AUDIT.md).**

- **Main model:** the **v2 preparation model**. It has the lowest category WAPE among the 12 approaches in the reported chronological comparison. Settings were selected within Jan–Mar; Apr–Jun was excluded from fitting, but had previously been explored by the team. Other demand models are retained as comparisons; the recommender, display optimizer and pilot reporter solve connected business tasks.
- **Demo:** open [`app/planner.html`](app/planner.html) (preparation plan) and [`app/wejha.html`](app/wejha.html) (add-on offers and display test) in a browser. No install or internet is needed.
- **Honesty rule:** results measured on held-out data are labelled as measured. Money figures that depend on assumptions are labelled as scenarios. Nothing here is a measured company saving; that needs the pilot at the end.

---

> ### Where is the model?
>
> | What | File |
> |---|---|
> | **Train the model** (Jan–Mar data, tested on Apr–Jun) | [`src/strict_model.py`](src/strict_model.py) → `python src/strict_model.py` |
> | **Use the model** (forecast + how many to prepare) | [`src/predict_strict.py`](src/predict_strict.py) → `python src/predict_strict.py --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0` |
> | **The trained model file** | [`models/strict_half_model.joblib`](models/strict_half_model.joblib) (selected method: `regularized_poisson_v2`) |
> | **Its results** | [`reports/strict_half_results.json`](reports/strict_half_results.json) and [`reports/strict_half_metrics.csv`](reports/strict_half_metrics.csv) |
> | **How it works, in words** | [`MODEL_CARD.md`](MODEL_CARD.md) |
> | **Models it beat** (linear regression, decision tree, random forest, XGBoost) | [`src/linear_baseline_check.py`](src/linear_baseline_check.py), [`src/tree_models_check.py`](src/tree_models_check.py), [`src/xgboost_check.py`](src/xgboost_check.py) |
> | **Add-on recommender** (second model) | [`src/basket_recommender.py`](src/basket_recommender.py), trained file [`models/basket_recommender.json`](models/basket_recommender.json) |
>
> Every saved model is listed in [`models/README.md`](models/README.md).

## 1. Business insights

| # | What we found | Evidence | What Stories should do |
|---|---|---|---|
| 1 | **Garden's low margin is a cost problem.** Garden's gross margin is 47%, against Harbour's 65%. Of the 18-point gap, **84% comes from higher unit costs**: every one of the 45 products costs about 43% more at Garden. Lower prices explain 14% and product mix 2%. | Same split in Jan–Mar and Apr–Jun ([why_drivers](reports/analysis_details.md#why-the-numbers-move-srcwhy_driverspy)) | Compare Garden's supplier invoices with Harbour's. At Harbour's recorded costs, Garden would have earned about **$16k more gross profit in six months**. This is a scenario until the cost definitions are confirmed. |
| 2 | **Growth comes from more visits, not bigger baskets.** Gross profit grew 52.5% from January to June: more identified customers active (37%), more visits each (28%) and more unidentified orders (34%). Basket size, prices and margins added **about 0%**. **54% of orders contain only one product.** | Exact decomposition, the same in both halves | Grow the basket: offer one add-on at the till. Our recommender ranks it (insight 6). |
| 3 | **Demand follows the calendar.** On months the model never saw, branch + growth + weekday explain **92.7%** of daily order swings. Every one of the 26 Tuesdays is busy: **1.53× a normal weekday**, at every branch, hour and product. Saturday runs at 0.70×, and every branch peaks at 21:00. | Held-out fit, both directions ([why_drivers](reports/analysis_details.md#why-the-numbers-move-srcwhy_driverspy)) | Plan preparation and staff by weekday and hour; this is what the v2 model does. Ask Stories what happens on Tuesdays and try to copy it on Saturdays. |
| 4 | **Time of day changes what and how people buy.** Morning items are 24% hot coffee and 27% bakery. Evening items are 17% iced coffee, 17% matcha/tea and 11% desserts. Takeaway is 54% of daytime orders but 28% of evening orders. | All 203 variable pairs measured separately in each half; 202 are stable ([correlation scan](reports/analysis_details.md)) | Bake for the morning, prepare desserts for the evening, and staff the seating area at night. |
| 5 | **No factor passes the strict two-direction external-event test.** Weather at three candidate locations, holidays, Ramadan, the March–April war period and Beirut strike days: **0 of 39** tests improve prediction of unseen days after a multiple-testing correction. The specified planted 20% scenarios were detected consistently; this does not guarantee detection of every real effect. The separate calendar-conditioned traffic head improves WAPE slightly (9.46% to 9.27%). | [Held-out factor test](EXTERNAL_CONTEXT.md), [event dossiers](EXTERNAL_CONTEXT.md) | Treat events as review alerts for the manager, not automatic forecast changes. Confirm branch locations first, because exposure is unknown. |
| 6 | **"Drink + food" is real; specific pairings are not.** Orders with two categories pair a drink with a food 59% of the time, against 50% by chance. The team category-pair check found no robust distinctive pairing (largest category-pair lift about 1.10); noisy SKU-level lifts are not that same test. All 28 pastry pairs used in the display test have lift below 1. | Shuffle tests; recommender test on 8,801 Apr–Jun orders | Offer a generic food item after a drink and vice versa (the recommender's "complementary" rule). Treat the separated-display layout as an experiment, not a proven win. |
| 7 | **No robust promotion lift was demonstrated.** The standing 10% discount hits 5.6% of lines, spread evenly across branches, hours and customer types, without a detected consistent increase in quantity or return visits. During Harbour's 15% promotion, promo-category gross profit per day fell **14.6%** relative to the other two branches, and $472 was given away. Doubling six-month discount expenditure gives a **$4.4k annualized scenario**, not proven recoverable savings. | Difference-in-differences with control branches and placebo windows ([notebook 07](notebooks/07_discounts.ipynb)) | Find out who grants the 10% discount; pause it at one branch and compare. Replace flat promotions with a measured add-on offer. |
| 8 | **Prices and costs never changed, while Lebanon's inflation was about 10%.** None of the 135 branch-product prices or unit costs moved in six months, while consumer prices rose about 9.8%. The menu got about 9% cheaper in real terms, and the recorded costs could be fixed standard costs; the company must confirm. | [Notebook 08](notebooks/08_lebanon_prices.ipynb), CPI sources in `data/external/` | Confirm the real ingredient costs: if they rose 10% from March, recorded profit is about $9k too high. Test a small price rise at one branch before changing the price list. |
| 9 | **A second batch at 16:00 only pays when it is cheap to make.** In a June replay, two-stage preparation earns **+$411** when a second batch costs $0.25 per product batch plus $0.10 per extra unit, but **−$486** at $1 per batch plus $0.25 per unit. | Simulated replay of 1,710 branch-item-days ([MODEL_CARD](MODEL_CARD.md)) | Measure the real labour cost of an afternoon batch before rolling it out. |
| 10 | **More frequent visits explain the main recorded spend difference for regulars.** The exploratory checks found little difference in per-visit spending or product mix; shared IDs across branches require confirmation. | [Notebooks 04 and 06](notebooks/README.md) | Test a visit-based card valid at all branches, not spend-based rewards. |

### Everything we tested that did not help predict or explain sales

These were tested and **not** supported. "Not supported" means the data showed no measurable, robust effect; it is not proof that the factor has no effect, especially because branch locations are unconfirmed and one-day events are hard to detect.

| Factor tested | How we tested it | Result | Where |
|---|---|---|---|
| **War period** (Mar 2–Apr 16) | Orders vs the pre-war trend, by branch; held-out two-direction test | Orders **+2.8%** vs trend (no drop); Garden −2.9%. Held-out test: not supported (it made forecasts worse) | [Notebook 02](notebooks/02_war_impact.ipynb), `reports/external_factor_test.json` |
| **Strike days** (April 8 and 11 Beirut strike days) | Day vs expected, by branch and time of day; held-out test | April 8: **−1%** for the network. Not supported | [Notebook 09](notebooks/09_event_checks.ipynb) |
| **Ceasefire** (April 17) | Day vs expected; after-ceasefire trend | **0%** on the day; sales stayed on trend | Notebooks 02 and 09 |
| **17 public holidays** | Each holiday vs a normal day with the same weekday and trend | Average **about 0%**. Harbour's Labour Day (+46%) is the strongest lead but does not survive a multiple-testing correction in notebook 09 (q = 0.13) | [Notebook 03](notebooks/03_holidays.ipynb) |
| **Eid al-Fitr** (Mar 20–21) | Same as holidays | −15% and −16%, not significant after correction | Notebooks 03 and 09 |
| **Ramadan** | Daily trend; daytime vs evening share; iftar hour | On trend; **no daytime drop and no iftar spike** | Notebook 03 |
| **Weather** (temperature, rain, wind) in Batroun, Broummana and Beirut | Correlation with demand after growth and weekday; rainy vs dry days; held-out test | Correlations between **−0.12 and +0.13**; rainy days within about 3% of dry days; 0 of 27 weather tests pass | [Notebook 05](notebooks/05_weather.ipynb) |
| **Seasons** (winter, spring, summer) | Orders vs expected; iced-drink share | Every season within **about 2%** of expected; iced drinks **about 57% of drinks in every season**, even with June about 10°C warmer | Notebook 05 |
| **Inflation** (CPI +9.8% Jan–Jun) | Prices, costs and demand trend vs CPI | Demand trend unchanged; recorded prices and costs never moved | [Notebook 08](notebooks/08_lebanon_prices.ipynb) |
| **Specific events**: AUB commencement, Feb 14 road closures, May 23 waterfront run, March 29 clock change | Day vs expected by branch and time of day, with a false-discovery correction; held-out test | None survives the correction; one-day events cannot be scored by the held-out test | Notebook 09, `data/external/external_events.json` |
| **12 flagged unusual days** | 2,896 day/branch/time combinations with a false-discovery correction | **None** of the checked days survives | Notebook 09 |
| **The 10% discount** | Quantity per line, product choice, return within 30 days (also within the same customer) | No detected change in any of them | [Notebook 07](notebooks/07_discounts.ipynb) |
| **The 15% Harbour promotion** | Units vs control branches, with placebo windows | Promoted units **+0.9%**; the gain vs the control branches is within the range of placebo windows | Notebook 07 |
| **"Gateway" first purchases** | Whether a customer's first product predicts becoming a regular, vs chance | 41 of 45 products within chance; only a weak lead | [Notebook 06](notebooks/06_regulars_products.ipynb) |
| **What regulars buy** | Regulars' product mix vs other customers, with shuffle tests | Same mix: only 1 of 45 products outside chance | Notebook 06 |
| **A personal "usual" order** | Share of orders with each customer's favorite product vs random | 24%, the same as random | Notebook 06 |
| **Specific product pairs** (for example matcha + pastry) | Pair lift vs chance | No distinctive pair beyond chance (insight 6) | `src/correlation_scan.py` |
| **Branch switching** | Order of branches visited vs a shuffled order | Matches random (62% vs 62%): no branch habit or migration | Exploratory check |
| **Other model types** | Linear regression, decision tree, random forest, XGBoost and two hybrids, same split | All **worse** than the v2 Poisson model (25.7%–28.6% vs 24.7% error) | [MODEL_CARD](MODEL_CARD.md) |
| **More detailed model features** | Branch×category×weekday terms, per-line growth, recency weighting | Lost on the training-period validation, so not used | MODEL_CARD |

Wording and numbers follow the agreed [metric definitions](reports/metric_reconciliation.md). [ANALYSIS.md](ANALYSIS.md) has the recommendations in more detail, and [reports/analysis_details.md](reports/analysis_details.md) has the supporting analysis.

---

## 2. The main model: v2 preparation model

**What it does.** For each branch, product and day it predicts how many units will sell. It turns that forecast into two numbers: how many to prepare at opening, and how many more to make at 16:00 given what is left on the shelf. A limit keeps expected fulfilment at 90% or more.

**How it works, in simple terms.**
1. A **Poisson regression** (a model for counts) learns each branch's and category's normal level, the weekday pattern (for example Tuesday 1.53×) and the steady growth trend. It uses only Stories' own transactions.
2. **v2 adds two improvements**, both chosen only on the last 21 training days:
   - a **calibration factor of 0.942**, because the original model over-forecast by 6.2%;
   - **item shares** that split each category forecast across products using all training days, smoothed.
3. A **profit-aware preparation rule** converts the forecast into opening and 16:00 quantities using each item's price and cost.

**How it was tested.** We trained on **Jan 1 – Mar 31** (90 days). Settings were chosen on the last 21 of those days. Everything was then frozen and the model forecast **Apr 1 – Jun 30** (91 days), which it never saw.

| Model, Apr–Jun test (lower error is better) | Category WAPE | Status |
|---|---:|---|
| **Poisson v2** | **24.74%** | **Main model (selected)** |
| Poisson v1 (original) | 25.21% | Tried: earlier version |
| Linear regression (log counts) | 25.75% | Tried |
| Random forest correcting the Poisson forecast | 26.30% | Tried |
| Decision tree correcting the Poisson forecast | 26.51% | Tried |
| Seasonal baseline with growth | 26.64% | Baseline |
| Linear regression (counts) | 26.65% | Tried |
| XGBoost correcting the Poisson forecast | 27.08% | Tried |
| XGBoost (Poisson objective) | 27.39% | Tried |
| Random forest | 28.04% | Tried |
| Four-weekday average | 28.22% | Baseline |
| Decision tree | 28.55% | Tried |

The tree models and XGBoost can't extend the growth trend past the last training date, so on their own they under-forecast April–June by 10–12%. That is the main reason the count model wins.

**v2 against v1 on the same test:**
- Perishable-item error: 51.57% (v1 52.03%).
- Simulated gross contribution: $11,944, which is **$1,010 more** than v1 (7-day block interval about $884–$1,126), with 528 fewer unsold units.
- Fulfilment: 90.92% of recorded demand (v1 92.09%).

The four-weekday baseline simulates more contribution ($13,410), but only by fulfilling 88.2% of recorded units, so we don't select it on profit alone. Full protocol, tuning and all results: [MODEL_CARD.md](MODEL_CARD.md), [EVALUATION.md](EVALUATION.md), `reports/strict_half_results.json`, plus the comparison files `reports/linear_baseline_check.json`, `reports/tree_models_check.json` and `reports/xgboost_check.json`.

**Example.** Harbour, Turkey & Cheese Sandwich, June 30: the forecast is 3.7 units. Prepare **4** at opening. At 16:00, if 0 are left prepare 4 more, if 1 is left prepare 3, and so on.

```bash
pip install -r requirements-lock.txt          # Python 3.11, versions used to save the models
python src/strict_model.py                    # train, select, freeze and evaluate (writes models/strict_half_model.joblib)
python src/predict_strict.py --date 2026-06-30 --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0
python src/verify_submission.py               # all 44 unit tests + artifact checks
```

**Connected decision components and supporting experiments:**
- **Add-on recommender** (`src/basket_recommender.py`): suggests one extra item for the current order. In 8,801 test orders it recovers more margin than "suggest the most popular item" ($0.51 vs $0.47 per order), but finds the hidden item slightly less often (Hit@3 13.3% vs 13.7%). Against margin-weighted popularity, the recovered-item margin proxy improves 2.51% ($0.5080 versus $0.4956). This is relevance for an already-purchased hidden item, not additional sales. See [the evaluation](reports/recommendation_evaluation.md).
- **Display-layout comparison** (`src/wejha_layout.py`): uses pooled same-order co-purchases to compare adjacent, separated and random-reference pastry arrangements, matching the proposed wejha exposure experiment. These are geometric scores, not measured sales or per-person layouts.
- **External-context models** ([EXTERNAL_CONTEXT.md](EXTERNAL_CONTEXT.md)): an order-count model with calendar features improves traffic error from 9.46% to 9.27%, but item volumes and profit did not improve.
- **Earlier rolling and Bayesian planners** ([MODEL_CARD.md](MODEL_CARD.md)): experiments with a different protocol, kept for the record.

---

## 3. Limits and the next test

- **Data limits:** sales are recorded, but waste, stockouts, preparation times and real costs are not. Branch locations are anonymised, and six months can't show annual seasonality. This dataset was explored before the final split was fixed, so the April–June test is chronological but not a brand-new sample.
- **Pilot to confirm the value:**
  1. Run the planner in **shadow mode** first: log its advice next to what staff actually do.
  2. Then randomly assign branch-days, within branch and weekday, to the planner or to current practice.
  3. Measure waste, sell-outs, batch labour and gross contribution.

  `app/planner.html` has a pilot log and `src/pilot_report.py` summarises it.
- **Add-on and display pilot:** [WEJHA_PILOT.md](WEJHA_PILOT.md) describes it. Log orders in the [order-level template](data/pilot/basket_pilot_template.csv), then run `python src/basket_pilot_report.py <export.csv> --control popular`. The report compares the model's offer with a simple popular-item offer and gives recorded contribution after item costs, waste, extra labour and other costs, with branch-day uncertainty. Test the add-on offer and the display layout separately. The [business review](reports/wejha_business_review.md) lists what must be confirmed first.

---

## Repository map

| Where | What |
|---|---|
| [`app/`](app/README.md) | Offline demo pages: `planner.html` (preparation plan, Day explorer, Why, evidence, pilot log) and `wejha.html` (add-on offers, display layouts, profit scenario) |
| `src/` | All code. Main model: `strict_model.py`, `predict_strict.py`. Checks: `verify_submission.py` |
| `models/` | Saved trained models (`strict_half_model.joblib` is the main v2 model) |
| `tests/` | Unit tests: `python -m unittest discover -s tests` |
| `data/` | `stories_transactions.csv` (raw, never edited) and [external sources](data/external/README.md) |
| [`notebooks/`](notebooks/README.md) | Exploratory analysis notebooks 01–09 |
| [`reports/`](reports/README.md) | Generated results, figures (`figures/ui/` has demo screenshots) and team documents |
| [MODEL_CARD.md](MODEL_CARD.md) · [EVALUATION.md](EVALUATION.md) | Model details, every comparison and the test protocol |
| [ANALYSIS.md](ANALYSIS.md) · [reports/analysis_details.md](reports/analysis_details.md) | Business recommendations and supporting analysis |
| [EXTERNAL_CONTEXT.md](EXTERNAL_CONTEXT.md) | Weather, events and external-context models |
| [PROJECT_REPORT.md](PROJECT_REPORT.md) | Everything we did, measured vs estimated, rejected approaches, and a checklist against the brief |
| [SUBMISSION.md](SUBMISSION.md) · [WEJHA_PILOT.md](WEJHA_PILOT.md) | Quickstart and verification · add-on/display pilot design |
| [TEAM.md](TEAM.md) · [AGENTS.md](AGENTS.md) | Team and the shared work rules |
