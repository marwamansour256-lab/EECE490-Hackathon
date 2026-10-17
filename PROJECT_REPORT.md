# Project report: what we did, what we found, and how it meets the brief

**Team:** Charbel Manih (202502099), Omar Al Aghbari (202514349), Marwa Mansour (202505788) · [TEAM.md](TEAM.md)
**Repository:** https://github.com/marwamansour256-lab/EECE490-Hackathon (public)

## Conclusion (one paragraph)

Stories' recorded January-to-June gross-profit growth of 52.5% came almost entirely from more orders, while 54% of orders contained one product. We built tools to test two repeatable decisions: **how much fresh food to prepare** and **which add-on to offer**. Fitted on Jan–Mar, v2 achieved 24.74% category WAPE on Apr–Jun versus 26.64% for the seasonal-growth baseline. Under a common one-batch policy it simulated $11,944 contribution versus $9,915, with recorded-demand fulfilment of 90.92% versus 92.19%; these are assumptions-based results, not company savings. Separate opening/16:00 recommendations require cheap, feasible replenishment. Garden's recorded unit costs explain 84% of its margin gap with Harbour; matching Harbour's costs gives an approximately $16k six-month scenario requiring invoice verification. No factor passed the strict 39-test external-event screen; a separate monthly-calendar traffic model improved WAPE slightly, without establishing preparation-profit gains. The dataset was explored before this frozen evaluation, so a randomised prospective pilot must confirm value.

---

## 1. The business problem and why we chose it

Stories' data team already knows how many lattes each branch sells, so a sales summary would add nothing. We looked for decisions the data can improve:

- **Daily preparation of fresh items** (bakery, desserts, sandwiches, breakfast). Too much becomes waste; too little loses sales. The decision is made every day, at every branch, for every item, so a small improvement repeats often.
- **Basket growth.** Growth did not come from bigger baskets (insight 2), so the add-on at the till is untapped.
- **Cost and pricing issues** the data exposed: Garden's costs, discounts, and prices that never changed during inflation.

## 2. What we did, step by step

| Step | What | Where |
|---|---|---|
| 1. Data quality | Checked 51,177 lines and 33,887 orders. Revenue and profit match price × quantity × (1 − discount) and cost on every line. No order spans two dates or branches. We found 132 exact repeated lines (kept in the raw data, with a sensitivity check), 36.5% of lines without a Customer ID (treated as "unidentified", not "new"), and no price or cost change for any of the 135 branch-product pairs. | [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), [analysis details: data notes](reports/analysis_details.md#data-notes) |
| 2. Exploration | Growth, weekdays and hours, margins, customers, discounts, war, holidays, weather, Lebanon CPI. | [notebooks 01–09](notebooks/README.md) |
| 3. Challenge the first explanations | Re-tested every early claim on two separate halves of the year, against shuffled or placebo versions, and with multiple-testing corrections. Several early claims were wrong and were withdrawn (section 6). | [why_drivers](src/why_drivers.py), [correlation_scan](src/correlation_scan.py), [external_factor_test](src/external_factor_test.py) |
| 4. One definition per number | Six figures differed between documents (for example single-item orders, retention, the promotion effect and the Tuesday ratio). We recomputed every variant and agreed one definition for each. | [reports/metric_reconciliation.md](reports/metric_reconciliation.md) |
| 5. Model | Trained on Jan–Mar, chose settings on the last 21 training days, froze everything, then tested on Apr–Jun. Compared 12 approaches. v2 selected. | [MODEL_CARD.md](MODEL_CARD.md), [EVALUATION.md](EVALUATION.md) |
| 6. Decision | Turned forecasts into opening and 16:00 quantities, and tested whether a second batch pays at different labour costs. | `src/profit_policy.py`, [MODEL_CARD.md](MODEL_CARD.md) |
| 7. Add-on offer and display test | Recommender evaluated on 8,801 April–June orders against popularity baselines. Display layouts scored as an experiment, not as a result. | [reports/recommendation_evaluation.md](reports/recommendation_evaluation.md), [WEJHA_PILOT.md](WEJHA_PILOT.md) |
| 8. Usable tools | Offline pages for staff and managers, a pilot log and pilot reports. | [app/README.md](app/README.md) |
| 9. Verification | Recorded 44-test numerical suite and artifact reconciliation; final saved-evidence checks and all 144 saved frontend/Python parity cases pass. Verification scope and reproduction commands are documented. | [SUBMISSION_AUDIT.md](SUBMISSION_AUDIT.md), `python src/verify_submission.py`, `node app/check_ui.js --cli` |

## 3. Findings

The ten insights, each with its evidence and recommended action, are in [README.md](README.md#1-business-insights). These go beyond a basic summary:

- **Garden's margin gap broken into parts:** 84% unit cost, 14% price, 2% mix, identical in both halves. It is not just "Garden earns less".
- **Tuesday:** 1.53× a normal weekday in every slice (branch, hour, category, customer type, dine-in or takeaway) with the same order value. A lift this uniform points to a chain-wide cause, such as a standing offer, operations or how sales are recorded, rather than a type of customer. We added the question to the [information request for Stories](reports/company_information_request.md).
- **How much is left to explain:** on held-out months, calendar and growth explain 92.7% of daily order swings, against a 93.4% benchmark under random arrivals. This contextualises the small tested aggregate gains; it does not establish why external effects are small or rule out local effects. The benchmark assumes a particular arrival model and is not a hard limit.
- **The obvious pairing is not an affinity:** "pastry + matcha" is common because both are popular (lift about 1.0). All 28 pastry display pairs have lift below 1. What *is* real is drink + food (59% vs 50% by chance). This changes the offer we recommend: a generic food-after-drink offer, not a matcha bundle.
- **Costs never moved while inflation was about 10%**, so recorded profit after March may be overstated by about $9k (scenario).
- **The 16:00 second batch:** in the earlier June seasonal-baseline replay, the low-cost scenario leaves $4.56 per branch-day for additional overhead after charging $0.25 per SKU batch and $0.10 per top-up unit. This is conditional on that scenario, not a universal labour-cost threshold.

## 4. The model, the baselines and the held-out test

- **Split:** train on Jan 1–Mar 31 (90 days); test on Apr 1–Jun 30 (91 days). Every test forecast is made with the model frozen on March 31. Settings were chosen only on the last 21 training days. Complete days and orders stay together.
- **Simple baselines:** "average of the last four same weekdays", and a seasonal baseline with growth.
- **Results:**

| Approach | Category WAPE (Apr–Jun) |
|---|---:|
| **Poisson v2 (main)** | **24.74%** |
| Poisson v1 | 25.21% |
| Linear regression (log counts) | 25.75% |
| Random forest on the Poisson forecast | 26.30% |
| Decision tree on the Poisson forecast | 26.51% |
| Seasonal baseline with growth | 26.64% |
| Linear regression (counts) | 26.65% |
| XGBoost on the Poisson forecast | 27.08% |
| XGBoost | 27.39% |
| Random forest | 28.04% |
| Four-weekday baseline | 28.22% |
| Decision tree | 28.55% |

- **Why the tree models lose:** they cannot extend a growth trend beyond the training dates, so on their own they under-forecast April–June by 10–12%.
- **Honest caveat:** the dataset was explored before the final split was fixed. The test is chronological and frozen, but it is not a brand-new sample.

## 5. Business value: measured results kept separate from estimates

| Item | Value | Type |
|---|---|---|
| Forecast error, v2 vs seasonal baseline | 24.74% vs 26.64% | **Measured** on held-out months |
| Recorded discount spend | about $2.2k in six months (about $4.4k a year) | **Measured** spend; savings from removing discounts are a scenario |
| Garden at Harbour's recorded unit costs | about +$16k gross profit per six months | **Scenario**: assumes costs are defined the same way and are attainable |
| v2 vs v1 preparation, Apr–Jun | +$1,010 simulated contribution (interval about $884–$1,126), 528 fewer unsold units, 90.9% vs 92.1% fulfilment | **Simulation**: assumes recorded sales equal demand and stated costs |
| v2 vs seasonal baseline, Apr–Jun | +$2,029 simulated contribution ($11,944 vs $9,915) | **Simulation** |
| Second batch at 16:00, June replay | +$411 at low batch cost; −$486 at $1 per batch + $0.25 per unit | **Simulation**: the result changes sign with labour cost |
| Real costs up 10% from March | about −$9k gross profit for Mar–Jun | **Scenario** |
| Add-on recommender vs margin-weighted popularity | +$0.012 recovered margin per order (interval $0.003–$0.025) | **Offline proxy**: not extra sales |
| Add-on acceptance and display uplift | unknown, set to 0 | **Must be measured** in the pilot |

**Tradeoffs:**
- The four-weekday baseline simulates more contribution ($13,410) but serves only 88.2% of recorded demand. We chose v2 for accuracy and service, not profit alone.
- v2 serves 150 fewer units than v1 in exchange for less waste.

## 6. Alternative explanations tested, and approaches we rejected

| First explanation or approach | What we tested | Decision |
|---|---|---|
| "The data is synthetic" (because IDs start with `SYN-`) | The brief says the data is real with replacement IDs | **Withdrawn** |
| "Demand is a habit, so customers don't react to price" | Prices never changed, so price sensitivity can't be estimated; one short promotion isn't a price test | **Withdrawn**; recommend a controlled price test instead |
| "Loyal customers explain the steady demand" | Unidentified orders grew just as fast and ignored the war just as much | **Rejected** |
| "Customers have a usual order" | Top-product share equals that of random shuffles | **Rejected**; generic add-on, no "your usual" |
| "Matcha + pastry is the bundle" | Pair lift about 1.0 in both halves | **Rejected**; drink + food offer instead |
| "91% retention" | It is the overlap between Jan–Feb and May–Jun customers | **Replaced** by 65.6% month-to-month repeat |
| Weather, holidays or the war drive sales | 39 held-out tests with placebo dates, Holm correction and planted-effect controls: 0 pass | No supported automatic event adjustment from this screen. External features remain allowed in the separate context models; the selected monthly-calendar traffic head improves WAPE from 9.46% to 9.27%, without established preparation-profit gains. |
| Promotion lift | Difference-in-differences against the other branches, plus placebo windows | No measurable volume lift; gross profit −14.6% |
| More complex forecast models (XGBoost, random forest, trees, linear) | Same frozen split | **Rejected**: all less accurate than v2 |
| Picking the most profitable baseline | Four-weekday baseline | **Rejected**: lower service |

## 7. Assumptions and how they could change the recommendation

| Assumption | If wrong |
|---|---|
| Recorded unit costs are comparable between branches | The Garden recommendation could shrink: part of the gap may be rent or delivery costs booked as product cost |
| Recorded costs are what Stories actually paid | If costs rose with inflation, profits are overstated and a price test becomes more urgent |
| Recorded sales equal demand (no hidden sell-outs) | If items sold out, true demand is higher, and the planner should prepare more |
| A second batch can be made at 16:00 at low cost | The tested $1 per SKU batch plus $0.25 per top-up unit scenario loses about $486 versus one batch in June; actual lead times, capacity and both cost components determine whether replenishment pays |
| Branch locations are unknown | Weather and event exposure can't be assigned to branches; event effects remain unverified |
| The 132 repeated lines are real items | If they are double records, revenue falls by $741 and gross profit by $436; conclusions don't change |

## 8. Weak areas, uncertainty and missing information

- **Not observed in the data:** waste, stockouts, preparation times, kitchen capacity, actual costs, opening hours, branch locations, the current display layout, and how the 10% discount is granted.
- **Coverage:** six months only, so there is no annual seasonality.
- **Simulations, not company measurements:** profit gains come from replays under stated costs; they are not measured savings.
- **Recommender metrics** measure recovering items customers already bought, not extra sales.
- **Questions for Stories** are listed in [reports/company_information_request.md](reports/company_information_request.md).

## 9. The practical test Stories could run before wider use

1. **Preparation planner:**
   - Run it in **shadow mode** for two weeks: log its advice next to what staff actually do.
   - Then randomly assign branch-days, within branch and weekday, to the planner or to current practice.
   - Measure waste, sell-out time, batch labour and gross contribution with `app/planner.html` (pilot log) and `src/pilot_report.py`.
   - Stop the pilot if sell-outs rise.
2. **Add-on offer:** compare the model's offer against a simple popular-item offer on randomised branch-days. Log every eligible order, including declined offers ([WEJHA_PILOT.md](WEJHA_PILOT.md), `src/basket_pilot_report.py`).
3. **Garden costs:** compare Garden's and Harbour's supplier invoices for the same products.
4. **Discount and price:** pause the 10% discount at one branch for 3–4 weeks, and test a small price rise at one branch, with the other branches as controls.

## 10. Checklist against the brief

| Brief requirement | Where it is met |
|---|---|
| A business problem that matters, beyond basic summaries | Sections 1 and 3; [README insights](README.md#1-business-insights) |
| Challenge the first explanation; consider other causes | Section 6 |
| Support conclusions with evidence; check data quality | Sections 2 (step 1) and 3; [analysis details](reports/analysis_details.md) |
| Test alternative explanations | Section 6; [EXTERNAL_CONTEXT.md](EXTERNAL_CONTEXT.md) |
| Estimate business value, costs and tradeoffs | Section 5 |
| Assumptions that could change the recommendation | Section 7 |
| Compare the model with a simple baseline | Section 4 (seasonal and four-weekday baselines) |
| Test on data the model did not use for training | Section 4 (frozen Jan–Mar → Apr–Jun) |
| Weak areas, uncertainty, possible failures, missing information | Section 8 |
| A practical test before wider use | Section 9 |
| Correlation is not presented as cause | Every insight is labelled association or accounting; event evidence is kept as "candidate" ([EXTERNAL_CONTEXT.md](EXTERNAL_CONTEXT.md)) |
| Measured results kept separate from estimated benefits | Section 5, "Type" column |
| Important choices and rejected approaches explained | Sections 4 and 6 |
| Assumptions stated and kept separate from observed facts | Section 7 vs sections 2–3 |
| Public GitHub repository with team names | Top of this file, [README.md](README.md), [TEAM.md](TEAM.md) |
| One-paragraph conclusion with clear business value | Top of this file and of the README |
| Evidence easy to inspect | Links in every table; generated results in [reports/](reports/README.md); `python src/verify_submission.py` |

## How to inspect or reproduce

```bash
pip install -r requirements-lock.txt     # Python 3.11
python src/verify_submission.py          # 44 tests + artifact checks
python src/predict_strict.py --date 2026-06-30 --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0
node app/check_ui.js --cli               # demo pages match the Python model
```

To see the results without installing anything, open `app/planner.html` and `app/wejha.html` in a browser. Screenshots are in [reports/figures/ui/](reports/figures/ui/).
