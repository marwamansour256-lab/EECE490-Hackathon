# Analysis and recommendations

*Prepared by Omar Al-Aghbari (Claude Code teammate 1), 2026-10-17. Based on notebooks 01–09, Marwa's analysis scripts and the README. Figures use the definitions agreed in [reports/metric_reconciliation.md](reports/metric_reconciliation.md). All dollar amounts are scenarios computed from the recorded data, not measured savings or net profit; see the [methodology audit](reports/methodology_audit.txt) and [AGENTS.md](AGENTS.md) for the rules these follow.*

## For Stories (the business recommendations)

### 1. Confirm the real ingredient costs first

No product's recorded unit cost changes from January to June, even though Lebanon's CPI rose about 9.8% (+4.91% in March, +3.04% in April) and food prices were 18–19% higher than a year earlier ([notebook 08](notebooks/08_lebanon_prices.ipynb)). The cost column is probably a fixed standard cost, not what was paid each month. Every profit figure in the project depends on it. If real costs rose 10% from March, recorded March–June gross profit is about $9k too high, and about $17k too high at +18%.

**Ask the company:** are unit costs standard recipe costs or actual monthly costs, and were they updated after March?

### 2. Investigate Garden's cost gap

Garden's recorded unit cost is about 43% above Harbour's on every one of the 45 products, while its prices are about 5% lower (gross margin 47% vs 65%) ([notebook 01](notebooks/01_eda.ipynb)). At Harbour's recorded costs, Garden would have earned about $16k more gross profit over six months. This assumes the cost definitions are comparable and Harbour's costs are achievable at Garden. Some of the gap may be justified, for example by rent, smaller deliveries or waste.

**Next step:** compare Garden's supplier invoices with Harbour's.

### 3. Test a small price increase before changing the price list

List prices never changed in six months while general prices rose about 10%, so the menu became about 9% cheaper in real terms. Keeping the same gross profit in dollars after a 10% cost rise would need about **+4% on prices** (Harbour +3.5%, Garden +5.3%, Uptown +3.7%), still below inflation. The data contains no list-price changes, so it can't say how customers would react.

**Test:** +5% at Garden for 4 weeks, with Harbour and Uptown as the comparison. Track orders per day and gross profit per day; stop if Garden's orders fall more than about 8% relative to the other branches.

### 4. Test removing the untargeted 10% discount

The 10% discount appears on 5.6% of product lines (8.3% of orders contain one) everywhere: every branch, product, hour and customer type. We detected no difference in quantity, product choice or return visits for discounted purchases ([notebook 07](notebooks/07_discounts.ipynb)). For the Harbour 15% promotion, compared with Garden and Uptown over the same days, promoted units were up about 9% (within the range of placebo windows) while the promoted items' gross profit was about **15% lower**, and the promotion gave away about $470 ([agreed figures](reports/metric_reconciliation.md)). That's 16 days, two categories and one branch, so it isn't a measure of price sensitivity. Discounts cost about $4,400 a year at current volumes. Not detecting an effect isn't proof there is none, and we don't know how the discount is assigned.

**Next step:** find out who grants the 10% discount and why, then pause it at one branch for a few weeks and compare.

### 5. Reward regulars for visiting, not for spending

Regulars (1.5 or more visits a month) are 30% of identified customers and 41% of identified revenue. On average, 66% of a month's identified customers order again the next month. They buy the same products and spend the same per visit as everyone else, use two or three branches, and get no more discounts than other customers ([notebooks 04](notebooks/04_regular_customers.ipynb) and [06](notebooks/06_regulars_products.ipynb)). 157 regulars are overdue (no visit for more than twice their usual gap).

**Candidate offers to test:** a visit-based card valid at all branches (for example a free pastry after 8 visits) and an automatic win-back offer for overdue regulars. Whether these increase profit needs a controlled test.

### 6. Grow the order at the till

Average order value stayed at about $9.00 from January to June, and **53.9% of orders contain only one distinct product**. Drinks and food do go together: of two-line orders, 59% are one drink + one food, against 50% if items were combined at random, in both halves of the year. But **no specific pair stands out**: Bakery + Matcha & Tea, the most frequent pair, appears about as often as chance predicts (lift 0.96–1.01). Matcha + pastry is common because both are popular, not because they're especially linked ([`src/correlation_scan.py`](src/correlation_scan.py), [notebook 06](notebooks/06_regulars_products.ipynb)).

**Test:** a generic "add any food to your drink" (or "any drink to your food") prompt at the till, compared with a simple popular-item offer. A matcha + pastry bundle can be one arm of the test, but the data doesn't support it as the best pairing.

### 7. Plan preparation and staff by the calendar

Daily demand follows a stable pattern: orders per day grew **57%** from January to June, **Tuesday is about 1.53 times a normal weekday** (Monday and Wednesday–Friday; every one of the 26 Tuesdays is above a typical day) and the busiest hour is 21:00. This is what the team's preparation planner uses (see the [model card](MODEL_CARD.md)).

The evidence for this is now tested on data the model hadn't seen: fitted on January–March and scored on April–June, **branch + growth trend + weekday explain 92.7% of daily orders**, against a ceiling of 93.4% set by customers arriving at random. That leaves under 1% for anything else, and **0 of 39 held-out tests of outside factors** (weather, holidays, Ramadan, the war period, strike days and five events from the event list) pass ([`reports/why_drivers.json`](reports/why_drivers.json), [`reports/external_factor_test.json`](reports/external_factor_test.json)). This doesn't prove outside events have no effect: branch locations aren't confirmed, so exposure is unknown, and small or one-day effects can't be detected with this data.

## For the team (next actions)

| Owner | Action | Why |
|---|---|---|
| Omar (teammate 1) | Send the company or organizers a question list: branch locations and opening hours, closures, whether costs are standard or actual, stockout records, promotion dates and how the 10% discount is granted | Branch locations decide which events and weather apply; costs decide every profit figure |
| Omar, with Charbel | Merge Charbel's event sources into the event list (he published them in [reports/source_reconciliation.md](reports/source_reconciliation.md)); his `outlier_event_registry.json` is still not on GitHub | Avoid two registries with different sources |
| Omar | ~~Fill in the event registry with sources, known-at dates and branch exposure~~ **Done:** [data/external/external_events.json](data/external/external_events.json); exposure stays "unknown" until locations are confirmed | Required by AGENTS.md before any event can affect a forecast |
| Omar, with Marwa | ~~Agree one definition for each number that differs between documents~~ **Done:** agreed 2026-10-17 in [reports/metric_reconciliation.md](reports/metric_reconciliation.md) | So the presentation uses consistent figures |
| Everyone | Present every dollar amount as a scenario paired with a test | Matches the team rules and holds up to questions |
