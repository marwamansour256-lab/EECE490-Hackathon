# EECE490 Hackathon: Stories Café Sales Analysis

Analysis of six months of transactions (January to June 2026) from Stories Café's three branches: Harbour, Garden and Uptown. The goal is to find business opportunities in the data.

## Project structure

```
EECE490-Hackathon/
├── data/
│   └── stories_transactions.csv   # raw data: one row per product line, Transaction ID = order
├── notebooks/
│   ├── 01_eda.ipynb                # full exploratory analysis with notes
│   ├── 02_war_impact.ipynb         # before / during / after the 2026 Lebanon war, by branch
│   ├── 03_holidays.ipynb           # public holidays and Ramadan, by branch
│   ├── 04_regular_customers.ipynb  # who comes regularly, their value, and who is at risk
│   ├── 05_weather.ipynb            # weather and seasons (winter, spring, summer), by branch
│   ├── 06_regulars_products.ipynb  # what regular customers buy vs other customers
│   └── 07_discounts.ipynb          # how discounts and the 15% promo change demand
├── src/
│   ├── eda_visuals.py              # script that draws the key-finding charts
│   └── pitch_visuals.py            # charts backing the pitch
├── reports/
│   └── figures/                    # saved charts (PNG)
├── requirements.txt
└── README.md
```

## Getting started

```bash
pip install -r requirements.txt

# Show the charts (add --save to write them to reports/figures/)
python src/eda_visuals.py
python src/eda_visuals.py --save
```

For the full walkthrough, open `notebooks/01_eda.ipynb` in VS Code or Jupyter. In VS Code you can also run `src/eda_visuals.py` one chart at a time with **Run Cell** above each `# %%` line.

## Data notes

- About 51,000 product lines across about 33,900 orders and 45 products in 9 categories.
- 132 exact duplicate lines are dropped in the analysis.
- 36.5% of lines have no `Customer ID`. These are walk-in orders and are excluded from the customer analysis only.
- Transaction IDs start with `SYN-`, so the data is synthetic.

## Key findings so far

| # | Finding | Chart |
|---|---|---|
| 1 | Garden's unit costs are about 43% above Harbour's on every product, so its margin is 47% against 65%. That's worth about $16k of gross profit over 6 months. | `1_garden_cost_base.png` |
| 2 | Orders grew about 50% from January to June, but the average order stayed flat at about $9, and 54% of orders contain a single product. | `2_growth_orders_vs_aov.png` |
| 3 | Evening is the core business: demand peaks at 21:00, and about 72% of orders after 18:00 are dine-in. | `3_evening_core.png` |
| 4 | Tuesday averages about 55% more orders than other weekdays. Weekends are the weakest days. | `4_weekday_hour.png` |
| 5 | Food margins (46-53%) are well below drink margins (61-82%). | `5_margins.png` |
| 6 | The 15% promo at Harbour showed no volume lift. Its effect on profit is unclear. | `6_promo_effect.png` |
| 7 | 94% of identified customers use 2 or more branches, with a median of 6 visits in 6 months. | `7_customers.png` |
| 8 | The most common pairing is a pastry with matcha, which makes a natural bundle. | `8_cross_sell.png` |
| 9 | Feature heatmaps: Garden branch and unit cost are the strongest drivers of lower margin. | `9_feature_heatmaps.png` |

## War impact (notebook 02)

The notebook compares the period before the 2026 Lebanon war (Jan 1 to Feb 28), the war (Mar 1 to Apr 15) and the period after the Apr 16 ceasefire. The data shows **no measurable war effect**: orders stayed about 3% above the pre-war growth trend, and order value, margin, product mix, time of day and takeaway share did not change. Because the data is synthetic, this doesn't show how a real café was affected. The end date is a parameter (`WAR_END`) at the top of the notebook. Every section is split by branch. All three branches grew through the war; Garden ran slightly below its own trend, but the gap widened after the ceasefire, so it's likely unrelated to the war.

## Holidays and Ramadan (notebook 03)

The notebook compares each of the 17 Lebanese national holidays (Jan to Jun 2026) with a normal day on the same weekday and at the same point in the growth trend, split by branch. On average, holidays are normal days (about 0%), and time of day, order value, takeaway share and product mix don't change. Ramadan shows no daytime drop and no spike after iftar. **Harbour is the exception:** 5 of its 17 holidays were unusual (where about 1 would be expected by chance), mostly busier on civic and Christian holidays, such as Labour Day (+48%) and St Maroun (+35%). The holiday dates are a list (`HOLIDAYS`) at the top of the notebook.

## Regular customers (notebook 04)

Identified customers are split by visits per month since their first visit: **Regular** (1.5 or more), **Occasional** (0.75 to 1.5), **Rare** (below 0.75) and **New** (first visit in the last 45 days). The thresholds are parameters at the top of the notebook.

- **Regulars are 30% of identified customers but 41% of identified revenue** (about $78 each vs $51 for occasional). They come about every 13 days.
- **Their value is frequency, not basket size:** same order value (about $9.00), items and product mix as everyone else.
- **They aren't rewarded:** discounted on about 9% of orders, the same as other segments.
- **They're brand-loyal, not branch-loyal:** 98-100% use two or three branches. Regulars bring in 26-27% of revenue at every branch.
- **157 regulars (15%) are overdue** (no visit for more than twice their usual gap and at least 3 weeks), worth about $12k so far. They're a natural target for a "we miss you" offer.

## Weather and seasons (notebook 05)

The notebook treats all three branches as being in the North and uses daily **Batroun** weather (`data/external/batroun_weather_2026H1.json`). It then repeats the key tests with **Jabal** weather (Broummana, about 630 m, `broummana_weather_2026H1.json`) and the earlier Beirut weather, in case some branches are in the mountains. All weather comes from the [Open-Meteo archive](https://open-meteo.com/). The period is split into winter (Jan–Feb), spring (Mar–May) and summer (Jun), with a full section for each branch (Harbour, Garden, Uptown) and a side-by-side comparison. The weather swung for real: in Batroun about 18°C → 20°C → 28°C maximum, with 33 → 37 → 0 rainy days; the Jabal was about 2.5–4°C colder by day and up to 7°C colder at night. **Sales don't react at any branch, whichever location's weather is used.** After removing growth and weekday, every season is within about 2% of expected. Temperature, rain and wind correlate with leftover demand between −0.12 and +0.13. Rainy days are within about 3% of dry days. The iced share of drinks stays at about 57% in every season, even though June was about 10°C warmer than January, which is the clearest sign the data was generated without weather. Seasons and the rainy-day threshold are parameters at the top of the notebook.

## What regular customers buy (notebook 06)

Uses the same segments as notebook 04 and checks each result against a shuffle test (customers randomly relabeled 300 times) so chance differences aren't mistaken for real ones.

- **Regulars buy what everyone buys:** category mix within about 1 point of other customers, and only 1 of 45 products outside the chance range. A slight pastry lean (bakery +1.3 points) is too small to act on.
- **No product depends on regulars:** every product gets 23–29% of its revenue from them.
- **No reliable gateway product:** the first purchase barely predicts becoming a regular (41 of 45 products within chance; cold drinks and Chocolate Croissant slightly higher, a weak lead).
- **No "usual":** a regular's favorite product is in 24% of their orders, the same as random; they try about 11 products in 9 visits.
- **They don't trade up on their own:** order value and mix are flat from the 1st to the 7th+ visit. Their favorite pairing is matcha or iced latte + pastry, the same at every branch.

## Discounts and demand (notebook 07)

Two discounts exist: a standing **10% off** on about 6% of lines everywhere, and a **15% promo** at Harbour on Cold Coffee and Desserts (Mar 12–19, May 12–19). Prices never change otherwise, so these are the only price changes to learn from.

- **The 10% discount is random:** about 5.6% of lines in every branch, category, hour, day, month and customer type; its spread across customers matches a random hand-out.
- **It doesn't make orders bigger:** discounted orders look bigger ($9.60 vs $8.94) only because bigger orders have more lines that could be discounted. Per line, the rate is the same in 1-, 2- and 3-line orders, and quantities don't rise.
- **It doesn't change product choice or bring customers back:** same mix and item price; 65% return within 30 days either way, even within the same customer.
- **The 15% promo had no measurable lift:** promo units at Harbour +1%; against the control branches, the March and May windows beat only 76% and 58% of 101 fake 8-day windows (about 95% would be needed).
- **Cost:** about $4,400 a year ($3,400 from the 10% discount). Recommendation: stop untargeted discounts, redirect the budget to frequency rewards and bundles, and test any future offer at one branch against the others.

## The pitch: demand is a habit, so fix the price list

```bash
python src/pitch_visuals.py --save   # writes reports/figures/pitch_*.png
```

**Conclusion.** Stories' demand behaves like a habit, not a reaction. Weekday plus a steady trend explain 93% of daily orders, and growth comes from the same ~3,500 loyal customers visiting more often (91% retained, almost no new ones by June). That makes the price list the lever, not promotions. Garden pays ~43% more per product than Harbour yet charges ~5% less, Uptown charges ~9% less, and a 15% promo at Harbour bought no extra volume. Fixing Garden's supply costs, aligning Garden and Uptown prices to Harbour, stopping untargeted discounts and adding a till add-on is worth **about $65k a year of gross profit**, mostly from the price list. A 4-week price test at Garden should come first (details below).

| # | Claim | Evidence | Chart |
|---|---|---|---|
| 1 | Demand follows the calendar, not the news | Weekday + trend: R² = 93%. Beirut weather vs leftover demand: corr. −0.07 (temp), +0.02 (rain). War start (2 Mar) −1% vs expected; deadliest day (8 Apr, ~357 killed) −1%; only Eid al-Fitr dips (−16%). | `pitch_1_demand_is_habit.png` |
| 1b | Mix does not react either | Beirut +11°C Jan→Jun, iced-drink share flat at ~31%. Takeaway share 41% before, during and after the war. Ramadan daytime share 36.0% vs 35.8%. | `pitch_1b_mix_does_not_react.png` |
| 2 | Growth = same customers, more visits | New identified customers 1,979 (Jan) → 20 (Jun). Visits per active customer 1.45 → 1.75. | `pitch_2_loyal_base.png` |
| 3 | The price list is backwards | Garden: cost ×1.43, price ×0.95 vs Harbour on all 45 products (margin 47% vs 65%). Uptown: price ×0.91, cost ×0.96. | `pitch_3_price_list.png` |
| 4 | Price cuts buy no volume | Harbour 15% off Cold Coffee + Desserts (12–19 Mar, 12–19 May): units/day 33.9 → 33.8, gross profit/day $130 → $100. Control branches moved ±10%. | `pitch_4_promo_no_lift.png` |
| 5 | Basket upside, but not personalised | 51% of orders are one item. Customers' top-product share (15%) matches random ordering (17%), so there is no personal "usual" to target. | `pitch_5_basket_upside.png` |
| 6 | Value | Garden costs $31.8k, Garden prices $9.9k, Uptown prices $10.4k, discounts $4.4k, add-on $9.0k per year. | `pitch_6_value.png` |

**Measured vs estimated.** The Garden cost gap and discount spend are measured from the data. The price gains assume no volume loss. They still pay off unless volume falls more than 10% (Garden) or 13% (Uptown). The add-on assumes 1 in 10 single-item orders accepts a pastry ($2.60 gross profit each).

**Data quality finding: the sales do not react to real 2026 events.** A war started in Lebanon on 2 March 2026 with repeated strikes on Beirut, yet orders rose and nothing in the mix changed. Together with the `SYN-` IDs and flat per-branch multipliers, this means the dataset follows a generated pattern. So we only rely on its internal structure (prices, costs, promo response, customer counts) and do not claim anything about crisis or weather behaviour.

**Assumptions that could change the recommendation.**
- Uptown's lower prices may reflect a lower-income area. If so, aligning its prices is riskier than aligning Garden's.
- The promo covered two categories for 8 days and we do not know if it was advertised. "No lift" may partly mean "nobody knew".
- Prices never changed within a branch, so price sensitivity cannot be estimated directly.
- Some of Garden's extra cost may be justified (rent, smaller deliveries).

**Test before rollout.** Raise Garden prices 5% for 4 weeks, with Harbour and Uptown as controls. Track orders/day and gross profit/day. Stop if Garden orders fall more than ~8% vs the controls, which is safely inside the 10% break-even. In parallel, ask Garden's suppliers for Harbour's price list.

**External data.** Beirut daily weather from the [Open-Meteo archive](https://open-meteo.com/) (`data/external/beirut_weather_2026H1.json`). Event dates from Wikipedia's [2026 in Lebanon](https://en.wikipedia.org/wiki/2026_in_Lebanon) and [Timeline of the 2026 Lebanon war](https://en.wikipedia.org/wiki/Timeline_of_the_2026_Lebanon_war).
