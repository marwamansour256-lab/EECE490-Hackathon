# EECE490 Hackathon: Stories Café Sales Analysis

Analysis of six months of transactions (January to June 2026) from Stories Café's three branches: Harbour, Garden and Uptown. The goal is to find business opportunities in the data.

## Project structure

```
EECE490-Hackathon/
├── data/
│   └── stories_transactions.csv   # raw data: one row per product line, Transaction ID = order
├── notebooks/
│   ├── 01_eda.ipynb                # full exploratory analysis with notes
│   └── 02_war_impact.ipynb         # before / during / after the 2026 Lebanon war
├── src/
│   └── eda_visuals.py              # script that draws the key-finding charts
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

The notebook compares the period before the 2026 Lebanon war (Jan 1 to Feb 28), the war (Mar 1 to Apr 15) and the period after the Apr 16 ceasefire. The data shows **no measurable war effect**: orders stayed about 3% above the pre-war growth trend, and order value, margin, product mix, time of day and takeaway share did not change. Because the data is synthetic, this doesn't show how a real café was affected. The end date is a parameter (`WAR_END`) at the top of the notebook.
