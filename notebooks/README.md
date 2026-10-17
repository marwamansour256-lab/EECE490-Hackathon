# Notebooks

Each notebook runs on its own from this folder (`../data/...` paths) and is saved with its outputs, so it can be read on GitHub without running it. Requirements: `pandas`, `numpy`, `matplotlib` (see `../requirements.txt`).

Per [AGENTS.md](../AGENTS.md), notebooks 01–07 are kept as the team's original work. Where a claim in them has been superseded (for example that the data is synthetic, causal readings of the war period, or price-sensitivity statements), the correction lives in the README and the methodology audit, not in the notebook.

| Notebook | Question | Branch split | Status and caveats |
|---|---|---|---|
| [01_eda](01_eda.ipynb) | Overall patterns: growth, weekday/hour, margins, products, customers, discounts, feature heatmaps | Partly | Original. Drops 132 repeated lines; describes the data as synthetic (superseded) |
| [02_war_impact](02_war_impact.ipynb) | Sales before, during and after the 2026 war period | Yes | Original. Associations only; branch exposure unknown |
| [03_holidays](03_holidays.ipynb) | Public holidays and Ramadan vs expected sales | Yes | Original. Religious/holiday labels are not customer attributes |
| [04_regular_customers](04_regular_customers.ipynb) | Customer segments, value, retention, overdue regulars | Yes | Original. "First visit" is the first observed purchase, not the real acquisition date |
| [05_weather](05_weather.ipynb) | Weather and seasons vs sales, one section per branch | Yes | Uses Batroun (North) weather, with Broummana (Jabal) and Beirut checks; locations are assumptions |
| [06_regulars_products](06_regulars_products.ipynb) | What regular customers buy, with shuffle tests | Partly | Same segment caveats as 04 |
| [07_discounts](07_discounts.ipynb) | Discounts, the 15% promotion, placebo windows | Partly | No detected lift is not proof of zero effect; discount assignment is unknown |
| [08_lebanon_prices](08_lebanon_prices.ipynb) | Lebanon CPI vs the café's recorded prices and costs | Yes | Keeps all rows; no elasticity or causal claims; cost-increase figures are scenarios |

The business recommendations drawn from these notebooks are in [ANALYSIS.md](../ANALYSIS.md). External data sources are documented in [data/external/README.md](../data/external/README.md).
