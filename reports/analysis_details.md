# Analysis details

Supporting analysis moved here when the README was shortened to the business insights and the main v2 model. Each section names the script or notebook that produces it.

## Exact submission and three-person work split

| Owner | Concrete responsibility | Acceptance condition |
|---|---|---|
| Codex teammate | External-context traffic/volume/mix heads, every-day evidence ledger, transaction-only control, saved artifacts and profit optimizer | Stories-only sales labels; training-only selection; sourced external features; forecast timing, costs and service visible |
| Claude Code teammate 1 (Omar Al-Aghbari) | External evidence and branch exposure; confirm event timing, coordinates and weather provenance | Every usable feature has a source, known-at time, geography and future validation; unresolved events remain alerts |
| Claude Code teammate 2 | Presentation/demo interface and pilot logging using the existing JSON/CSV contracts | Show opening plan, remaining-stock input, replenishment action, cost sensitivity, service and evidence; never present historical cards as live recommendations |

**Implemented and committed:** trained half/half model and saved artifact; forecast baselines; external-context traffic/volume/mix models; every-day dossiers; two-stage optimizer; cost/service scenarios; supplemental intraday Gamma-Poisson/NB comparisons; basket recommender; display optimizer; CLI integration; offline UI and decision data; 39 tests plus artifact/frontend checks. **Still needed for live operation:** verified operational inputs, capacity/lead-time handling and a controlled prospective profit pilot. Confirmed team identities are in [TEAM.md](../TEAM.md).

Start the company pilot with shadow decisions and measured preparation, waste, stockouts and batch labor. Then randomize eligible branch-days within branch and weekday between current practice and the planner, after safety and service rules are agreed. Choose the duration using baseline profit variance and a business-relevant minimum detectable gain; do not claim three branches automatically provide sufficient power. Evaluate incremental contribution after waste and extra labor, customer fulfillment and overrides. Freeze the policy before the pilot. The [build plan](deeper_research_and_build_plan.txt) gives the data contract and demo narrative; [AGENTS.md](../AGENTS.md) is the shared work contract for all three contributors.

## Data notes

- Raw data: **51,177 product lines, 33,887 orders, 54,087 units, 45 products and 9 categories** across 181 days.
- Raw totals: **$305,925.14 revenue and $179,430.49 gross profit**. Gross profit is not net profit.
- The original analyses remove 132 exact repeated lines. Without a line identifier, these may represent repeated items within an order or duplicate records. Removing them reduces revenue by $741.43 and gross profit by $436.21, with no change in order count. Preserve raw data and report deduplication sensitivity until the record semantics are established.
- About 36.5% of lines lack a `Customer ID`. This means unidentified orders; it does not establish that those customers are first-time visitors or walk-ins.
- The brief states that the transactions are real and identifiers have been replaced. Branch aliases do not establish geographic locations. Beirut weather is a geographic proxy unless actual branch mapping is available.
- The window does not cover a full year. It cannot establish repeatable annual seasonality or autumn behavior.

## Original descriptive findings

The approximate figures below come from the team's original deduplicated analysis. They describe associations in this file, not causal effects.

| Finding | Interpretation | Original chart |
|---|---|---|
| Garden unit costs are about 43% above Harbour; gross margins are about 47% versus 65% | A cost-definition and purchasing opportunity to investigate; the roughly $16k six-month gap is not guaranteed savings | `1_garden_cost_base.png` |
| Orders grew about 50% from January to June while order value stayed near $9 | Growth in recorded orders, without identifying its cause | `2_growth_orders_vs_aov.png` |
| Demand peaks near 21:00 and Tuesday is the busiest weekday | Useful calendar baselines; they do not explain why those patterns occur | `3_evening_core.png`, `4_weekday_hour.png` |
| Food gross margins are below drink gross margins | Relevant to product-level profit decisions, subject to preparation, waste and capacity costs | `5_margins.png` |
| Raw promo-category units/day were similar before and during Harbour promotions | An unadjusted comparison; incremental sales and profit remain uncertain | `6_promo_effect.png` |
| Many identified customers appear at multiple branches | Cross-branch observed activity; identifiers and actual acquisition dates are separate concepts | `7_customers.png` |
| Pastry and matcha appear together frequently | A candidate bundle to test against a popular-item baseline, not proof of incremental demand | `8_cross_sell.png` |

## Why the numbers move (`src/why_drivers.py`)

Each question is answered by breaking the result into its drivers and checking the split in both halves of the year (Jan-Mar, Apr-Jun). These are accounting breakdowns and held-out fits, not causal effects.

| Question | Answer (same in both halves) |
|---|---|
| Why did gross profit grow 52.5% (Jan→Jun)? | More identified customers active **37%**, more visits per identified customer **28%**, more walk-in orders **34%**. Items per order, price/mix per item and margin % together contribute **about 0%**. Jan→Mar: 36/30/35%; Apr→Jun: 34/30/32%. |
| Why is Garden's margin 47% vs Harbour's 65%? | Of the 18.1-point gap: higher unit costs **15.1 pts (84%)**, lower prices 2.5 pts (14%), product mix 0.4 pts (2%), at Garden's own quantities. Uptown's 2.4-point gap is its lower prices (−3.2 pts), partly offset by lower costs (+1.4 pts). |
| Why do daily orders go up and down? | Fitted on Jan-Mar and scored on Apr-Jun, the held-out R² is 35.9% for branch alone, 63.0% after adding growth and **92.7% after weekday**. A conditional Poisson arrival-noise benchmark is **93.4%** (residual variance 1.10× the assumed counting variance). Most broad variation is accounted for by this baseline, but the benchmark is not a hard limit proving external interactions cannot matter. Reverse-direction R² is a diagnostic, not a deployable past forecast. |
| What is different about Tuesday? | All **26 of 26** Tuesdays are above the typical day. Tuesday runs **1.43-1.65×** an average Mon/Wed-Fri in every slice: each branch, both halves, every daypart, dine-in and takeaway, identified customers and walk-ins, every category. Order value is unchanged ($9.03 vs $8.98). A uniform lift points to a chain-wide cause (calendar, standing offer, operations or recording), not a customer or product segment. The data cannot name it; the company can. Saturday is about 0.70× a normal weekday. |

Outputs: `reports/why_drivers.json`, `app/why_data.js` and `reports/figures/why_drivers.png`. Tuesday is compared with Monday and Wednesday-Friday, using per-day rates; the period has 25 Wednesdays and 26 of every other weekday. The 1.66× in ANALYSIS.md compares Tuesday with all six other days, including the quiet weekend. Both are correct for their own definition; see the [metric reconciliation](metric_reconciliation.md).

## Metric reconciliation (`src/metric_reconciliation.py`)

This is the ANALYSIS.md team action for Omar and Marwa: one definition for each number that differs between documents. The script recomputes every variant exactly as it was defined where it appears, on raw lines and with the 132 repeated lines dropped. It then proposes one definition and wording per metric.

| Metric | Proposed | Variants seen |
|---|---|---|
| Single-item orders | 53.9% contain one distinct product | 53.6% one line (nb 01), 50.6% one unit (pitch) |
| Retention | 65.6% month-to-month repeat | 91.2% Jan-Feb/May-Jun overlap (pitch; not retention), 70.7% Jan cohort active in June (nb 04) |
| Promotion profit | −14.6% promo-category GP/day vs controls (DiD); units +8.7%; $472 given away | −22.5% vs 2 weeks before, no controls (pitch); −3.9% whole-store (nb 01); units +0.9% (nb 07) |
| Tuesday | 1.53× a normal weekday (per day) | 1.66× vs all six other days (ANALYSIS.md); 1.55× ratio of totals (≈ nb 01/03 "+55%") |
| 10% discount | 5.6% of lines | 8.3% of orders (nb 01's text says "about 6%", but its table shows 0.08) |
| Order growth | +57% orders per day, Jan→Jun | +52% monthly totals |

Status: **agreed by Omar and Marwa, 2026-10-17.** The full table, with every variant and its source, is in `reports/metric_reconciliation.md` and `.json`. Notebooks are not edited (AGENTS.md).

**Supporting scan (`src/correlation_scan.py`):** all 203 order-level variable pairs are measured separately in each half, and 202 differ by less than 0.03. The stable relationships are:
- Time of day shapes the mix. Morning items are 24% hot coffee and 27% bakery; evening items are 12% hot coffee, 17% matcha/tea, 17% iced coffee and 11% desserts.
- Takeaway is about 54% of daytime orders and about 28% of evening orders.
- All three branches have the same hourly profile (correlation ≥ 0.99) and peak at 21:00.
- Two-category orders pair a drink with a food 59% of the time, against 50% under label shuffles.

Specific category pairs do *not* go together beyond chance: the strongest lift is 1.10, and pastry + matcha is 0.96-1.01. A drink + food offer is supported; a specific pairing is not. The link between a customer's visit count and number of branches used is mechanical (shuffled 2.51 vs actual 2.50 branches), and visit frequency is unrelated to spend per visit (ρ = 0.04). Outputs: `reports/correlation_scan_pairs.csv`, `reports/correlation_scan.json` and `reports/figures/correlation_scan.png`.

## Customer analysis

Notebook 04 labels customers using visits per month since their **first observed** transaction. Its Regular/Occasional/Rare/New groups are descriptive, and the thresholds are configurable.

- Regulars account for about 30% of identified customers and 41% of identified revenue in the original analysis. Higher spending is associated with more recorded visits.
- Similar discount rates across groups do not establish the absence of other loyalty benefits or prove that a loyalty scheme would be profitable.
- January's 1,979 and June's 20 first-seen IDs are **not acquisition counts**. Existing customers first observed after the dataset starts are included. The claim that acquisition stopped is withdrawn.
- The reported 91% overlap describes January-February customers observed again in May-June. It is not a general retention probability.
- The 157 customers flagged overdue are a heuristic campaign audience. Their historical roughly $12k spend is not predicted revenue at risk or recoverable profit. Time-based validation is needed; classifying regularity over the whole period can exclude formerly frequent customers who lapsed earlier.

## Original price and basket scenarios

The original pitch contains useful financial scenarios, but its headline that demand is a habit and therefore insensitive to prices is unsupported. All **135 branch-product combinations have one list price** throughout the sample. List-price elasticity cannot be separated from branch differences. Two short nonrandomized discount periods do not resolve that issue.

| Original annualized component | Approximate amount | Assumption required |
|---|---:|---|
| Garden costs at Harbour's recorded unit costs | $31.8k | Cost definitions are comparable and the lower cost is attainable |
| Garden prices aligned to Harbour | $9.9k | Volume and mix remain constant |
| Uptown prices aligned to Harbour | $10.4k | Volume and mix remain constant |
| Remove recorded discounts | $4.4k | Customers still buy at the higher effective price |
| Add a bakery/dessert item to 10% of single-item orders | $9.0k | Assumed incremental acceptance, no cannibalization, adequate capacity |

The roughly **$65k total is a scenario obtained by doubling six observed months**, not an annual forecast, achieved savings, net profit or ROI. Measured cost gaps and discount expenditure are not themselves incremental savings. Demand response, changing costs, seasonality, waste, labor, implementation costs and interactions between actions can change the outcome. The original approximate 10%/13% break-even volume losses assume stable mix and unit economics; they are scenario thresholds, not estimated demand responses.

Use a controlled, time-bounded pilot before changing the company-wide price list or discount policy. Evaluate incremental gross profit with uncertainty, customer and product substitution, and service outcomes. Compare contextual add-ons with a simple popular-item offer rather than concluding from favorite-item shares that personalization cannot work.

## Deployment data contract

Proceed with documented assumptions for the hackathon. Before live use, establish branch coordinates and timezones, transaction-line semantics, opening hours and closures, product availability and stockouts, ingredient/waste costs, promotion exposure, historical customer registration, and the information available when a decision is made. These inputs determine which external signals can be joined credibly and whether forecast improvements translate into money.

The original external input is Beirut weather from [Open-Meteo's historical weather API](https://open-meteo.com/en/docs/historical-weather-api). Consult the new analysis report and source registry for verified event sources and the distinction between observed weather, calendars and forecast-time inputs.

## Latest team weather notebook

Notebook 05 now uses Batroun weather under the team's North-region assumption and repeats comparisons with Broummana and Beirut. The new weather files and notebooks 06-07 were pulled in commit 9f17086. Obtain company confirmation of the geographic assumption; no alias-to-location mapping is established by branch names. The existing Codex model comparison still uses the documented Beirut proxy, so it does not silently claim to have rerun with Batroun. Small residual correlations do not establish that weather has no effect or that real transactions are synthetic. The assignment describes real transactions with replacement IDs.

Notebook 06 compares regular-customer product choices with 300 label shuffles. Most reported differences are small. Its first-observed-purchase and customer segmentation limits still apply; this does not establish that a future targeted offer cannot work.

Notebook 07 compares discount exposure, order composition, later returns and Harbour's two promotion windows against control branches and placebo windows. It reports little detected promotion lift. A uniform observed discount rate does not establish randomized assignment, and failure to detect lift is not proof of zero effect. Its roughly $4,400 annualized discount expenditure is a scenario, not guaranteed recoverable profit. Use these results to design a controlled offer test before eliminating discounts or claiming frequency rewards will increase profit.

## Lebanon prices vs recorded prices and costs (notebook 08)

Notebook 08 compares Lebanon's 2026 consumer price index with the café's recorded prices and costs. CPI figures come from the Central Administration of Statistics via Credit Libanais, BLOMINVEST and Trading Economics, with source links, retrieval date and derived-vs-published flags in `data/external/lebanon_cpi_2026H1.json`. All transaction rows are kept.

- **Lebanon:** CPI rose about **+9.8% from January to June 2026** (+4.91% in March and +3.04% in April, then flat). Year-on-year inflation peaked at 20.0% in April; food was +18–19% and restaurants and hotels +16% year-on-year by April. USD/LBP stayed near 89,500.
- **Recorded data:** **none of the 135 branch-product combinations changes its list price or its unit cost** in six months, so the menu became about 9% cheaper in real terms by June.
- **Implication:** the recorded unit costs are probably standard costs rather than monthly invoice costs. If real ingredient costs rose 10% from March, March–June gross profit would be about $9k lower than recorded (about $17k at +18%), with Garden the most exposed. These are scenarios, not measurements; company cost data is needed. The notebook does not estimate price elasticity or attribute demand changes to inflation.
