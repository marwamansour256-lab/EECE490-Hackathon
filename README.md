# Stories Profit Planner | EECE490 Hackathon

Analysis of January-June 2026 transactions from three anonymized branches: Harbour, Garden and Uptown. The goal is to identify decisions that improve gross profit, validate external demand signals, and distinguish measured results from business assumptions.

**Team synchronization, October 17:** the documentation is being published first so all three contributors can work concurrently. The analysis/model artifacts described below have been implemented and checked in the Codex working copy; files not yet present on GitHub are pending a separate code publication. The next Codex extension is an **intraday Bayesian demand update with stockout-censoring handling**, followed by a comparison against the existing planner. It is work in progress, with no claimed improvement yet. The two Claude Code contributors can begin from the interfaces and acceptance criteria below and in AGENTS.md.

## What we are building

**A branch-and-item preparation planner that chooses an opening batch and a second batch at 16:00, prices the extra work, and reports its service and profit tradeoffs.** External-event research supplies evidence-linked review alerts. It changes numerical forecasts only after exposure and predictive value are established.

The implemented decision model is a **two-stage stochastic inventory dynamic program**, using Poisson demand and an expected unit-fill constraint. Its inputs are demand, branch-item selling price and ingredient cost, intraday mix, replenishment cost and salvage. Its output is an opening quantity plus a lookup table: **given remaining stock at 16:00, prepare this many more units**. This is an executable decision, not just a sales correlation.

The forecasting candidates are a regularized pooled Poisson regression and a strong seasonal baseline: the preceding four matching weekdays at branch/category level, allocated to items using prior 28-day shares. The complex forecast wins the original category accuracy comparison but **loses the new service-constrained profit comparison**. The demo defaults to the transparent baseline; this choice is exploratory and needs a future prospective evaluation. We do not claim complexity itself creates value.

### Run the working demo

```bash
pip install -r requirements.txt
python src/recommend.py --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0
python src/recommend.py --mean 5 --early-share 0.4 --price 6 --cost 2 --setup 1 --premium 0.25
python -m unittest discover -s tests -v
```

The first command reviews a **June 30, 2026 historical decision**, not today's forecast. The second lets the company change business inputs. Output includes opening units, the second-batch table, expected contribution and assumptions. It never uses afternoon sales to choose the afternoon batch. Immediate replenishment and unconstrained capacity are simulation assumptions; production use requires actual lead times and capacity.

### Where the money comes from

June retrospective replay: 1,710 branch-item-days, 57 perishable branch-item combinations, 4,657 recorded units. Both policies use the same forecast, prices and ingredient costs. The 90% numerical expected-fill floor applies to the whole day for one batch, and separately to each stage for two batches; report realized service as well as contribution.

| Scenario, strong seasonal forecast | One-batch contribution | Two-stage contribution | Difference |
|---|---:|---:|---:|
| No incremental replenishment cost | $4,930.92 | $5,740.15 | +$809.24 |
| $0.25 per SKU second batch + $0.10 per extra unit | $4,930.92 | $5,341.65 | **+$410.73** |
| $1 per SKU second batch + $0.25 per extra unit | $4,930.92 | $4,444.43 | **-$486.49** |
| $3 per SKU second batch + $0.50 per extra unit | $4,930.92 | $2,869.24 | -$2,061.68 |

In the low-cost scenario, simulated fulfillment rises from **90.68% to 93.43%**, unsold units fall from **2,859 to 2,763**, and 920 SKU replenishments occur: **10.2 per branch-day**. The remaining gain can cover only **$4.56 additional overhead per branch-day**. A 7-day block resampling interval is approximately $178-$619, conditional on this replay and these assumptions; it does not measure uncertainty in missing costs or demand. The regression forecast earns **$149.17 less** than the seasonal forecast with the same two-stage policy. Separate operational flexibility from AI forecast value.

These are **simulated contribution differences, not measured savings or net profit**. Sales may be censored by unknown stockouts; actual preparation, waste and current company policy are absent. June had already been examined when this policy was developed, so it is **not an untouched confirmation set**. A service floor of zero produces larger contribution but poor fulfillment; it is a stress scenario, not our recommended operating objective. See [complete results and assumptions](reports/two_stage_results.json), [category breakdown](reports/two_stage_by_category.csv) and [all replay decisions](reports/two_stage_replay.csv).

![Cost boundary for the preparation intervention](reports/figures/two_stage_cost_boundary.png)

### What deeper external research found

We screened 2,224 rolling day/branch/daypart comparisons, investigated 18 date dossiers and recorded verified events separately from proposed mechanisms. Most eye-catching days do not survive the approximate multiple-testing screen. Only **Harbour on May 1** is flagged: 149 orders versus about 100 expected, approximately +49%, adjusted q = 0.023. A single Labour Day cannot establish a repeatable causal uplift.

| Date | Forecast-adjusted sales fingerprint | External evidence and decision |
|---|---|---|
| Feb 14 | Network orders -18.5%; Harbour -28.9%, Uptown +1.5% | ISF announced downtown road closures. Check branch access before interpreting this as Valentine's demand. Locations are anonymized, so exposure is unresolved. |
| Mar 28 | Network +24.1%; morning -13.8%, midday +40.2% | Conflict reporting does not explain a positive sales day by itself. The clock change starts the following day and cannot explain this daytime spike. |
| May 23 | Network -7.9%; evening -16.5% | A verified morning waterfront race did not yield an automatic network sales increase. Event presence alone is a poor boost rule. |
| June 5 | Network +11.3%; evening +14.5% | AUB's 19:30 graduation is temporally compatible; Haigazian also held graduation. Catchment exposure and incremental demand remain unproven. |
| June 24 | Network -15.4%; morning -27.5%; Uptown -39.4% | Reuters documented displacement-related waterfront changes. This is a candidate mechanism, not a verified cause at an anonymous branch. |

Source URLs, publication/known-at dates and explicit counterchecks are in the [event registry](data/external/outlier_event_registry.json) and [outlier dossiers](reports/outlier_dossiers.json). The [research and pilot specification](reports/deeper_research_and_build_plan.txt) explains additional checks on airport arrivals, exchange rates, fuel, seasons and holidays. Official monthly USD exchange rates are constant in the window; monthly airport arrivals are too coarse and too few to explain individual days. Do not manufacture daily predictors from them.

![Branch and time-of-day fingerprints for investigated events](reports/figures/outlier_mechanism_fingerprints.png)

Across the original external-driver models, calendar and available weather features **did not beat internal history on held-out overall forecast accuracy**. The strongest exploratory product lead is reduced sandwich demand over four Muslim holiday closure episodes, around 4 units/day across the network; four episodes cannot establish a universal religious effect. Annual seasonality is unidentified from January-June. War effects cannot be geographically attributed without the real branch map.

### Exact submission and three-person work split

| Owner | Concrete responsibility | Acceptance condition |
|---|---|---|
| Codex teammate | Forecast/replay engine, profit optimizer, intraday Bayesian update and stockout handling, leakage/accounting tests, canonical metrics | Reproducible output, honest strong-baseline comparison, costs and service visible |
| Claude Code teammate 1 | External evidence and branch exposure; confirm event timing, coordinates and weather provenance | Every usable feature has a source, known-at time, geography and future validation; unresolved events remain alerts |
| Claude Code teammate 2 | Presentation/demo interface and pilot logging using the existing JSON/CSV contracts | Show opening plan, remaining-stock input, replenishment action, cost sensitivity, service and evidence; never present historical cards as live recommendations |

**Implemented:** both forecasting candidates; external association and robustness checks; outlier dossiers; two-stage optimizer; six cost/service scenarios; CLI demo; decision JSON; six mathematical/accounting tests. **Still needed for live operation:** verified operational inputs, UI integration, capacity/lead-time handling and a controlled prospective pilot. Full team names must be added before submission; this repository does not establish all three identities.

Start the company pilot with shadow decisions and measured preparation, waste, stockouts and batch labor. Then randomize eligible branch-days within branch and weekday between current practice and the planner, after safety and service rules are agreed. Choose the duration using baseline profit variance and a business-relevant minimum detectable gain; do not claim three branches automatically provide sufficient power. Evaluate incremental contribution after waste and extra labor, customer fulfillment and overrides. Freeze the policy before the pilot. The [build plan](reports/deeper_research_and_build_plan.txt) gives the data contract and demo narrative; [AGENTS.md](AGENTS.md) is the shared work contract for all three contributors.

### Reproduce the analysis

```bash
python src/external_analysis.py
python src/demand_model.py
python src/holiday_robustness.py
python src/outlier_forensics.py
python src/profit_policy.py
python src/decision_visuals.py
python src/verify_decision_artifacts.py
python -m unittest discover -s tests -v
```

External sources are cached and cited. The pipeline does not require querying live news to reproduce numerical results. `src/recommend.py` can use committed decision cards immediately. No frontend or automated live forecasting service is claimed to be complete.

## Earlier analysis and the team's original work

The hackathon brief describes **real transaction data with replacement identifiers**. The `SYN-` prefix does not establish that the transactions are synthetic. Earlier claims to that effect, and several causal claims in the original notebooks and pitch figures, are superseded by the [methodology audit](reports/methodology_audit.txt).

The [nine-page first-stage analysis](output/pdf/deep_analysis_report.pdf) predates the two-stage policy above. It documents [external-driver analysis](src/external_analysis.py), [forecasting and single-batch preparation](src/demand_model.py), [holiday sensitivity checks](src/holiday_robustness.py), and a [source registry](reports/external_evidence.json). Its $179 scenario is the earlier unconstrained single-batch comparison, not the current service-constrained result. The original notebooks and plotting scripts retain their earlier calculations and chart titles; interpret them with the [methodology audit](reports/methodology_audit.txt).

The original accuracy-selected model is a regularized Poisson regression using sales history, weekday and growth. June branch-category WAPE is **23.21%**, versus **24.79%** for a four-weekday baseline. April-May selected this forecast; June evaluated it with rolling weekly refits. Its earlier no-service-floor preparation simulation improves contribution by **$179 over the stronger category-based baseline** or **$707 over direct item averaging**. These amounts cannot be transferred to a different service policy. The [original 1,710 historical preparation decisions](reports/preparation_recommendations.csv) remain available for audit.

```bash
pip install -r requirements.txt
python src/external_analysis.py
python src/demand_model.py
python src/holiday_robustness.py
python src/build_analysis_report.py
```

## Project structure

```text
data/stories_transactions.csv       Raw product lines; Transaction ID identifies an order
data/external/                      External weather and event inputs
notebooks/01_eda.ipynb               Original exploratory analysis
notebooks/02_war_impact.ipynb        Original before/during/after comparisons
notebooks/03_holidays.ipynb          Original holiday and Ramadan comparisons
notebooks/04_regular_customers.ipynb Original customer descriptions and targeting rules
notebooks/05_weather.ipynb           Team weather and season analysis by branch
notebooks/06_regulars_products.ipynb Team product-mix and regular-customer shuffle checks
notebooks/07_discounts.ipynb         Team discount and placebo-window comparisons
src/eda_visuals.py                  Original descriptive charts
src/pitch_visuals.py                Original pitch charts, subject to the audit
src/external_analysis.py            External-driver analysis
src/demand_model.py                 Forecast comparison and validation
reports/methodology_audit.txt       Corrections to original claims
output/pdf/deep_analysis_report.pdf New findings, model and business decision plan
reports/external_evidence.json     External source registry
reports/figures/                    Saved charts
```

To reproduce the team's original visualizations:

```bash
python src/eda_visuals.py --save
python src/pitch_visuals.py --save
```

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

## War, holidays and weather

**War comparisons:** the sourced definitions for the new analysis use escalation on **March 2, 2026** and the temporary ceasefire effective **April 17, 2026, local time**; continuing hostilities mean there is no clean post-war period. See the [UN/UNHCR source registry](reports/external_evidence.json). Notebook 02 instead uses March 1-April 15, while the pitch script uses March 2-April 16. Their boundaries and interpretation are inconsistent. The original pre-war trend projection has no uncertainty bounds or geographically matched exposure measures. Its near-zero aggregate deviation does not establish that the war had no effect, and a later Garden shortfall does not rule out delayed effects.

**Holidays:** notebook 03 adjusts for weekday and trend, but its single-day bands are approximate and comparisons across holidays and branches are not corrected for multiple testing. Harbour's apparent Labour Day and St Maroun increases are exploratory leads. They do not establish a repeatable Christian-versus-Muslim holiday effect. Dates with uncertain observance need sensitivity checks, and calendar associations cannot identify customers' religions.

**Weather:** the pitch fits weekday plus trend to the full January-June sample and reports an in-sample R-squared near 93%. Its small correlations between residual demand and Beirut weather do not establish that weather is irrelevant. Predictive usefulness requires time-ordered held-out comparisons and branch/category interactions. Historical realized weather supports retrospective analysis; operational forecasts must use information available at the decision time.

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
