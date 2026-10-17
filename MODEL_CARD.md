# Stories trained demand and preparation model

**Implemented basket-growth extension:** `src/basket_recommender.py` fits a hierarchical, smoothed conditional co-purchase model with branch/daypart/aggregate regular-customer contexts and training unit margins. A 500-anchor shrinkage prior is selected on March 11–31 by recovered-item margin@3. Refit Jan–Mar; test Apr–Jun without updating counts, cohorts or economics. `models/basket_recommender.json` is the portable artifact; `app/recommendation_data.js` serves identical offline weights/rankings. The CLI and `app/wejha.html` exclude basket/unavailable items and aggregate multiple anchors by maximum candidate score. This is aggregate context, not individual customer personalization or a joint multi-item probability model.

Test: 8,801 multi-product orders / 18,401 masked targets, equal order weight. Hit@3 13.29%; recovered-item margin proxy $0.5080/order versus margin-popularity $0.4956 (+2.51%); conditional paired seven-day block interval for the difference $0.0028–$0.0251/order. Plain popularity has higher Hit@3 (13.74%). Historical relevance and recovered already-purchased margin do not estimate incremental sales. No offer acceptance, exposure, layout or customer-path labels exist. The six-training-purchase-date cohort has 183 IDs and only 3.10% of test revenue; broader loyalty claims require their own definition and denominator. No IDs are included in exported assets. [Full recommendation report](reports/recommendation_results.json).

`src/wejha_layout.py` separately optimizes a co-purchase-count-weighted normalized slot distance, respecting optional fixed positions/storage compatibility. Default eight Bakery candidates permit exact enumeration; larger displays use a seeded local search with no global optimum claim. Adjacent/separated/random-reference distances are 0.373/0.511/0.425. These values measure a geometric hypothesis, not revenue uplift. A company-confirmed current layout and controlled contribution experiment are needed. The current complete suite has **44 passing tests**, covering split rejection, ranking parity, exclusions, sparse probabilities, feasible layouts, inventory math, integrated profit arithmetic and portable provenance.

**Current scope:** external context may predict Stories sales. Another business's sales must not replace Stories test targets. The earlier transaction-only model is retained as the control; it is not a user-imposed limit on features. See [EXTERNAL_CONTEXT.md](EXTERNAL_CONTEXT.md) for the active multi-head extension and every-day evidence engine. It includes an internal category-volume head, externally conditioned order-count head, category-specific item-mix experts and the preparation optimizer, with explicit evidence/source gates.

The new traffic head's training-selected month-cycle expert reduces April-June branch-order WAPE from 9.4644% to 9.2658%. The volume expert selected on training slightly worsens category WAPE (25.2262% vs 25.2077%), so there is no claim of a general volume improvement. Item-mix experts improve all-item WAPE from 53.2678% to 52.6710%; most of that gain is internal pooling, not external features. Their perishable contribution is $10,817.66 vs the control's $10,934.29 at similar service. The profit-selected experimental mix is worse still ($10,560.98). These candidates are retained for review, not declared superior business policies. Twenty-three tests pass across all stages. A separate national reachability audit adds measurement/access hypotheses without inferring a store outage.

## Transaction-only control: frozen chronological half/half

This retained control trains on the earlier half and tests on the later half, using transaction-only features. Its implementation is `src/strict_model.py`, with reusable artifact `models/strict_half_model.joblib` and prediction CLI `src/predict_strict.py`. External features are allowed in the active extension; all sales targets remain Stories. Earlier rolling and Bayesian models below remain supplemental research; their results must not be relabeled as strict half/half results.

**Model:** pooled L2-regularized Poisson regression with branch/category effects, category-weekday and branch-weekday interactions, linear growth and category-specific growth terms. Allocate category demand to items using smoothed item shares from the last 28 training days. A preparation optimizer converts demand and training-period economics into quantities. Features come only from identifiers, dates and transactions in the supplied CSV.

**Training:** 90 dates, Jan 1-Mar 31, 22,395 product lines. The last 21 training dates select alpha from 0, 0.001, 0.01, 0.1 and 1 using category Poisson deviance. Alpha 0.001 is selected, then refit on all training dates. The trained model beats the two baselines on inner validation and is selected there, without test-period selection.

**Test:** 91 dates, Apr 1-Jun 30, 28,782 lines. Every prediction is made using the March 31 artifact. Parameters, product shares, economics and intraday splits remain frozen; no test lag or test-period refit is allowed. No external files are predictive inputs. The split is by daily timeline, not exactly half the product-line count. See [EVALUATION.md](EVALUATION.md).

Test category WAPE is **25.21%**, versus **26.64%** for a seasonal baseline with training-only growth and **28.22%** for a frozen seasonal baseline. Perishable-item WAPE is **52.03%**, versus **53.58%** and **53.75%**. Under a common one-batch 90% expected unit-fill rule, contribution is **$10,934.29**, versus **$9,915.15** and **$13,409.52**, respectively. Realized fulfillment is **92.09%**, **92.19%**, and **88.22%**. The contribution difference versus the growth baseline is $1,019.13; the frozen baseline's higher contribution trades away fulfillment. Simulated preparation is not measured company behavior.

**Improved version v2 (teammate 1, selected on the inner validation window only):** level calibration 0.942 (inner-validation actual/predicted; the original over-forecast those 21 days by 6.2%) and item shares from all training dates with smoothing 10. April-June: category WAPE **24.74%** (vs 25.21%), all-item **52.09%** (vs 53.27%), perishable **51.57%** (vs 52.03%), contribution **$11,944.21** (vs $10,934.29; +$1,010, 7-day block interval about $884-$1,126), fulfillment **90.92%** (vs 92.09%), unsold units 8,401 (vs 8,929). The artifact's selected method is now `regularized_poisson_v2`; `growth_plan.py` and `predict_strict.py` use it through `strict_model.category_mean`/`share_column`. Category-engine upgrades (series × weekday, per-series trend, recency weights) lost on inner validation and are not used. Linear regression with the same features and selection rule was also tested (`src/linear_baseline_check.py`, `reports/linear_baseline_check.json`): April-June category WAPE 26.65% on raw counts and 25.75% on log counts, both worse than the Poisson model (25.21%) and v2 (24.74%). Raw-count linear regression can also predict negative sales, which a count model avoids. XGBoost (Poisson objective; depth, learning rate and tree count chosen on the last 21 training dates) was tested the same way (`src/xgboost_check.py`, `reports/xgboost_check.json`): category WAPE 27.39% on its own and 27.08% when boosting on top of the Poisson forecast, both worse than the Poisson model (25.21%) and v2 (24.74%). Tree models cannot extrapolate the growth trend past the last training date, which is why the stand-alone version under-forecasts April-June by about 10%. A decision tree and a random forest were tested the same way (`src/tree_models_check.py`, `reports/tree_models_check.json`): stand-alone category WAPE 28.55% and 28.04% (under-forecasting by 11-12%, the same extrapolation problem), and 26.51% and 26.30% when they only learn a correction multiplier on the Poisson forecast. All are worse than the Poisson model (25.21%) and v2 (24.74%).

The complete 20-candidate share search is reproducible with `python src/audit_item_share_selection.py`; [all grid results](reports/strict_v2_share_selection.json) confirm the committed all-training/smoothing-10 choice using Jan1–Mar10 fitting and Mar11–31 validation only. Calibration is estimated on those validation outcomes, so that window is a selection/calibration set, not an unbiased validation of its own calibration. The rebuilt saved artifact uses the locked scikit-learn 1.9.1 environment. The v2 contribution gain includes **150 fewer fulfilled units** than the original; do not describe its service as identical.

The conditional seven-day block resampling interval for the $1,019 difference is approximately $770-$1,276. It does not include missing operational costs, model-selection uncertainty or future ROI. Independent artifact verification reconciles 49,140 item-day-method records (four methods, including v2) and reproduces forecasts from the saved model, including the v2 calibration.

These checks are part of the 44-test suite run by `python src/verify_submission.py`, including test-label independence of the frozen model and baselines, training economics, posterior math, stockout censoring and inventory/cash conservation. This dataset was explored before the new protocol was requested; do not claim a newly untouched test set or established future ROI.

```bash
python src/strict_model.py
python src/predict_strict.py --date 2026-06-30 --remaining 0
python -m unittest discover -s tests -v
```

## Supplemental Bayesian experiment

## Business decision

For each branch and perishable item, choose an opening preparation quantity and a later replenishment quantity. Account for ingredient expense, extra batch labor, service and unsold stock. At 16:00, a manager enters remaining inventory; the engine explains the relevant demand distribution and second-batch action.

The model implementations, trained artifacts and generated results are committed to GitHub. The policy and matched-prior comparisons below use the earlier rolling protocol, not the primary strict split. The current complete suite has 44 tests plus four artifact-verification scripts; `python src/verify_submission.py` runs them.

## Model architecture

1. **Demand prior:** branch/category regularized Poisson regression, or a four-matching-weekday seasonal baseline. Allocate to products using prior 28-day smoothed item shares. Only earlier transactions enter these features.
2. **Probability model:** compare independent Poisson demand, independent negative-binomial stage demand and a shared Gamma-Poisson daily intensity. For the shared model, early and late demand can move together through a latent daily demand state.
3. **Intraday update:** if early stock remains, update using the observed early sales count under an uncensored-sales assumption. If all opening stock sells, condition on demand being **at least** that quantity. A stockout does not reveal the true demand count.
4. **Decision optimization:** integrate both the future inventory observation and the late decision in a two-stage dynamic program. Maximize expected contribution subject to expected unit-fill constraints. No actual late demand enters either decision.
5. **Model gate:** select the probability model using April-May predictive log score. Keep a control that does not update from early sales; the adaptive model must earn its place. Report business results separately from prediction scores.

This is a concrete implementation of established probabilistic and optimization methods, not a claim to have invented Gamma-Poisson inference. References: [Stan posterior prediction](https://mc-stan.org/docs/2_36/stan-users-guide/posterior-prediction.html), [SciPy negative-binomial distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html), [MIT inventory dynamic programming lecture](https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/resources/mit6_231f15_lec4/).

## Bayesian update

Let `Z ~ Gamma(k, rate=k)`. Conditional on Z, early demand has mean `mu_early * Z` and late demand has mean `mu_late * Z`.

For an exact uncensored early count `e`, the posterior for Z is `Gamma(k + e, rate=k + mu_early)`. The updated late mean is:

```text
mu_late * (k + e) / (k + mu_early)
```

When opening stock `q` sells out, the engine instead conditions on `early demand >= q`, using the joint early/late distribution. The posterior is a mixture over all compatible early counts. It never substitutes simulated lost demand for information available to the manager.

The static control keeps early and late demand independent. Thus a large early count cannot automatically force an afternoon increase when validation does not support that relationship.

## Training and evaluation

- Original mean models: January-March initial training, April-May chronological selection and rolling June evaluation. Weekly refits can use earlier June observations for later June decisions.
- Intraday candidate selection: April-May only, 3,477 perishable branch-item-days. Two mean families, seven finite dispersion shapes with/without updating, plus a Poisson control for each mean family: **30 candidates**.
- June retrospective check: **1,710 branch-item-days**, 4,657 recorded units across three branches and 57 perishable branch-item combinations.
- Selection metric: mean negative log predictive probability for late demand; lower is better. Also report WAPE, absolute error, discrete interval coverage and interval width.
- June was examined in earlier analysis before this model was designed. This is a chronological retrospective check, **not an untouched confirmation set**.

## What the current results support

The April-May probability winner is the internal mean model with negative-binomial stage demand, dispersion shape 10 and **no early-sales update**. Its validation negative log score is 1.4681, versus 1.4891 for the seasonal Poisson control. The best adaptive candidate has shape 25 and score 1.4717. On June, those three scores are 1.5672, 1.5877 and 1.5731 respectively.

These results favor allowing more uncertainty than Poisson for the internal forecast, but **do not establish that early-sales updating improves prediction**. A posterior update is implemented and available for comparison; the evidence does not justify making it the default solely because it is more complex.

The first completed low-cost June policy replay gives:

| Candidate | Contribution | Recorded units fulfilled | Difference versus its own one-batch comparator |
|---|---:|---:|---:|
| Seasonal Poisson control | $5,341.65 | 93.43% | +$410.73 |
| Probability-selected static NB, shape 10 | $4,437.94 | 95.15% | +$223.37 |
| Best adaptive candidate, shape 25 | $4,895.53 | 94.33% | +$527.26 |

The adaptive candidate's larger difference versus its own one-batch plan **does not mean it beats the seasonal planner**: its absolute contribution is lower, and service differs. The selected probability model also does not maximize replay contribution. Do not choose the headline by switching comparators. A same-joint-prior frozen-signal control earns $4,913.70 with 94.57% fulfillment; adaptive updating earns $18.18 less with 94.33% fulfillment. This isolates the decision to use early information while holding the joint prior and one-batch comparator fixed. Other static-vs-adaptive controls change dependence as well as updating and are labeled accordingly.

Costs here are $0.25 per SKU second batch plus $0.10 per replenished unit, zero salvage and a 90% expected unit-fill floor. Two-stage floors apply separately to stages; one-batch floors apply to the day. Expected-fill constraints do not guarantee an actual fill rate or a probability of no stockout.

## Implementation and commands

```bash
python src/intraday_model.py --score-only
python src/intraday_model.py
python src/adaptive_recommend.py --candidate selected --remaining 0
python src/adaptive_recommend.py --candidate best-adaptive --remaining 0
python src/adaptive_recommend.py --candidate seasonal-control --remaining 0
python -m unittest discover -s tests -v
```

`src/intraday_model.py` creates feature, prediction, selection, replay and decision-card artifacts. `src/bayesian_policy.py` implements the joint demand distribution, censored update and dynamic program. `src/adaptive_recommend.py` returns the historical plan and posterior explanation; it accepts cost and service settings.

The 44-test suite (`python src/verify_submission.py`) covers the primary and supplemental models, including conjugate-posterior agreement, shared-state covariance, censored-versus-exact observations, the zero-stock observation, agreement with the earlier Poisson optimizer, inventory/cash accounting and the inability of the late action to see true lost or future demand.

## Limits and deployment gate

Recorded sales are treated as demand, although real stockouts are unknown. Preparation, waste, current inventory rules, batch times, shelf lives, total capacity, substitution and actual incremental labor are missing. The replay assumes immediate replenishment at 16:00 and same-day carryover. Simulated unsold stock is not measured company waste. No model result establishes achieved savings or net profit.

The business prototype therefore keeps the seasonal planner as the reference, offers the probabilistic candidates for review, and exposes cost/service tradeoffs. Before production: verify the missing operations data, freeze a policy and run a prospective controlled pilot against the company's actual practice. Use an agreed profit gain and service guardrails to decide whether to scale.

External events remain source-linked review alerts until geography, timing and out-of-time predictive value are verified. A real event on an outlier date is not proof that it caused the sales change.

# Moved from README: preparation decision, earlier planner and first-stage analysis

These sections were in the README before it was shortened to the business insights and the main v2 model. They describe the two-stage preparation replay, the earlier rolling planner and the first-stage analysis. The main model is v2; see the top of this file and README.md.

## Business application and supplemental experiments

**A branch-and-item preparation planner that chooses an opening batch and a second batch at 16:00, prices the extra work, and reports its service and profit tradeoffs.** External-event research supplies evidence-linked review alerts. It changes numerical forecasts only after exposure and predictive value are established.

The implemented decision model is a **two-stage stochastic inventory dynamic program**, using Poisson demand and an expected unit-fill constraint. Its inputs are demand, branch-item selling price and ingredient cost, intraday mix, replenishment cost and salvage. Its output is an opening quantity plus a lookup table: **given remaining stock at 16:00, prepare this many more units**. This is an executable decision, not just a sales correlation.

The forecasting candidates are a regularized pooled Poisson regression and a strong seasonal baseline: the preceding four matching weekdays at branch/category level, allocated to items using prior 28-day shares. The complex forecast wins the original category accuracy comparison but **loses the new service-constrained profit comparison**. The demo defaults to the transparent baseline; this choice is exploratory and needs a future prospective evaluation. We do not claim complexity itself creates value.

## Earlier rolling-planner demo

```bash
pip install -r requirements.txt
python src/recommend.py --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0
python src/recommend.py --mean 5 --early-share 0.4 --price 6 --cost 2 --setup 1 --premium 0.25
python -m unittest discover -s tests -v
```

The first command reviews a **June 30, 2026 historical decision**, not today's forecast. The second lets the company change business inputs. Output includes opening units, the second-batch table, expected contribution and assumptions. It never uses afternoon sales to choose the afternoon batch. Immediate replenishment and unconstrained capacity are simulation assumptions; production use requires actual lead times and capacity.

## Where the money comes from

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

## Reproduce the analysis

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

External sources are cached and cited. The pipeline does not require querying live news to reproduce numerical results. `src/recommend.py` can use committed decision cards immediately. No automated live forecasting service is claimed to be complete.

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


# Appendix: previous README overview, kept word for word

The README was shortened on 2026-10-17 to the business insights and the main v2 model. The text below is the earlier README's overview, model section and project tree, unchanged, so nothing is lost. Some status lines describe the team's work in progress at that time (for example test counts that have since grown).

## Stories Profit Planner | EECE490 Hackathon

**Team:** Charbel Manih (202502099, mastercoder67), Omar Al Aghbari (202514349, aghbar11), Marwa Mansour (202505788, marwamansour256-lab). [Team and responsibilities](TEAM.md).

**A working sales and preparation decision system for Stories Coffee.** Basket Growth recommends relevant add-ons, Wejha Lab compares product-display arrangements, and the preparation model turns demand into opening batches and replenishment actions. The display experiment separates paired products so customers may encounter more of the counter; placement-induced sales growth still requires a pilot. Source code, trained artifacts and generated offline demo data are committed. [Submission quickstart](SUBMISSION.md), [shared team contract](WEJHA_BUILD_PLAN.md).

| Start immediately | Owner | Deliverable and status |
|---|---|---|
| Model and backend | Codex teammate | **Implemented and published:** frozen basket recommender, five candidate comparisons, offer/preparation integration, external-context models, constrained layout optimizer and generated comparisons; all 39 tests pass. |
| Offline interface | Claude Code teammate 2 / Marwa | New `app/wejha.html`, linked from the existing planner. Load `window.STORIES_RECOMMENDATIONS` and `window.STORIES_WEJHA`; contracts are in the build plan. Own these UI files. |
| Business validation and experiment | Claude Code teammate 1 / Omar | New `WEJHA_PILOT.md`, separate `data/pilot/basket_pilot_template.csv`, and `reports/wejha_business_review.md`. Check browsing/service feasibility, pair evidence, margin assumptions and controlled profit measurement. Own these files. |

**Sync rule:** the latest user instruction authorizes ongoing pushes of code, generated artifacts and documentation. Each owner pulls safely, preserves others' work and pushes usable milestones. Planned filenames are not evidence of a completed implementation. Existing preparation cards, pilot schemas, baseline results and the newly pulled Day Explorer remain supported.

**Basket/layout pilot is ready:** [WEJHA_PILOT.md](WEJHA_PILOT.md), a separate [order-level CSV template](data/pilot/basket_pilot_template.csv), and `python src/basket_pilot_report.py <observed_export.csv> --control popular` measure recorded contribution after sold-item costs, allocated waste, extra labor and other costs. The reporter preserves declined/no-offer orders, checks branch-day assignment and duplicate orders, and reports branch-day uncertainty. It does not subtract discounts twice or invent outcomes for the empty template. [Business review](reports/wejha_business_review.md). **44 tests now pass.** The layout remains based on pooled same-order co-purchases, as clarified by the user; it is not a per-person rearrangement.

**Run the integrated demo:** open [`app/wejha.html`](app/wejha.html) in a browser for basket recommendations, availability controls, three display arrangements and an editable incremental-profit scenario. [`app/planner.html`](app/planner.html) links to it and retains preparation planning and the every-day diagnostic. Both pages work offline with the committed generated scripts.

**Offer → stock → profit integration is implemented:** `python src/growth_plan.py --branch Harbour --item Espresso --daypart morning --output reports/growth_demo.json` produces complementary food offers, the frozen preparation plan for each food item, an explicit zero-default incremental-profit scenario, and a separately labeled retrospective daily diagnostic. It loads committed artifacts and adds no invented offer uplift to demand. `python src/growth_plan.py --export-preparation` produces 57 historical perishable cards in `app/preparation_data.js`. [Integration contract and optional UI additions](GROWTH_INTEGRATION.md). **39 total tests pass.** The complementary filter is a business rule; the evaluation numbers below belong to unrestricted ranking.

```bash
pip install -r requirements.txt
python src/verify_submission.py
python src/basket_recommender.py recommend --branch Harbour --item "Butter Croissant" --daypart morning
python src/basket_recommender.py train
python src/wejha_layout.py
python -m unittest discover -s tests -v
```

**Trained recommendation model:** empirical-Bayes shrinkage of conditional item co-purchases toward pooled popularity, with branch/daypart/aggregate regular-customer contexts and unit-margin ranking. March validation selects strength 500; all parameters/economics freeze on March 31. In **8,801 April–June multi-product orders**, the recovered-item margin@3 proxy is **$0.5080/order versus $0.4956** for margin-weighted popularity, a **2.51% relative gain**. The paired seven-day block interval for the difference is **$0.0028–$0.0251/order**. This metric rewards recovering an item that was already bought; it is **not additional sales or company profit**. Hit@3 is **13.29%**, versus **12.90%** for the margin baseline and **13.74%** for plain popularity. [Full evaluation](reports/recommendation_evaluation.md), [generated results and cohort denominators](reports/recommendation_results.json).

**Wejha Lab:** exact enumeration of all arrangements of eight bakery candidates compares adjacent, separated and random-reference layouts. Weighted pair distance is **0.373 / 0.511 / 0.425** respectively. These are geometric exposure scores; higher distance is the user's testable mechanism, not proven sales uplift. Fixed positions and compatible storage zones are supported through a constraints JSON. Real counter dimensions, current placement and customer paths are unobserved. [Layouts and assumptions](reports/wejha_layouts.json). The frozen six-training-date regular cohort accounts for **3.10% of test revenue**; this specific definition does not support the premise that most revenue comes from loyal customers.

**Data scope:** outside weather, calendar, access and other context are allowed as features. Training and test **sales targets remain Stories' own sales**. The transaction-only model below is the control. See [EXTERNAL_CONTEXT.md](EXTERNAL_CONTEXT.md) for expanded research, external models and the every-day evidence ledger.

The published external-context implementation includes nine cached hourly weather/forecast/air-quality streams, six months of IODA internet reachability, 73 daily context columns plus the separate internet audit, **2,337 exploratory test rows** and **724 day/branch dossiers covering all 181 dates**. A separate externally conditioned order-count head improves test WAPE from **9.46% to 9.27%**; daily product-volume gains are not established, and improved item forecast error did not improve simulated profit. No external sales dataset is used. [Source reconciliation](reports/source_reconciliation.md) supplies primary URLs missing from the teammate event registry.

Analysis of January-June 2026 transactions from three anonymized branches: Harbour, Garden and Uptown. The goal is to identify decisions that improve gross profit, validate external demand signals, and distinguish measured results from business assumptions.

**Published implementation:** demand, inventory, external-context, recommendation and layout code, saved models, browser assets and numerical audits are in the repository. `python src/verify_submission.py` runs all 39 tests and four artifact-verification scripts. Browser script/schema/ranking checks are recorded in `reports/ui_contract_qa.json`; the committed UI screenshots in reports/figures/ui were visually inspected; this Codex session had no connected live browser. The **frozen chronological half/half transaction-only model** remains the control, and earlier rolling/Bayesian comparisons remain separate experiments. [Evaluation protocols](EVALUATION.md), [model card](MODEL_CARD.md).

### Transaction-only control: chronological half/half

The model is an **L2-regularized pooled Poisson regression**, followed by item allocation and a profit-aware preparation rule. It learns branch/category demand, weekday patterns and growth using only `stories_transactions.csv`. It uses no weather, holiday, war, news or external data files as model inputs.

- **Training:** Jan 1-Mar 31, 90 dates and 22,395 product lines.
- **Testing:** Apr 1-Jun 30, 91 dates and 28,782 product lines.
- **Selection:** the last 21 training dates select regularization; the selected alpha is 0.001. Refit on the 90 training dates, then freeze everything.
- **Strict boundary:** the whole test period is forecast at March 31. No test sales lags, refits, item shares, prices, costs or intraday splits enter the model. Half means half of the daily timeline, keeping complete dates/orders together; growing transaction volume makes row counts unequal.

| Frozen model, April-June test | Category WAPE | Perishable item WAPE | Simulated contribution | Actual recorded units fulfilled |
|---|---:|---:|---:|---:|
| **Improved model (v2, selected on inner validation)** | **24.74%** | **51.57%** | **$11,944.21** | **90.92%** |
| Trained Poisson model (original) | 25.21% | 52.03% | $10,934.29 | 92.09% |
| Seasonal baseline with training-only growth | 26.64% | 53.58% | $9,915.15 | 92.19% |
| Frozen four-weekday baseline | 28.22% | 53.75% | $13,409.52 | 88.22% |

**v2 (teammate 1):** the same engine and alpha, plus two changes chosen only on the last 21 training dates: (1) a level calibration factor of 0.942 (inner-validation actual ÷ predicted, correcting a +6.2% over-forecast) and (2) item shares from all 90 training dates with smoothing 10 instead of the last 28 days with smoothing 1 (best of 20 window/smoothing settings on inner-validation item deviance). On April-June it improves category WAPE (25.21% → 24.74%), all-item WAPE (53.27% → 52.09%) and perishable WAPE (52.03% → 51.57%), and adds **$1,009.92** simulated contribution versus the original model (7-day block interval about $884-$1,126) with 528 fewer unsold units, while realized fulfillment stays above 90% (90.92%). The artifact now selects v2; the original rows are kept for comparison. Upgrades to the category engine itself (branch × category × weekday terms, per-series trends, recency weighting) were tried and lost on inner validation, so they are not used. `python src/predict_strict.py` also accepts `--cost-multiplier`, `--price-multiplier` and `--no-discount` to run the ANALYSIS.md cost, price-test and discount scenarios; these hold demand constant and do not estimate customer response.

The original model improves simulated contribution by **$1,019.13 versus the growth baseline**, with 240 fewer unsold units and 13 fewer fulfilled units. The frozen baseline earns more simulated contribution while fulfilling fewer recorded units; selecting it solely for profit would conceal that service tradeoff. All use the same one-batch rule with a 90% **expected** unit-fill constraint, which does not guarantee 90% realized fulfillment. These are conditional inventory simulations, not measured savings or a comparison with Stories' actual operating policy. This dataset had already been explored before the new split was requested; the clean computational boundary does not make it a newly untouched test set.

```bash
pip install -r requirements.txt
python src/strict_model.py
python src/predict_strict.py --date 2026-06-30 --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0
python -m unittest discover -s tests -v
```

Training saves a reusable artifact at `models/strict_half_model.joblib`; prediction loads that artifact without reading transaction or external data. With the selected v2 method, the example forecasts about **3.67 Turkey & Cheese Sandwiches at Harbour**, prepares **4 at opening**, and prepares **4 more at 16:00 if none remain** (the original method forecast 4.04), under stated low replenishment costs. This second-batch example is a decision-model scenario; the table above evaluates the simpler one-batch policy. Committed test predictions, finance and metadata are in `reports/strict_half_predictions.csv`, `reports/strict_half_finance.csv` and `reports/strict_half_results.json`.


### Project structure

Everything below is committed. Start with [SUBMISSION.md](SUBMISSION.md) to run the demo. Each folder with many files has its own index: [reports/README.md](reports/README.md), [notebooks/README.md](notebooks/README.md), [data/external/README.md](data/external/README.md), [app/README.md](app/README.md).

```text
Documentation (repo root)
  README.md                 Project overview, results and commands (this file)
  SUBMISSION.md             How to run the demo and verify the submission (start here)
  AGENTS.md / CLAUDE.md     Shared three-person work contract and rules
  MODEL_CARD.md             Demand and preparation model, including v2
  EVALUATION.md             Frozen half/half evaluation protocol and results
  EXTERNAL_CONTEXT.md       External-context models and evidence
  GROWTH_INTEGRATION.md     How offers connect to preparation and profit scenarios
  WEJHA_BUILD_PLAN.md       Basket recommender and display-layout sprint contract
  ANALYSIS.md               Business recommendations and next actions (teammate 1)

app/                        Offline demo pages (open in a browser, no server): see app/README.md
  wejha.html                Basket Growth and display layouts (teammate 2)
  planner.html              Preparation, day explorer, evidence, pilot log (teammate 2)
  *_data.js                 Generated by the Python scripts; do not edit by hand

data/
  stories_transactions.csv  Raw Stories product lines (never edited; all rows kept)
  external/                 Weather, CPI, events, context features: see data/external/README.md
  pilot/                    Pilot-log templates

models/                     Saved trained models (preparation, basket recommender, context heads)
notebooks/                  Analysis notebooks 01-09: see notebooks/README.md
src/                        All model, analysis and verification code; src/verify_submission.py runs every check
tests/                      Unit tests (python -m unittest discover -s tests)
reports/                    Generated results, evidence and team documents: see reports/README.md
  figures/                  Saved charts (figures/ui/ holds demo screenshots)
output/pdf/                 First-stage analysis report (PDF)
requirements.txt / requirements-lock.txt   Dependencies (use the lock file to load saved models)
```

To reproduce the team's original visualizations:

```bash
python src/eda_visuals.py --save
python src/pitch_visuals.py --save
```

