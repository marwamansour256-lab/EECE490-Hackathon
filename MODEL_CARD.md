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

Sixteen tests pass, including test-label independence of the frozen model and baselines, training economics, posterior math, stockout censoring and inventory/cash conservation. This dataset was explored before the new protocol was requested; do not claim a newly untouched test set or established future ROI.

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

Sixteen tests currently pass across the primary and supplemental models, including conjugate-posterior agreement, shared-state covariance, censored-versus-exact observations, the zero-stock observation, agreement with the earlier Poisson optimizer, inventory/cash accounting and the inability of the late action to see true lost or future demand.

## Limits and deployment gate

Recorded sales are treated as demand, although real stockouts are unknown. Preparation, waste, current inventory rules, batch times, shelf lives, total capacity, substitution and actual incremental labor are missing. The replay assumes immediate replenishment at 16:00 and same-day carryover. Simulated unsold stock is not measured company waste. No model result establishes achieved savings or net profit.

The business prototype therefore keeps the seasonal planner as the reference, offers the probabilistic candidates for review, and exposes cost/service tradeoffs. Before production: verify the missing operations data, freeze a policy and run a prospective controlled pilot against the company's actual practice. Use an agreed profit gain and service guardrails to decide whether to scale.

External events remain source-linked review alerts until geography, timing and out-of-time predictive value are verified. A real event on an outlier date is not proof that it caused the sales change.
