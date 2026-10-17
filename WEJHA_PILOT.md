# Measure whether Stories makes more money

The prototype is implemented; extra sales and layout effects have not been observed. This is the operating test that turns the model into a business decision.

## Two experiments, separate effects

**Recommendation test:** assign eligible branch-days to current practice, a simple popular-item offer, or the selected model's offer. If only two arms are feasible, compare model with popular rather than an artificially weak offer. Define eligibility before assignment, offer at most one relevant available add-on, and keep all assigned eligible orders in the analysis even when staff skip the offer, stock is unavailable or the customer declines. This avoids selecting only successful offers. Do not automatically discount.

**Layout test:** compare Stories' measured current layout with the separated candidate after confirming the physical constraints. Hold recommendation policy constant. An adjacent candidate may be a third arm if the company can support the sample size. Never add the gains from two separate tests and assume the combination has that same gain.

Randomize branch-day blocks, balance by branch and weekday, publish the assignment schedule in advance and log overrides. A branch-day has one arm. Account for repeat-customer carryover and staff learning; longer blocks or a washout may be needed for layout changes. Only three branches are available, so duration and minimum detectable profit change need baseline variance and operational feasibility, not an arbitrary one-week claim. The report's fewer-than20-days flag is a caution, not a power calculation or a pass threshold.

## Feasibility before moving food

Confirm whether customers browse/walk or staff serve everything from one till, the existing arrangement and display dimensions, refrigeration, safe storage, allergen-separation requirements, accessibility, queue flow, signage and product availability. The illustrative photo is not Stories geometry. A staff-served counter may require a visibility/signage experiment instead of a walking-path mechanism.

All 28 pastry-pair lifts in the current candidate display set are below one. The optimizer is separating popular co-purchases; the dataset does not establish a distinctive pastry-pair affinity. Do not announce that croissant–cookie separation is already known to improve spending. The six-training-date “regular” cohort is also only 3.10% of test revenue; it does not support a majority-loyal-revenue claim. Broader regularity definitions must retain their own denominator and time boundary.

## Primary money outcome

For **every assigned eligible order**, record paid revenue after discount, sold-item ingredient cost and allocated measured waste/labor/other experiment costs. Contribution is:

`paid revenue − sold-item cost − waste cost − extra experiment labor − other experiment cost`

Revenue already reflects discounts; do not subtract the discount column again. Allocate each branch-day cost once across the eligible orders, and reconcile allocated totals with the company cost log. Log setup/printing/software costs and amortization assumptions explicitly. Missing costs must remain missing and block a financial calculation; do not replace them with invented zeros. A measured absence of cost can be recorded as zero.

Compare contribution per eligible order and per branch-day, with uncertainty over assigned branch-days. Report availability, service time, observed offer acceptance, discounts and compliance alongside profit. Offer acceptance is not incremental acceptance; a customer may have bought the same item without an offer. Examine substitution/cannibalization and queue delays before rollout.

## Executable log and report

Use the separate [basket-pilot template](data/pilot/basket_pilot_template.csv). It does not alter the existing preparation pilot schema. This is a CSV/POS export workflow; the current preparation-log UI does not capture basket offers.

```bash
python src/basket_pilot_report.py data/pilot/basket_pilot_template.csv
python src/basket_pilot_report.py your_observed_export.csv --control popular --output reports/basket_pilot_observed.json
```

The empty committed template returns “no pilot outcomes observed,” with no fabricated financial uplift. For an actual export, the reporter checks duplicate orders, branch-day assignments, valid arms, measured finite costs, logged assignment probabilities and valid acceptance flags. It reports whole-branch-day bootstrap intervals, stratified by branch within arms. The analysis does not verify that assignment was actually random, handle adaptive allocation automatically, or remove repeat-customer/serial dependence. Changing allocation probabilities requires a prespecified weighting analysis.

## Column definitions

| Columns | Meaning |
|---|---|
| `experiment_id`, `experiment_type` | Unique experiment; type is `recommendation` or `layout`. Do not mix interventions within one ID. |
| `date`, `branch`, `arm` | Local calendar day/branch assignment. Recommendation arms: `current`, `popular`, `model`; layout: `current`, `adjacent`, `separated`. |
| `order_id`, `eligible` | One unique order per experiment; predeclared eligibility 0/1. No customer identity is required in this shared log. |
| `offer_shown`, `offer_item`, `accepted` | Actual exposure and paid acceptance, 0/1. Use zero for a no-offer control. Retain declined/no-offer orders. |
| `order_revenue_usd`, `order_cost_usd`, `discount_usd` | Net paid revenue, ingredient cost of sold items and separately audited discount expenditure. Refunds/cancelled orders need a prespecified reconciliation rule. |
| `waste_cost_allocated_usd`, `extra_labor_allocated_usd`, `other_cost_allocated_usd` | Measured costs allocated once, never full daily cost repeated on every order. |
| `service_seconds`, `stock_available` | Service duration when measured; empty duration stays unknown. Availability 0/1 is logged without excluding assigned orders. |
| `assignment_probability`, `override_reason` | Prespecified probability of the assigned arm and operational deviations. Probability is not an estimated offer-conversion rate. |

Freeze success/stop rules before launch: a business-relevant positive contribution difference after all costs, acceptable uncertainty, service/availability guardrails and no unsafe storage/access changes. Use shadow operation to verify logging before experimental offers or rearrangement. No pilot has been run in the hackathon data.
