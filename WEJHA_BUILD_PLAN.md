# Basket growth and the Stories “wejha” experiment

**Priority handoff, October 17.** The recommender, trained artifact, layout engine, offline UI and offer-to-preparation CLI are now implemented and published. Read [SUBMISSION.md](SUBMISSION.md) for working commands and [GROWTH_INTEGRATION.md](GROWTH_INTEGRATION.md) for complementary filtering and the prep-card export. The interface contracts below are implemented; company-confirmed geometry/operations and prospective profit measurement remain open. Continue publishing usable milestones immediately and preserve teammate work.

## What we are building

Stories Basket Growth recommends one relevant, available add-on to the current order. The Wejha Lab compares arrangements of displayable products, including the user's idea of separating frequently paired items so customers see more of the display. Together with the existing preparation planner, this is a decision system: **what to offer, how to display it, and whether enough stock can be prepared profitably**.

The business target is incremental gross contribution per eligible order after discounts, incremental waste, labor and implementation costs. Larger baskets and recommendation accuracy are supporting metrics. We cannot estimate incremental purchases or placement uplift from historical purchases alone.

The photo is an illustration, not a measured Stories counter. We do not have shelf dimensions, customer paths, actual stock, staff-service arrangements or the current planogram. A separation layout is an experiment candidate, not a proven profit optimum. If staff retrieve all items, customers may never walk between products; test sightlines/signage rather than assuming a walking path.

## Start now: three people, separate files

| Owner | Immediate deliverable | Files owned for this sprint | Acceptance |
|---|---|---|---|
| Codex teammate | Train/evaluate the basket recommender; export offline data; build the constrained layout comparison; meaningful tests; publish existing checked model work | `src/basket_recommender.py`, `src/wejha_layout.py`, `tests/test_basket_recommender.py`, `tests/test_wejha_layout.py`, `models/basket_recommender.json`, `reports/recommendation_*`, `reports/wejha_*`, generated `app/recommendation_data.js`, `app/wejha_data.js` | Frozen Jan–Mar fitting, Apr–Jun evaluation, strong popularity baselines, no fabricated uplift, CLI outputs and generated data agree |
| Claude Code teammate 2 / Marwa | Build the usable offline recommendation and display interface; link it from the existing planner | **New `app/wejha.html`**; minimal link in `app/planner.html`; UI-specific screenshots/checks | Works offline; add/remove basket items, branch/daypart/cohort and availability controls; shows add-on reasons and evidence status; renders layout alternatives; explicit profit scenario inputs |
| Claude Code teammate 1 / Omar | Validate the business mechanism and feasible pilot; investigate whether shortlisted pairs exceed popularity; prepare concise presentation wording | **New `WEJHA_PILOT.md`**, **new `data/pilot/basket_pilot_template.csv`**, **new `reports/wejha_business_review.md`**; additions to company-information request | Records unknown counter/service constraints; distinguishes observed pairs from incremental demand; specifies randomized baseline comparison, contribution and service guardrails; reconciles any pair claims with the new model report |

Both Claude teammates can begin their files immediately using this contract. Do not edit the other owner's new files. Codex maintains shared README/AGENTS/CLAUDE changes during this sprint; send proposed shared-document wording through the normal team workflow rather than rewriting those files concurrently. These human work assignments do not authorize launching additional AI agents.

## Repository rules for the deadline

1. Inspect `git status`, then pull safely before starting and before integrating. Preserve uncommitted work. If changes overlap, inspect and merge; no hard reset, destructive checkout, force push or blind conflict resolution.
2. **The latest user instruction authorizes pushing all task-related code, generated artifacts and documentation**, not only Markdown. Commit and push at each usable milestone. Do not wait for the whole system to finish. Exclude credentials, virtual environments, caches and temporary debug output.
3. Commit only files you own or changes you have reviewed. Do not scoop up another person's incomplete edits with `git add .`. Before pushing, check the staged diff and pull/merge safely if another teammate has advanced main.
4. Mark planned, implemented and validated capabilities accurately in README. A file name in this contract is not evidence that its model has run. Preserve all existing baseline and failed-experiment results.
5. Do not change the existing preparation-card or pilot-log schemas. Basket/placement experiments get their own data contract and log. Keep the current offline planner working.

## Recommendation model and evaluation rules

- Transactions are baskets keyed by Transaction ID, with branch/date consistency checked. Repeated product lines count once for co-occurrence; preserve original quantities/economics separately. Never deduplicate raw sales files.
- Sales targets come exclusively from Stories. External weather/calendar/event context remains allowed under the existing known-at/source rules; it is optional for this sprint, not a requirement to force a correlation into the recommender.
- Train on Jan 1–Mar 31; evaluate on Apr 1–Jun 30. Choose model settings using only the final 21 dates of the training half, refit once, then freeze. No April–June baskets, customer visits, prices, costs or product counts in fitted parameters or model selection.
- Main offline task: **masked-item basket completion**. Hide an item from a multi-product order and recommend from the remaining items. Order-line position is not verified purchase sequence; do not call this a next-click or causal upsell test. Keep complete orders/dates together and distinguish multi-product completion from empty-basket popularity.
- Compare ordinary item popularity, margin-weighted popularity and a regularized pair model. Shrink sparse conditional counts toward training popularity; report support, conditional frequency and lift separately. If the pair model fails to beat the baseline on training validation, use the baseline and say so.
- Rank by a training-estimated purchase-frequency/margin score. This is an association score, not an estimated probability of accepting an offer or incremental dollars. Disclose negative/weak pair evidence rather than choosing only attractive pairs after seeing test results.
- Show Hit@3, reciprocal rank and recovered-item gross-margin proxy, sample sizes and date-based uncertainty where feasible. Report repeat-customer and unidentified-customer subsets without using test-period history to define them.
- Initial regular-customer definition: at least **six distinct purchase dates in Jan–Mar**, configurable and frozen before testing. “Unidentified” is not proof of a walk-in. Do not claim loyal customers provide most revenue without a denominator and supporting computation.
- Never recommend an item already in the basket or marked unavailable. No automatic discounts. Product exclusions, dietary suitability and allergens need real company inputs; do not infer them from product names or customer religion.
- UI customer groups are aggregate `all` and `regular`, not customer identities. Do not publish a customer-ID lookup table in browser assets. Sparse contexts fall back to pooled training evidence and disclose the fallback.

## Offline frontend contract, version 1

Codex will generate `app/recommendation_data.js`, which assigns **`window.STORIES_RECOMMENDATIONS`**. It is loaded with a local script tag; no server, fetch or live API required. The same payload is saved as `models/basket_recommender.json`.

```javascript
{
  schema_version: 1,
  status: "Historical model; prospective sales uplift unmeasured",
  trained_through: "2026-03-31",
  selected_method: "<filled by training validation>",
  branches: ["Harbour", "Garden", "Uptown"],
  dayparts: ["all", "morning", "midday", "evening"],
  cohorts: ["all", "regular"],
  products: [{product: "<name>", category: "<category>"}],
  contexts: {
    "Harbour|all|all": {
      training_orders: 0,
      fallback: "<data-driven pooling description>",
      baseline: [],
      by_anchor: {"<product>": []}
    }
  },
  evaluation: { /* generated test metrics; never fill with placeholder wins */ }
}
```

Every recommendation row contains `product`, `category`, `score`, `conditional_probability`, `baseline_probability`, `lift`, `pair_orders`, `anchor_orders`, `unit_margin_usd`, `reason`, and `evidence_status`. Probabilities describe smoothed historical basket frequency, not offer acceptance. Unit margin is a Jan–Mar branch estimate; discounts, waste and extra labor are separate costs.

**Exact frontend ranking:** choose the context key `branch|daypart|cohort`. With an empty basket, use `baseline`. Otherwise collect the lists in `by_anchor` for all basket items, keep the highest-score row for each candidate, exclude basket/unavailable items, sort descending `score` then ascending `product`, and show three. An absent context falls back to `branch|all|all`; an unknown anchor falls back to `baseline`. Show the fallback. The Python CLI must use the same rule. Never derive a supposed multi-item joint probability by multiplying pair probabilities.

Dayparts are morning before 12:00, midday 12:00–16:59, evening from 17:00, local transaction clock. The frontend must not recompute or fit model weights.

Codex entry point: `python src/basket_recommender.py train` generates data/evaluation. Query: `python src/basket_recommender.py recommend --branch Harbour --item "Butter Croissant" --daypart morning --cohort all`. Repeat `--item` for a larger basket; repeat `--unavailable` to exclude products. These commands are planned until the implementation milestone is pushed.

## Layout model and frontend contract

- Start with a configurable, one-dimensional row of display slots. Normalize distance by total display length. Equal-width slots are an explicit demo assumption, not measured geometry.
- Only physically eligible items can occupy slots. Ambient bakery and chilled dessert products must not be interchanged without verified storage constraints. Drinks made to order are not physical pastry slots; signs can be a separate future option.
- Compare at least **adjacent-pair**, **separated-pair**, and **random-order reference** candidates for the same products and slots. Preserve all products exactly once; respect fixed slots and incompatible zones if supplied. Do not claim a globally optimal layout from a local search.
- A weighted pair-distance score describes potential exposure under the user's mechanism. It is not sales, customer movement or dollars. Report both distance/exposure and the possible friction/service tradeoff. If pair lift is near one, say separation is based on popular co-purchases rather than a distinctive affinity.
- Retain current arrangement as a company-supplied pilot baseline; a random reference is not Stories' current layout. Do not fabricate current placement from the illustrative photo.
- Codex generates `reports/wejha_layouts.json` and local-script **`app/wejha_data.js`**, assigning **`window.STORIES_WEJHA`** with `schema_version`, `status`, `products`, `pairs`, `layouts`, `assumptions`, `pilot`. Each layout has `name`, `slots` (ordered product names), `weighted_pair_distance`, and `interpretation`. Pair rows have `item_a`, `item_b`, `pair_orders`, `lift`, `weight`, and `evidence_status`.
- UI can render these alternatives immediately. Manual reordering is optional; label it a scenario unless it is rescored by the documented backend formula. No fabricated monetary uplift beside a layout score.
- Planned command: `python src/wejha_layout.py` generates the default comparisons after the recommendation model exists. Cold-storage/shelf dimensions remain configurable future operational inputs.

## Money, experiment and presentation

Show an editable scenario: `eligible_orders × incremental_acceptance × add_on_unit_margin − discounts − incremental_waste − extra_labor − implementation_cost`. Incremental acceptance means the **difference versus the control**, not the observed acceptance rate or model accuracy. Default unknown effects to zero; an illustrative 1–5% setting is explicitly an assumption. Discounted prices must change margin and cannibalization must be investigated.

Test recommendations and layout separately before combining their claimed gains. Compare a simple popular-item offer with the selected model; compare current placement with the separated candidate in randomized branch-day blocks balanced by branch and weekday. Log eligible offers/exposures, assignments, offers shown, accepted items, total order revenue/cost, discounts, stock availability, waste, staff time, service time and overrides. Cluster inference at the randomization unit, account for repeat customers and carryover, and predefine duration/MDE using pilot variance. Three branches and a fixed small sample do not guarantee power.

The company-facing story is: **we identify an add-on worth testing, show how to display it, prepare enough stock, and measure incremental profit against a simpler alternative.** Do not promise that moving a croissant or cookie already increases sales. No claims about customer faith, branch religion or verified walking paths.

## Submission checklist

- README links to the usable offline demo, exact training/query commands, generated results and the three-person ownership table.
- Recommender and layout tests pass; unavailable/basket exclusions and split integrity are checked. Run existing inventory tests before publishing integrated changes.
- UI handles missing generated data with an honest pending message; no fake demo wins, invented acceptance or placeholder “actual” revenue.
- All finished code, artifacts, Markdown and dependency changes are committed and pushed. Check `git status` and remote state. Preserve raw data and teammate changes.
- If time expires, submit a functioning baseline-backed recommender and honest layout experiment with tested contracts. Mark extensions pending instead of claiming them complete.
