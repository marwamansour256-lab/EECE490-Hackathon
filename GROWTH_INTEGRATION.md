# Integrated offer → preparation → profit scenario

The recommender/layout engine and existing preparation model are published. `src/growth_plan.py` now connects them without retraining or adding assumed offer uplift to the demand forecast.

```bash
python src/growth_plan.py --date 2026-06-30 --branch Harbour --item Espresso --daypart morning --output reports/growth_demo.json
python src/growth_plan.py --export-preparation
python src/basket_recommender.py recommend --branch Harbour --item Espresso --daypart morning --mode complementary
```

The integrated command defaults to `complementary`: after drinks, offer food; after food, offer drinks; with an empty or mixed food/drink basket, leave the candidate pool unrestricted. This is an explicit optional business rule, not a learned dietary preference. The published 13.29% Hit@3 belongs to the unrestricted model and must not be relabeled as the filtered rule's performance. The model weights are unchanged. Current basket and unavailability exclusions still apply.

`growth_demo.json` contains `offers`, one `preparation` card per offer, `profit_scenario_for_top_offer` and `retrospective_diagnostic_not_used_to_select_offer_or_prepare`. Made-to-order/nonperishable categories receive demand references, not pastry batch instructions. Preparation quantities are from the frozen inventory-control model; no invented conversion uplift enters the forecast.

The profit scenario defaults incremental acceptance to **zero**. An explicit assumption can be explored with `--eligible-orders 1000 --incremental-acceptance 0.03 --labor 20 --waste 10 --implementation 10`. The model does not estimate this 3%. No automatic discount is applied; optional `--discount` means discount on each extra add-on only and does not capture discounts leaked to purchases that would have happened anyway. Cannibalization, such leakage and stock constraints need measurement before interpreting a positive scenario as attainable.

## UI integration handoff to Marwa

Keep ownership of `app/wejha.html`. Two optional additions can use the new backend without changing the existing generated recommendation/layout schema:

1. Add a complementary/unrestricted selector. Complementary mode excludes products in the basket's sole food/drink group before the existing max-score/ranking step. Drink categories are `Hot Coffee`, `Cold Coffee`, `Cold Drinks`, `Matcha & Tea`; all other supplied categories are food. With mixed groups or no basket, exclude neither group. Preserve the unrestricted evaluation label.
2. Load local `preparation_data.js`, assigning **`window.STORIES_PREPARATION`**. Its `date` is 2026-06-30 and `cards` is keyed by `branch|product`. For a food recommendation, show its `forecast_mean_units`, `opening_prepare_units`, and `at16_topup_by_remaining_stock` using the existing card behavior. Only committed historical cards exist; an absent card means no batch instruction, not zero demand. Preserve its status and cost/instant-replenishment/capacity assumptions. This export is separate from `window.STORIES_RECOMMENDATIONS` and does not change old preparation/pilot contracts.

Do not represent the retrospective diagnostic as an input to a pre-opening forecast. The integration is usable from the CLI even before optional UI enhancements land.
