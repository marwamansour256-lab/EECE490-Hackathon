# Saved models

| File | What it is | Made by | Used by |
|---|---|---|---|
| **`strict_half_model.joblib`** | **Main model: the v2 preparation model.** Poisson regression forecasting units sold per branch, category and day, then split into products; trained on Jan 1–Mar 31, frozen, tested on Apr 1–Jun 30. Selected method `regularized_poisson_v2` (24.74% category error on Apr–Jun) | `python src/strict_model.py` | `src/predict_strict.py`, `src/growth_plan.py`, `app/preparation_data.js` |
| `basket_recommender.json` | Add-on recommender: which item to offer with the current basket (co-purchase counts, context and margin) | `python src/basket_recommender.py train` | `app/recommendation_data.js`, `app/wejha.html` |
| `context_v2_model.joblib`, `context_v2_orders_model.joblib`, `context_v2_mix_model.joblib` | Experimental models that add outside context (weather, calendar, internet) to the preparation model; kept for comparison | `src/context_model.py`, `src/context_mix_model.py` | `EXTERNAL_CONTEXT.md` results |

To use the main model:

```bash
python src/predict_strict.py --date 2026-06-30 --branch Harbour --product "Turkey & Cheese Sandwich" --remaining 0
```

Load `.joblib` files with the package versions in `requirements-lock.txt`. Details: [MODEL_CARD.md](../MODEL_CARD.md).
