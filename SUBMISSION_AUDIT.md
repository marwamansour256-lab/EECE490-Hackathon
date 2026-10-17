# Final submission audit against the EECE490 brief

**Team:** Charbel Manih (202502099), Omar Al Aghbari (202514349), Marwa Mansour (202505788).

**Submit this single public repository URL:** https://github.com/marwamansour256-lab/EECE490-Hackathon

The repository contains the deliverables requested in the brief. Start with the one-paragraph conclusion in [README.md](README.md), then [PROJECT_REPORT.md](PROJECT_REPORT.md) for the business case and evidence. Live profit improvement remains unmeasured; the brief asks for an estimated business case and practical test, which are supplied separately from observed results.

| Brief requirement | Evidence included |
|---|---|
| Useful business decision beyond a sales summary or database chatbot | Item preparation quantities, available basket add-ons, display alternatives and a cost-invoice investigation; [README](README.md), [integrated command](GROWTH_INTEGRATION.md) |
| Data quality and evidence | Preserved raw transactions, line accounting checks, repeated-line sensitivity and missing-ID treatment; [data notes](reports/analysis_details.md), [final saved-evidence check](reports/submission_readiness.json) |
| Challenge initial explanations and test alternatives | Rejected loyalty, usual-order, pair-affinity and promotion claims; placebo dates, geographic sensitivity, multiple-testing corrections; [report section 6](PROJECT_REPORT.md#6-alternative-explanations-tested-and-approaches-we-rejected), [external-context research](EXTERNAL_CONTEXT.md) |
| Compare a model with simple baselines | Frozen four-matching-weekday and seasonal-growth baselines; popularity and margin-popularity for basket completion; [evaluation](EVALUATION.md), [recommendation results](reports/recommendation_evaluation.md) |
| Test on data excluded from fitting | Jan–Mar training and inner selection, frozen Apr–Jun scoring; complete dates/orders retained. The team previously explored the full dataset, so this is not new independent confirmation; [protocol](EVALUATION.md) |
| Business value, costs and tradeoffs | Common-policy contribution, fulfilment, waste and unmet recorded demand; replenishment cost stress; Garden cost scenario; [report section 5](PROJECT_REPORT.md#5-business-value-measured-results-kept-separate-from-estimates) |
| Assumptions that could reverse the decision | Hidden stockouts, cost definitions, labour, lead times, storage, geography and attainable supplier costs; [report sections 7–8](PROJECT_REPORT.md#7-assumptions-and-how-they-could-change-the-recommendation), [business review](reports/wejha_business_review.md) |
| Correlation distinguished from cause | Event occurrence, exposure, predictive usefulness and causality separated. The 39-test screen and the small calendar traffic improvement are different experiments; [EXTERNAL_CONTEXT](EXTERNAL_CONTEXT.md) |
| Measured results distinguished from benefits | Error metrics are measured on recorded sales; preparation contribution and cost matching are simulations/scenarios; basket completion is a relevance proxy; no observed offer/layout uplift; [report value table](PROJECT_REPORT.md#5-business-value-measured-results-kept-separate-from-estimates) |
| Practical test before wider use | Shadow preparation, randomised branch-day comparisons, existing preparation log, separate order-level basket log, contribution reporter and service guardrails; [pilot](WEJHA_PILOT.md), [submission quickstart](SUBMISSION.md) |
| Choices and rejected approaches explained | Twelve reported forecast approaches; more complex models retained with results; [model card](MODEL_CARD.md), [project report](PROJECT_REPORT.md) |
| Clear team names, public URL, concise conclusion, inspectable evidence | [TEAM](TEAM.md), README conclusion, linked results and executable code. Public Git read was checked without a credential helper on October 17, 2026. |

## Final verification

- **Numerical suite previously recorded:** 44 tests plus four artifact checks and the pilot-template smoke check. The final audit checks the saved preparation model, relevant source, raw input and numerical output hashes against that run's [manifest](reports/strict_half_manifest.json). No model fitting, weights, inventory mathematics or raw sales were changed during finalisation.
- **Fresh saved-evidence audit:** `python src/verify_submission_readiness.py --output reports/submission_readiness.json` uses only Python's standard library. It checks raw accounting and the split, 49,140 prediction rows, all four finance totals, 8,801 recommendation test orders, generated browser/JSON equality, linked files and the empty pilot's honest status. It is an additional inspection tool, not a substitute for executing the numerical suite.
- **Fresh offline UI checks:** the actual `app/check_ui.js` was executed in the connected Node runtime. Page-script syntax, schemas, 72 ranking queries, complementary exclusions and 57 preparation cards passed. Current frontend rankings were also compared with all 144 committed Python parity cases across both offer modes; see [final UI check](reports/submission_ui_readiness.json). This is code-level verification; visual-review screenshots are in [reports/figures/ui](reports/figures/ui).
- **Reproduction:** use Python 3.11, run `python -m pip install -r requirements-lock.txt`, then `python src/verify_submission.py`. The locked versions match the saved models and recorded numerical run.

## Ninety-second inspection

1. Read the README conclusion and the measured/simulation/scenario table in PROJECT_REPORT.
2. Open `app/planner.html` and `app/wejha.html` locally; both use committed scripts and need no server.
3. Inspect `reports/strict_half_finance.csv`: v2's extra simulated contribution comes with a service tradeoff. The higher-profit frozen baseline fulfils fewer recorded units.
4. Inspect the offer/layout pilot. Unknown incremental acceptance defaults to zero; no company savings or placement uplift is asserted.

No presentation deck or deployed website is required by the supplied brief: it explicitly accepts any repository format. The outstanding company inputs and future pilot are disclosed limitations, with runnable logging/reporting tools already included.
