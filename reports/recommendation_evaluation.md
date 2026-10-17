# Basket recommendation evaluation

Selected on March training validation: **shrunk_pairs**, shrinkage 500.

Jan–Mar fitting; Apr–Jun frozen evaluation. Sales labels and economics are Stories-only.

| Method | Strength | Hit@3 | MRR | Recovered-item margin@3/order |
|---|---:|---:|---:|---:|
| popularity | 100 | 13.74% | 0.1503 | $0.4721 |
| margin_popularity | 100 | 12.90% | 0.1422 | $0.4956 |
| shrunk_pairs | 20 | 11.29% | 0.1335 | $0.4312 |
| shrunk_pairs | 100 | 12.42% | 0.1407 | $0.4726 |
| shrunk_pairs | 500 | 13.29% | 0.1451 | $0.5080 |

Test: 8,801 multi-product orders; 18,401 masked items. Each order has equal weight.

Recovered-item margin is an offline relevance proxy. It is not additional purchases, offer conversion, incremental profit or company savings.

Seven-day paired block intervals and cohort results are in recommendation_results.json. The test was previously explored by the team; future experimental confirmation is required.

Display separation and cross-selling require separate randomized pilots. A strong co-purchase can represent something customers already buy, creating no additional sale.
