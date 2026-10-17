# Business review of the implemented Wejha system

The basket recommender, constrained display comparisons, preparation integration and offline interface run from committed artifacts. The commercial hypothesis is increased contribution per eligible order, with the preparation planner supporting availability and cost control.

The historical evidence does not establish that loyal customers account for most revenue or that a specific pastry pairing has exceptional affinity. The frozen six-training-date cohort supplies 3.10% of April–June revenue; the original notebook's broader full-period regularity definition is a different, retrospective denominator. All 28 training pastry pairs used in the default layout have lift below one, although some have frequent co-purchases. Separating those pairs is therefore an exposure hypothesis, not a discovered causal sales effect.

The recommendation model improves an offline recovered-item margin proxy by 2.51% versus margin-weighted popularity, while plain popularity retains better Hit@3. It cannot estimate purchases caused by an offer. The v2 prep model improves a conditional contribution simulation with lower realized fulfillment than the original. Neither result is measured additional company profit.

The money calculation and decision path are implemented, including zero-default incremental acceptance and a reporter for future observed order-level pilot outcomes. The company can now compare current/popular/model offers or current/separated layouts with recorded contribution, costs, availability and service; it must verify randomization and operational feasibility before interpreting contrasts causally.

Read [WEJHA_PILOT.md](../WEJHA_PILOT.md) for the exact experiment, [basket pilot CSV](../data/pilot/basket_pilot_template.csv) for the log, and `python src/basket_pilot_report.py <export.csv> --control popular` for recommendation-vs-popular evaluation. The submitted template contains no made-up pilot outcomes.
