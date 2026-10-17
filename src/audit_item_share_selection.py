"""Reproduce the v2 item-share grid using the inner training dates only.

This audit does not change model selection or read outer test outcomes. It makes
the teammate's 5-window x 4-smoothing search explicit and preserves every result.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from strict_model import daily_panel,fit_engine,features

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=pd.read_csv(ROOT/"data/stories_transactions.csv",parse_dates=["Date"])
    train=raw[raw.Date.le("2026-03-10")].copy()
    valid=raw[raw.Date.between("2026-03-11","2026-03-31")].copy()
    catalog=train[["Branch","Category","Product"]].drop_duplicates()
    cats=sorted(catalog.Category.unique())
    train_panel=daily_panel(train,catalog,pd.date_range("2026-01-01","2026-03-10"))
    valid_panel=daily_panel(valid,catalog,pd.date_range("2026-03-11","2026-03-31"))
    model=fit_engine(train_panel,.001,"2026-01-01",cats)
    means=model.predict(features(valid_panel,"2026-01-01",cats)[0])
    calibration=float(valid_panel.actual.sum()/means.sum())
    valid_cat=valid_panel[["Date","Branch","Category"]].assign(prediction=means*calibration)
    actual=valid.groupby(["Date","Branch","Product"]).Quantity.sum().rename("actual").reset_index()
    rows=[]
    for window in [14,28,42,56,None]:
        recent=train if window is None else train[train.Date.gt(train.Date.max()-pd.Timedelta(days=window))]
        counts=recent.groupby(["Branch","Category","Product"]).Quantity.sum().rename("units").reset_index()
        counts=catalog.merge(counts,on=["Branch","Category","Product"],how="left").fillna({"units":0})
        groups=counts.groupby(["Branch","Category"])
        for smoothing in [.5,1,3,10]:
            counts["share"]=(counts.units+smoothing)/(groups.units.transform("sum")+smoothing*groups.Product.transform("size"))
            panel=valid_cat.merge(counts,on=["Branch","Category"]).merge(actual,on=["Date","Branch","Product"],how="left").fillna({"actual":0})
            mu=np.maximum(panel.prediction*panel.share,1e-8)
            rows.append(dict(window_days="all" if window is None else window,smoothing=smoothing,
                sku_poisson_deviance=float(mean_poisson_deviance(panel.actual,mu)),
                sku_wape_pct=float(100*np.abs(panel.actual-mu).sum()/panel.actual.sum())))
    best=min(rows,key=lambda r:r["sku_poisson_deviance"])
    result=dict(fit_dates=["2026-01-01","2026-03-10"],validation_dates=["2026-03-11","2026-03-31"],
        outer_test_sales_used=False,alpha=.001,calibration_factor=calibration,
        candidates=rows,best_on_inner_validation=best,
        committed_choice=dict(window_days="all",smoothing=10),
        committed_choice_matches_audit=best["window_days"]=="all" and best["smoothing"]==10,
        caveat="Calibration itself is estimated on the inner validation window. This is model selection, not an unbiased validation of that calibration. Previously explored outer test still requires future confirmation.")
    (ROOT/"reports/strict_v2_share_selection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame(rows).to_csv(ROOT/"reports/strict_v2_share_selection.csv",index=False)
    print(json.dumps({"best":best,"committed_choice_matches":result["committed_choice_matches_audit"],"outer_test_sales_used":False},indent=2))


if __name__=="__main__":
    main()
