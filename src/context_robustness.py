"""Targeted challenges to the strongest new exploratory external associations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from context_associations import outcomes,fit

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    allctx=pd.read_csv(ROOT/'data/external/context_v2/daily_features.csv',parse_dates=['Date'])
    cases=[('beirut','network_orders','month_edge_hypothesis'),
           ('beirut','meat_to_tuna_log_ratio','triple_calendar_overlap'),
           ('beirut','evening_order_share_pp','rain_forecast_surprise'),
           ('batroun','sunset_after_share_of_4h_window_pp','rain_sum')]
    rows=[]
    for region,outcome,feature in cases:
        ctx=allctx[allctx.region.eq(region)].sort_values('Date').reset_index(drop=True)
        ys=outcomes(raw,ctx)
        y,_=ys[outcome]
        for month in range(1,7):
            keep=ctx.Date.dt.month.ne(month)
            sub=ctx[keep].reset_index(drop=True)
            result=fit(y,sub,feature)
            if result:rows.append(dict(region=region,outcome=outcome,feature=feature,omitted_month=month,**result))
    # Explicit menu-count sensitivity: alternative smoothing and a binomial count
    # model avoid interpreting one log-ratio pseudocount as a definitive finding.
    ctx=allctx[allctx.region.eq('beirut')].sort_values('Date').reset_index(drop=True)
    dates=pd.DatetimeIndex(ctx.Date)
    meat=raw[raw.Product.isin(['Chicken Pesto Sandwich','Chicken Salad Bowl','Turkey & Cheese Sandwich'])].groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
    tuna=raw[raw.Product.eq('Tuna Sandwich')].groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
    menu=[]
    for smooth in [.25,.5,1.,2.]:
        result=fit(np.log((meat+smooth)/(tuna+smooth)),ctx,'triple_calendar_overlap')
        if result:menu.append(dict(specification='log_ratio_smoothing_'+str(smooth),**result))
    # Count logit includes the same nuisance factors, now without pseudocounts.
    x=pd.DataFrame({'const':1.,'trend':np.arange(len(dates))/30},index=dates)
    x['trend2']=x.trend**2
    for day in range(1,7):x['weekday_'+str(day)]=(dates.dayofweek==day).astype(float)
    for col in ['ramadan','maronite_lent','orthodox_lent','muslim_holiday','christian_holiday','escalation_known','promotion_harbour','triple_calendar_overlap']:
        x[col]=ctx[col].to_numpy()
    active=(meat+tuna).gt(0)
    model=sm.GLM(np.column_stack([meat[active],tuna[active]]),x.loc[active],family=sm.families.Binomial()).fit(cov_type='HAC',cov_kwds={'maxlags':7})
    beta=float(model.params['triple_calendar_overlap'])
    menu.append(dict(specification='binomial_meat_vs_tuna_count',log_odds_coefficient=beta,odds_ratio=float(np.exp(beta)),
                     p_value=float(model.pvalues['triple_calendar_overlap']),status='post_selection_sensitivity_not_new_validation'))
    pd.DataFrame(rows).to_csv(ROOT/'reports/context_v2_leave_month_out.csv',index=False)
    report=dict(leave_month_out=rows,menu_substitution_sensitivity=menu,
        caveats=['These are selected follow-up analyses on the same sample, not independent confirmation.',
          'Dropping a month can eliminate an event or alter confounding; missing estimates are not evidence of robustness.',
          'Calendar groups describe possible behavior, not the faith or observance of any customer.',
          'Rain forecast surprise based on reanalysis is only retrospectively observed, not an input known before opening.'])
    (ROOT/'reports/context_v2_robustness.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(rows)[['outcome','feature','omitted_month','coefficient_per_sd','p_value']].to_string(index=False))
    print(json.dumps(menu,indent=2))


if __name__=='__main__':main()
