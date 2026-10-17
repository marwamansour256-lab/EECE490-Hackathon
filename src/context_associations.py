"""Mechanism tests on daily totals, demand timing, menu substitution and format.

Exploratory associations, not causes. Daily series receive weekday/trend/holiday
and conflict controls, HAC7 uncertainty and global BH/BY search adjustment.
Candidate dates with <8 exposed days are descriptive only. Three weather regions
are sensitivity scenarios, never inferred branch coordinates.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from context_features import CALENDAR_FORECAST, OBSERVED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
CALENDAR_TESTS=['ramadan','maronite_noon_fast','maronite_abstinence_calendar','orthodox_lent','fasting_overlap',
 'triple_calendar_overlap','friday_abstinence_outside_easter','apostles_abstinence','month_edge_hypothesis',
 'month_position_sin','month_position_cos','bridge_day','school_remote_transition_7d','muslim_holiday','christian_holiday']
WEATHER_TESTS=OBSERVED+['coast_mountain_comfort_gap','weekend_comfort_gap','temperature_change_1d','rain_afternoon']


def outcomes(raw,ctx):
    dates=pd.DatetimeIndex(sorted(raw.Date.unique()))
    units=raw.groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
    orders=raw.groupby('Transaction ID').agg(Date=('Date','first'),Branch=('Branch','first'),Time=('Time','first'),
        order_type=('Order Type','first'),revenue=('Line Revenue (USD)','sum'),profit=('Line Gross Profit (USD)','sum'))
    orders['hour']=orders.Time.str[:2].astype(int)
    orders['timestamp']=pd.to_datetime(orders.Date.dt.strftime('%Y-%m-%d')+' '+orders.Time)
    orders['minute']=orders.timestamp.dt.hour*60+orders.timestamp.dt.minute+orders.timestamp.dt.second/60
    count=orders.groupby('Date').size().reindex(dates,fill_value=0)
    data={'network_units':(units,'units/day'),'network_orders':(count,'orders/day'),
          'basket_units_per_order':(units/count,'units/order')}
    dine=orders[orders.order_type.eq('Dine-in')].groupby('Date').size().reindex(dates,fill_value=0)
    data['dine_in_order_share_pp']=(100*dine/count,'percentage points')
    for label,lo,hi in [('morning',8,12),('afternoon',12,16),('evening',18,24)]:
        q=orders[orders.hour.ge(lo)&orders.hour.lt(hi)].groupby('Date').size().reindex(dates,fill_value=0)
        data[label+'_orders']=(q,'orders/day')
        data[label+'_order_share_pp']=(100*q/count,'percentage points')
    sunset=ctx.set_index('Date').sunset_minute
    orders['sunset_minute']=orders.Date.map(sunset)
    delta=orders.minute-orders.sunset_minute
    before=orders[delta.between(-120,0,inclusive='left')].groupby('Date').size().reindex(dates,fill_value=0)
    after=orders[delta.between(0,120,inclusive='left')].groupby('Date').size().reindex(dates,fill_value=0)
    data['sunset_after_share_of_4h_window_pp']=(100*after/(before+after).replace(0,np.nan),'percentage points')
    data['sunset_after_minus_before_orders']=(after-before,'orders/day')
    def product_units(products):return raw[raw.Product.isin(products)].groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
    meat=product_units(['Chicken Pesto Sandwich','Chicken Salad Bowl','Turkey & Cheese Sandwich'])
    tuna=product_units(['Tuna Sandwich'])
    data['meat_to_tuna_log_ratio']=(np.log((meat+.5)/(tuna+.5)),'log unit ratio, +0.5 smoothing')
    dairy_named=product_units(['Halloumi Wrap','Egg Sandwich','Granola Yoghurt','Frozen Yoghurt Cup'])
    plant_named=product_units(['Quinoa Bowl','Avocado Toast'])
    data['named_dairy_egg_to_plant_log_ratio']=(np.log((dairy_named+.5)/(plant_named+.5)),'log unit ratio; menu-name proxy, recipes unverified')
    drinks=raw[raw.Category.isin(['Cold Coffee','Cold Drinks','Hot Coffee','Matcha & Tea'])]
    cold=drinks[drinks.Category.isin(['Cold Coffee','Cold Drinks'])].groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
    all_drinks=drinks.groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
    data['cold_category_drink_share_pp']=(100*cold/all_drinks,'percentage points; excludes ambiguous matcha temperatures')
    for category,g in raw.groupby('Category'):
        y=g.groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
        data['category_units / '+category]=(y,'units/day')
    for branch,g in raw.groupby('Branch'):
        y=g.groupby('Date').Quantity.sum().reindex(dates,fill_value=0)
        data['branch_units / '+branch]=(y,'units/day')
        data['branch_share_pp / '+branch]=(100*y/units,'percentage points')
    return data


def fit(y,ctx,factor,linear_only=False,lead_days=0):
    dates=pd.DatetimeIndex(ctx.Date)
    x=pd.DataFrame(index=dates)
    for k in range(1,7):x['weekday_'+str(k)]=(dates.dayofweek==k).astype(float)
    x['trend']=(dates-pd.Timestamp('2026-01-01')).days/30
    if not linear_only:x['trend2']=x.trend**2
    for name in ['ramadan','muslim_holiday','christian_holiday','escalation_known','promotion_harbour']:
        if name!=factor:x[name]=ctx[name].to_numpy()
    # Overlap terms need parent fast calendars as controls to test extra overlap.
    if factor in ['fasting_overlap','triple_calendar_overlap']:
        x['maronite_lent']=ctx.maronite_lent.to_numpy()
    if factor=='triple_calendar_overlap':x['orthodox_lent']=ctx.orthodox_lent.to_numpy()
    focal=pd.Series(ctx[factor].to_numpy(),index=dates).shift(-lead_days)
    sd=float(focal.std())
    if sd<1e-9:return None
    x['factor_per_sd']=(focal-focal.mean())/sd
    x=sm.add_constant(x,has_constant='add')
    joined=x.join(y.rename('outcome')).dropna()
    xx=joined.drop(columns='outcome')
    if np.linalg.matrix_rank(xx.to_numpy())<xx.shape[1]:return None
    model=sm.OLS(joined.outcome,xx).fit(cov_type='HAC',cov_kwds={'maxlags':7})
    ci=model.conf_int().loc['factor_per_sd']
    return dict(coefficient_per_sd=float(model.params['factor_per_sd']),ci_low=float(ci.iloc[0]),ci_high=float(ci.iloc[1]),
                p_value=float(model.pvalues['factor_per_sd']),feature_sd=sd,n_days=len(joined))


def main():
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    all_ctx=pd.read_csv(ROOT/'data/external/context_v2/daily_features.csv',parse_dates=['Date'])
    rows=[];lookups={}
    for region,ctx in all_ctx.groupby('region',sort=False):
        ctx=ctx.sort_values('Date').reset_index(drop=True)
        ys=outcomes(raw,ctx)
        tests=WEATHER_TESTS+(CALENDAR_TESTS if region=='beirut' else [])
        for scope,(y,unit) in ys.items():
            for factor in tests:
                fit_result=fit(y,ctx,factor)
                if fit_result is None:continue
                exposed=int(ctx[factor].ne(0).sum())
                sparse=(ctx[factor].nunique()<=2 and min(exposed,len(ctx)-exposed)<8) or factor in ['apostles_abstinence','school_remote_transition_7d']
                row=dict(region=region,outcome=scope,feature=factor,outcome_unit=unit,exposed_days=exposed,
                    status='descriptive_sparse' if sparse else 'exploratory_association',**fit_result)
                if sparse:row.update(p_value=np.nan,ci_low=np.nan,ci_high=np.nan)
                rows.append(row);lookups[(region,scope,factor)]=(y,ctx)
    result=pd.DataFrame(rows)
    valid=result.p_value.notna()&np.isfinite(result.p_value)
    result.loc[valid,'q_bh']=multipletests(result.loc[valid,'p_value'],method='fdr_bh')[1]
    result.loc[valid,'q_by']=multipletests(result.loc[valid,'p_value'],method='fdr_by')[1]
    result=result.sort_values('p_value')
    result.to_csv(OUT/'context_v2_associations.csv',index=False)
    sensitivity=[]
    for row in result.head(25).itertuples():
        y,ctx=lookups[(row.region,row.outcome,row.feature)]
        for label,kwargs in [('linear_trend',dict(linear_only=True)),('future7_negative_control',dict(lead_days=7))]:
            value=fit(y,ctx,row.feature,**kwargs)
            if value:sensitivity.append(dict(region=row.region,outcome=row.outcome,feature=row.feature,specification=label,**value))
    pd.DataFrame(sensitivity).to_csv(OUT/'context_v2_association_sensitivity.csv',index=False)
    flagged=result[result.q_bh.lt(.05)].replace({np.nan:None})
    report=dict(tests=len(result),eligible_tests=int(valid.sum()),bh_flags=len(flagged),by_flags=int(result.q_by.lt(.05).sum()),
        bh_flagged_rows=flagged.to_dict('records'),top_exploratory=result.head(15).replace({np.nan:None}).to_dict('records'),
        mechanisms=['Clock-relative Ramadan sunset demand','Noon-relative Maronite fasting','Overlapping fasting calendars',
            'Menu-name meat/fish and dairy/plant substitution','Air/dust and dine-in/takeaway mix','Thermal novelty rather than absolute temperature',
            'Rain timing, first rain after dry spell and next-day catch-up','Weather forecast surprise','Coast/mountain comfort gradient on weekends',
            'Month-position liquidity hypothesis','School remote-learning transition versus generic war dummy','Branch substitution and basket size'],
        caveats=['Every test is exploratory and selected after the dataset had been examined. BH/BY does not make associations causal.',
            'Daily outcomes, not product lines, are the statistical observations; HAC7 handles some time dependence.',
            'Weather and air are geographic proxies. Model-based air-quality data is not a ground monitor.',
            'Public fasting rules and menu names do not establish customer religion, actual fasting or complete ingredient content.',
            'Temporal overlap, trend specification, unobserved operating hours and stockouts may explain apparent relationships.',
            'Future7 is a negative-control warning, not a deployable predictor; seasonality/autocorrelation can produce apparent associations.',
            'No significant association is not proof of no effect; six months has limited identifiable independent episodes.'])
    (OUT/'context_v2_association_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['tests','eligible_tests','bh_flags','by_flags']},indent=2),flush=True)
    print(result[['region','outcome','feature','coefficient_per_sd','p_value','q_bh','q_by']].head(15).to_string(index=False),flush=True)


if __name__=='__main__':main()
