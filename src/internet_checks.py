"""Audit national reachability without treating missing data as an outage.

This is an additional exploratory measurement/access hypothesis family. The
signal is reachable probed network blocks, not customer internet usage or POS
status. Calendar/war adjusted associations are not causal or day-ahead features.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from context_associations import outcomes,fit

ROOT=Path(__file__).resolve().parents[1]


def parse_signal(payload):
    if payload.get('error'):raise ValueError(str(payload['error']))
    entries=[]
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict) and 'values' in value:entries.append(value)
    visit(payload.get('data',[]))
    frames=[]
    for s in entries:
        if s.get('entityCode')!='LB' or s.get('datasource')!='ping-slash24':continue
        frames.append(pd.DataFrame({'timestamp':pd.to_datetime(s['from']+np.arange(len(s['values']))*s['step'],unit='s',utc=True),
                                    'reachable_blocks':pd.to_numeric(pd.Series(s['values']),errors='coerce')}))
    if not frames:raise ValueError('No country-LB active-probing signal')
    return pd.concat(frames,ignore_index=True)


def main():
    cache=ROOT/'data/external/context_v2'
    frames=[parse_signal(json.loads(p.read_text(encoding='utf-8'))) for p in sorted(cache.glob('internet_LB_2026_*.json'))]
    data=pd.concat(frames,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp')
    data['local']=data.timestamp.dt.tz_convert('Asia/Beirut')
    data['Date']=data.local.dt.tz_localize(None).dt.normalize()
    daily=data.groupby('Date').agg(reachability_mean=('reachable_blocks','mean'),reachability_min=('reachable_blocks','min'),
                                  observed_hours=('reachable_blocks','count'),sample_hours=('reachable_blocks','size'))
    day_edges=pd.date_range('2026-01-01','2026-07-01',freq='D',tz='Asia/Beirut',nonexistent='shift_forward')
    dates=day_edges[:-1].tz_localize(None).normalize()
    daily=daily.reindex(dates)
    daily['expected_hours']=day_edges.to_series().diff().dt.total_seconds().iloc[1:].to_numpy()/3600
    daily['coverage_fraction']=daily.observed_hours/daily.expected_hours
    prior=daily.reachability_mean.shift(1).rolling(7,min_periods=5).median()
    daily['reachability_change_vs7d_pct']=100*(daily.reachability_mean/prior-1)
    daily['reachability_min_drop_proxy_pct']=100*np.maximum(0,1-daily.reachability_min/prior)
    daily=daily.rename_axis('Date').reset_index()
    daily.to_csv(cache/'internet_daily.csv',index=False)
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    ctx=pd.read_csv(cache/'daily_features.csv',parse_dates=['Date'])
    ctx=ctx[ctx.region.eq('beirut')].merge(daily,on='Date',validate='one_to_one').sort_values('Date').reset_index(drop=True)
    ys=outcomes(raw,ctx)
    scopes=['network_orders','network_units','dine_in_order_share_pp','basket_units_per_order','morning_orders','evening_orders',
            'branch_units / Garden','branch_units / Harbour','branch_units / Uptown']
    rows=[]
    for scope in scopes:
        y,unit=ys[scope]
        for feature in ['reachability_mean','reachability_change_vs7d_pct','reachability_min_drop_proxy_pct']:
            r=fit(y,ctx,feature)
            if r:rows.append(dict(outcome=scope,feature=feature,outcome_unit=unit,**r))
    result=pd.DataFrame(rows)
    if len(result):result['q_bh_within_internet_family']=multipletests(result.p_value,method='fdr_bh')[1]
    main=pd.read_csv(ROOT/'reports/context_v2_associations.csv')
    main['search_family']='core_mechanisms'
    result['search_family']='internet_secondary'
    combined=pd.concat([main,result],ignore_index=True)
    valid=combined.p_value.notna()&np.isfinite(combined.p_value)
    combined.loc[valid,'q_global_bh']=multipletests(combined.loc[valid,'p_value'],method='fdr_bh')[1]
    combined.loc[valid,'q_global_by']=multipletests(combined.loc[valid,'p_value'],method='fdr_by')[1]
    combined.to_csv(ROOT/'reports/context_v2_all_associations.csv',index=False)
    joint=combined[combined.search_family.eq('internet_secondary')][['outcome','feature','q_global_bh','q_global_by']]
    result=result.merge(joint,on=['outcome','feature'],validate='one_to_one')
    result.to_csv(ROOT/'reports/context_v2_internet_associations.csv',index=False)
    report=dict(source='Georgia Tech IODA national active-probing reachability',documentation='https://api.ioda.inetintel.cc.gatech.edu/v2/',
        samples=len(data),date_count=len(daily),minimum_coverage_fraction=float(daily.coverage_fraction.min()),
        days_below90pct_coverage=int(daily.coverage_fraction.lt(.90).sum()),
        exploratory_tests=len(result),bh_flags_within_this_secondary_family=int(result.q_bh_within_internet_family.lt(.05).sum()) if len(result) else 0,
        tests_in_global_union=len(combined),eligible_tests_in_global_union=int(valid.sum()),
        internet_flags_global_bh=int(result.q_global_bh.lt(.05).sum()),internet_flags_global_by=int(result.q_global_by.lt(.05).sum()),
        top_rows=result.nsmallest(5,'p_value').replace({np.nan:None}).to_dict('records'),
        caveats=['Secondary exploratory family, not independent validation. Global BH/BY over the union with the main2310 rows is also reported; within-family q values are not the only screen.',
            'Reachability is a measurement proxy, not proof of a country-wide outage, a particular store POS failure or lost demand.',
            'Missing values remain missing; sample coverage is separate. Jan1 has only22 local hourly samples because retrieval starts UTCJan1.',
            'The past7-day comparison needs antecedent coverage; those initial missing feature values are not imputed as disruptions.',
            'No signal is used as a claimed issued day-ahead predictor.'])
    (ROOT/'reports/context_v2_internet_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
