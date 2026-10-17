"""Two-stage, service-constrained inventory decisions with an auditable replay.

Model: stochastic dynamic programming, not a claim that simulation is realized ROI.
At opening choose a batch; at 16:00 observe remaining stock and choose replenishment.
Both policies use identical demand forecasts, economics and numerical fill floor.
The two-stage floor applies separately to each stage; one-batch applies to the day.
No afternoon demand is observed when either decision is made. The replay assumes
instant replenishment and same-day carryover: verify these before live use.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson
from demand_model import panel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'


def expected_sold(mu, stock):
    stock=np.asarray(stock,dtype=float)
    return mu*poisson.cdf(stock-1,mu)+stock*poisson.sf(stock,mu)


def fill_floor_stock(mu, floor):
    if mu<=0 or floor<=0:return 0
    grid=np.arange(int(poisson.ppf(.999999,mu))+5)
    return int(grid[np.where(expected_sold(mu,grid)>=mu*floor-1e-10)[0][0]])


def optimize(mean,early_share,price,cost,setup=.25,premium=.10,salvage_fraction=0.,fill_floor=.90):
    if not (np.isfinite([mean,early_share,price,cost,setup,premium,salvage_fraction,fill_floor]).all() and mean>=0 and 0<=early_share<=1 and 0<=salvage_fraction<1 and 0<=fill_floor<1 and setup>=0 and premium>=0 and price>cost>0):
        raise ValueError('Invalid economic/forecast input')
    early,late=mean*early_share,mean*(1-early_share)
    salvage=salvage_fraction*cost
    cap=int(poisson.ppf(.999999,mean))+8
    grid=np.arange(cap+1)
    late_floor=fill_floor_stock(late,fill_floor)
    stock=grid[:,None]+grid[None,:]
    sales=expected_sold(late,stock)
    values=price*sales+salvage*(stock-sales)-(cost+premium)*grid[None,:]-setup*(grid[None,:]>0)
    values[stock<late_floor]=-np.inf
    late_orders=np.argmax(values,axis=1)
    late_value=values[grid,late_orders]
    egrid=np.arange(int(poisson.ppf(.9999999,early))+6)
    prob=poisson.pmf(egrid,early);prob[-1]+=max(0.,1-prob.sum())
    first=grid[grid>=fill_floor_stock(early,fill_floor)]
    inventory=np.maximum(first[:,None]-egrid[None,:],0)
    values=price*expected_sold(early,first)-cost*first+(late_value[inventory]*prob[None,:]).sum(axis=1)
    q0=int(first[np.argmax(values)])
    onegrid=grid[grid>=fill_floor_stock(mean,fill_floor)]
    sales=expected_sold(mean,onegrid)
    onevalue=price*sales+salvage*(onegrid-sales)-cost*onegrid
    oneq=int(onegrid[np.argmax(onevalue)])
    return dict(opening_units=q0,replenish_by_remaining_stock=late_orders.tolist(),one_batch_units=oneq,
        expected_two_stage_contribution=float(values.max()),expected_one_batch_contribution=float(onevalue.max()),
        mean_early=early,mean_late=late,price=price,cost=cost,setup=setup,premium=premium,salvage=salvage,fill_floor=fill_floor)


def realize(policy,early_demand,late_demand):
    if any(not np.isfinite(x) or x<0 or x!=int(x) for x in (early_demand,late_demand)):
        raise ValueError('Demand must be a nonnegative integer count')
    first=policy['opening_units'];sold_early=min(early_demand,first);left=first-sold_early
    topup=policy['replenish_by_remaining_stock'][int(left)]
    sold_late=min(late_demand,left+topup);unsold=left+topup-sold_late
    p,c=policy['price'],policy['cost']
    contribution=p*(sold_early+sold_late)+policy['salvage']*unsold-c*first-(c+policy['premium'])*topup-policy['setup']*(topup>0)
    oneq=policy['one_batch_units'];total=early_demand+late_demand
    oneprofit=p*min(total,oneq)+policy['salvage']*max(oneq-total,0)-c*oneq
    return dict(opening_units=first,replenished_units=topup,replenishment_executed=int(topup>0),sold_units=sold_early+sold_late,
        unsold_units=unsold,unmet_units=total-sold_early-sold_late,contribution=contribution,
        one_batch_units=oneq,one_batch_sold=min(total,oneq),one_batch_unsold=max(oneq-total,0),one_batch_unmet=max(total-oneq,0),one_batch_contribution=oneprofit,
        incremental_contribution=contribution-oneprofit)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--demo-date',default='2026-06-30')
    args=parser.parse_args()
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    raw['hour']=raw.Time.str[:2].astype(int)
    rec=pd.read_csv(OUT/'preparation_recommendations.csv',parse_dates=['Date'])
    sku=panel(raw,['Branch','Category','Product'])
    total=sku.groupby(['Date','Branch','Category']).history28.transform('sum')
    n=sku.groupby(['Date','Branch','Category']).Product.transform('size')
    sku['share']=(sku.history28+1)/(total+n)
    catpred=pd.read_csv(OUT/'model_predictions.csv',parse_dates=['Date'])
    sku=sku.merge(catpred[['Date','Branch','Category','four_weekdays']].rename(columns={'four_weekdays':'category_four_weekdays'}),on=['Date','Branch','Category'],how='left')
    sku['strong_baseline_mean']=sku.share*sku.category_four_weekdays
    rec=rec.merge(sku[['Date','Branch','Product','strong_baseline_mean']],on=['Date','Branch','Product'],validate='one_to_one')
    actual=raw.assign(part=np.where(raw.hour<16,'early','late')).groupby(['Date','Branch','Product','part']).Quantity.sum().unstack(fill_value=0).reset_index()
    rec=rec.merge(actual,on=['Date','Branch','Product'],how='left',validate='one_to_one').fillna({'early':0,'late':0})
    assert np.allclose(rec.early+rec.late,rec.actual)
    # All intraday shares are estimated BEFORE target date, pooled at branch/category.
    shares={}
    for date in sorted(rec.Date.unique()):
        history=raw[raw.Date.lt(date)&raw.Date.ge(date-pd.Timedelta(days=56))]
        for (b,cat),g in history.groupby(['Branch','Category']):
            prior=history.loc[history.Category.eq(cat)]
            global_share=prior.loc[prior.hour.lt(16),'Quantity'].sum()/prior.Quantity.sum()
            shares[(date,b,cat)]=(g.loc[g.hour.lt(16),'Quantity'].sum()+20*global_share)/(g.Quantity.sum()+20)
    rec['early_share']=[shares[(r.Date,r.Branch,r.Category)] for r in rec.itertuples()]
    configs=[dict(name='free_flexibility',setup=0.,premium=0.,salvage_fraction=0.,fill_floor=.90),
             dict(name='base_costed',setup=.25,premium=.10,salvage_fraction=0.,fill_floor=.90),
             dict(name='cost_stress',setup=1.,premium=.25,salvage_fraction=0.,fill_floor=.90),
             dict(name='high_cost',setup=3.,premium=.50,salvage_fraction=0.,fill_floor=.90),
             dict(name='half_cost_salvage',setup=.25,premium=.10,salvage_fraction=.50,fill_floor=.90),
             dict(name='no_service_floor',setup=.25,premium=.10,salvage_fraction=0.,fill_floor=0.)]
    details=[];demo=[]
    for config in configs:
        for model,col in [('internal','predicted_mean_units'),('strong_baseline','strong_baseline_mean')]:
            for row in rec.itertuples():
                policy=optimize(getattr(row,col),row.early_share,row.price,row.cost,**{k:v for k,v in config.items() if k!='name'})
                result=realize(policy,row.early,row.late)
                details.append(dict(Date=str(row.Date.date()),Branch=row.Branch,Category=row.Category,Product=row.Product,
                    scenario=config['name'],forecast=model,actual=row.actual,mean=getattr(row,col),early_share=row.early_share,
                    **result))
                if config['name']=='base_costed' and str(row.Date.date())==args.demo_date:
                    demo.append(dict(date=args.demo_date,forecast=model,branch=row.Branch,category=row.Category,product=row.Product,
                        forecast_mean=getattr(row,col),forecast_before16=policy['mean_early'],forecast_after16=policy['mean_late'],
                        opening_units=policy['opening_units'],price=row.price,cost=row.cost,
                        at_16_replenish_by_remaining_stock={str(k):v for k,v in enumerate(policy['replenish_by_remaining_stock']) if k<=policy['opening_units']},
                        status='Historical replay; updated production inputs and lead-time feasibility required'))
        print('Completed',config['name'],flush=True)
    detail=pd.DataFrame(details)
    detail.to_csv(OUT/'two_stage_replay.csv',index=False)
    measures=['actual','opening_units','replenished_units','replenishment_executed','sold_units','unsold_units','unmet_units','contribution',
        'one_batch_units','one_batch_sold','one_batch_unsold','one_batch_unmet','one_batch_contribution','incremental_contribution']
    summary=detail.groupby(['scenario','forecast'])[measures].sum().reset_index()
    summary['fill_rate_pct']=100*summary.sold_units/summary.actual
    summary['one_batch_fill_rate_pct']=100*summary.one_batch_sold/summary.actual
    summary['break_even_extra_branch_day_cost']=summary.incremental_contribution/90
    summary.to_csv(OUT/'two_stage_summary.csv',index=False)
    bybranch=detail.groupby(['scenario','forecast','Branch'])[measures].sum().reset_index()
    bybranch.to_csv(OUT/'two_stage_by_branch.csv',index=False)
    bycategory=detail.groupby(['scenario','forecast','Category'])[measures].sum().reset_index()
    bycategory.to_csv(OUT/'two_stage_by_category.csv',index=False)
    rng=np.random.default_rng(490);uncertainty=[]
    for (scenario,model),g in detail.groupby(['scenario','forecast']):
        deltas=g.groupby('Date').incremental_contribution.sum().to_numpy();n=len(deltas)
        starts=rng.integers(0,n,size=(2000,5))
        ix=(starts[:,:,None]+np.arange(7)[None,None,:])%n
        draws=deltas[ix.reshape(2000,-1)[:,:n]].sum(axis=1)
        uncertainty.append(dict(scenario=scenario,forecast=model,interval_95=[float(v) for v in np.quantile(draws,[.025,.975])]))
    # Deep inventory value should not be mislabeled as AI-only uplift.
    totals=summary.set_index(['scenario','forecast'])
    attribution=[]
    for cfg in configs:
        s=cfg['name'];a=totals.loc[(s,'internal')];b=totals.loc[(s,'strong_baseline')]
        attribution.append(dict(scenario=s,forecast_gain_at_one_batch=a.one_batch_contribution-b.one_batch_contribution,
            operational_flexibility_gain_using_baseline=b.incremental_contribution,
            forecast_gain_at_two_batches=a.contribution-b.contribution,
            combined_gain_over_baseline_one_batch=a.contribution-b.one_batch_contribution))
    result=dict(model='Two-stage stochastic dynamic program with Poisson stage demand and expected-fill constraints',
        replay_period='June 2026',decision_times=['before opening','16:00 after early sales'],configs=configs,
        summary=summary.to_dict('records'),gain_attribution=attribution,conditional_block_intervals=uncertainty,
        assumptions=['This new policy was developed after June had already been examined; June is a retrospective replay, not an untouched confirmation set.',
          'Sales are treated as uncensored demand. Preparation, stockouts, waste, lead time and the company current policy are unavailable.',
          'A second batch is instantly available at16:00; initial leftovers can carry to the late period; all remaining stock expires at close except stated salvage.',
          'Early and late demand are conditionally independent Poisson variables; only remaining stock, not unobserved lost sales, determines late action.',
          'Except the no_service_floor scenario, the constraint is 90% forecast EXPECTED unit fill in each stage for two-stage and over the whole day for one-batch. These differ in timing; report realized fill alongside contribution. This is not 90% stockout-free days or guaranteed actual fulfillment.',
          'Setup is incremental USD per SKU with a second batch; premium is incremental USD per second-batch unit. These are scenario inputs, not measured company costs.',
          'No total labor/capacity constraint is supplied. Break-even extra branch-day overhead is reported; no unqualified rollout recommendation.',
          'Both forecasts get the same decision rules and costs; report how much value comes from operational flexibility versus forecasting.',
          'External events require verified exposure and known_at timestamps before numerical adjustment; unresolved events are review alerts only.'])
    (OUT/'two_stage_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'decision_cards.json').write_text(json.dumps(demo,indent=2,allow_nan=False),encoding='utf-8')
    print(summary.to_string(index=False))
    print(json.dumps(attribution,indent=2))


if __name__=='__main__':main()
