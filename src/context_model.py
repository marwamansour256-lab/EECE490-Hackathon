"""Externally conditioned Stories demand, with an internal control and honest gate.

All targets are Stories sales. First90 dates train; final21 training dates select
context family/regularization. April-June evaluates frozen coefficients. Archived
24-hour-lead weather is available separately for each target day's forecast;
this is a day-ahead context experiment, not a March31 forecast of future weather.
Observed-weather/air variants are explicitly retrospective oracle diagnostics.
"""
from pathlib import Path
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from strict_model import daily_panel, fit_engine, features, metric
from context_features import CALENDAR_FORECAST, ISSUED_FORECAST, OBSERVED, REGIONS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
ALPHAS=[.01,.1,1.,10.]
FAMILIES={'calendar':CALENDAR_FORECAST,
          'month_cycle':['month_position_sin','month_position_cos','month_edge_hypothesis'],
          'religious_menu':['ramadan','maronite_noon_fast','maronite_abstinence_calendar','orthodox_lent','muslim_holiday','christian_holiday'],
          'issued_weather_only':ISSUED_FORECAST,
          'public_disruption':['escalation_known','school_remote_transition_7d'],
          'calendar_issued_weather':CALENDAR_FORECAST+ISSUED_FORECAST,
          'observed_oracle':CALENDAR_FORECAST+OBSERVED}
INTERACTIONS=['ramadan','maronite_noon_fast','maronite_abstinence_calendar','orthodox_lent',
 'muslim_holiday','christian_holiday','forecast_temp_max','forecast_rain_sum','heat_anomaly_7d','dust_mean']


def group_for(name):
    if name.startswith(('forecast_',)):return 'issued_weather'
    if name in ['ramadan','maronite_noon_fast','maronite_abstinence_calendar','orthodox_lent','fasting_overlap','muslim_holiday','christian_holiday']:return 'religious_calendar'
    if name.startswith('month_'):return 'month_position_hypothesis'
    if name in ['civic_holiday','bridge_day']:return 'civic_calendar'
    if name=='daylight_hours':return 'daylight'
    if name in ['escalation_known','school_remote_transition_7d']:return 'public_disruption_announcements'
    if any(word in name for word in ['dust','pm25','aerosol']):return 'observed_air_quality'
    return 'observed_weather'


def context_design(panel,context,names,categories):
    merged=panel[['Date','Branch','Category']].merge(context[['Date']+names],on='Date',validate='many_to_one')
    columns={};groups={}
    for name in names:
        columns[name]=merged[name].to_numpy()
        groups[name]=group_for(name)
        if name in INTERACTIONS:
            for cat in categories:
                key=name+' / '+cat
                columns[key]=merged[name].to_numpy()*merged.Category.eq(cat).to_numpy()
                groups[key]=group_for(name)
    # Promotion adjustment is specific to observed branch/category exposure.
    if 'promotion_harbour' in names:
        promo=panel[['Date','Branch','Category']].merge(context[['Date','promotion_harbour']],on='Date',validate='many_to_one')
        columns['observed_promotion']=promo.promotion_harbour.to_numpy()*promo.Branch.eq('Harbour').to_numpy()*promo.Category.isin(['Cold Coffee','Desserts']).to_numpy()
        groups['observed_promotion']='recorded_promotion_schedule'
    return pd.DataFrame(columns,index=panel.index),groups


def fit_context(x,y,base,alpha):
    pipeline=make_pipeline(StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=2000,tol=1e-7))
    # Weighted ratio-target Poisson loss is equivalent, up to constants/scaling,
    # to a log-link Poisson loss with log(base) as a fixed offset.
    pipeline.fit(x,np.asarray(y)/np.maximum(base,1e-8),poissonregressor__sample_weight=np.maximum(base,1e-8))
    return pipeline


def decompose(model,x,base,groups):
    scaler,reg=model.steps[0][1],model.steps[1][1]
    z=scaler.transform(x)
    pieces=z*reg.coef_[None,:]
    eta=pieces.sum(axis=1)
    anchor=base*np.exp(reg.intercept_)
    ratio=np.ones(len(eta))
    nonzero=np.abs(eta)>1e-9
    ratio[nonzero]=np.expm1(eta[nonzero])/eta[nonzero]
    result={'calibration':anchor-base}
    for group in set(groups.values()):
        indices=[i for i,col in enumerate(x.columns) if groups[col]==group]
        result[group]=anchor*ratio*pieces[:,indices].sum(axis=1)
    full=base*model.predict(x)
    np.testing.assert_allclose(base+sum(result.values()),full,rtol=1e-8,atol=1e-7)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',choices=['units','orders'],default='units')
    args=parser.parse_args()
    prefix='context_v2' if args.target=='units' else 'context_v2_orders'
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    prior=joblib.load(ROOT/'models/strict_half_model.joblib')
    if args.target=='orders':
        raw=raw.groupby('Transaction ID').agg(Date=('Date','first'),Branch=('Branch','first')).reset_index()
        raw['Category']='Orders';raw['Product']='Order';raw['Quantity']=1
        prior=dict(prior)
        prior['categories']=['Orders']
        prior['catalog']=raw[['Branch','Category','Product']].drop_duplicates()
    context_all=pd.read_csv(ROOT/'data/external/context_v2/daily_features.csv',parse_dates=['Date'])
    dates=pd.date_range(raw.Date.min(),raw.Date.max())
    frame=daily_panel(raw,prior['catalog'],dates)
    train=frame.Date.le('2026-03-31')
    inner=frame.Date.lt('2026-03-11')
    valid=frame.Date.between('2026-03-11','2026-03-31')
    test=frame.Date.ge('2026-04-01')
    inner_model=fit_engine(frame[inner],prior['selected_alpha'],prior['start'],prior['categories'])
    inner_base=inner_model.predict(features(frame,prior['start'],prior['categories'])[0])
    if args.target=='orders':
        prior['model']=fit_engine(frame[train],prior['selected_alpha'],prior['start'],prior['categories'])
    full_base=prior['model'].predict(features(frame,prior['start'],prior['categories'])[0])
    baseline_validation=metric(frame.loc[valid,'actual'],inner_base[valid])
    selection_rows=[];predictions=[];models={};decompositions=[];metrics=[]
    baseline_test=metric(frame.loc[test,'actual'],full_base[test])
    for region in REGIONS:
        ctx=context_all[context_all.region.eq(region)].copy()
        candidates=[]
        for family,names in FAMILIES.items():
            x,groups=context_design(frame,ctx,names,prior['categories'])
            keep=list(x.loc[train].columns[x.loc[train].nunique().gt(1)])
            # Constant initial-fit features receive zero coefficients there. A
            # final refit may learn from events present only in the last21 TRAIN
            # dates; nothing is selected from test-period feature variation.
            x=x[keep];groups={k:groups[k] for k in keep}
            tuning=[]
            for alpha in ALPHAS:
                fitted=fit_context(x.loc[inner],frame.loc[inner,'actual'],inner_base[inner],alpha)
                pred=inner_base[valid]*fitted.predict(x.loc[valid])
                row=dict(region=region,family=family,alpha=alpha,**metric(frame.loc[valid,'actual'],pred))
                selection_rows.append(row);tuning.append(row)
            best=min(tuning,key=lambda r:r['poisson_deviance'])
            fitted=fit_context(x.loc[train],frame.loc[train,'actual'],full_base[train],best['alpha'])
            prediction=full_base*fitted.predict(x)
            name=region+' / '+family
            models[name]=dict(model=fitted,columns=keep,groups=groups,family=family,region=region)
            candidates.append(best)
            metrics.append(dict(method=name,eligible_day_ahead=family!='observed_oracle',**metric(frame.loc[test,'actual'],prediction[test])))
            for r,p,b in zip(frame.itertuples(),prediction,full_base):
                predictions.append(dict(Date=str(r.Date.date()),Branch=r.Branch,Category=r.Category,actual=r.actual,
                    baseline_mean=b,context_mean=p,method=name,period='test' if r.Date>=pd.Timestamp('2026-04-01') else 'training_in_sample'))
            contributions=decompose(fitted,x,full_base,groups)
            for group,values in contributions.items():
                decompositions.extend(dict(Date=str(r.Date.date()),Branch=r.Branch,Category=r.Category,method=name,
                    group=group,contribution_units=float(value)) for r,value in zip(frame.itertuples(),values))
        allowed=[r for r in candidates if r['family']!='observed_oracle']
        winner=min(allowed,key=lambda r:r['poisson_deviance'])
        selected='internal_baseline' if baseline_validation['poisson_deviance']<=winner['poisson_deviance'] else region+' / '+winner['family']
        print(region,'validation-selected:',selected,flush=True)
        models['selection_'+region]=dict(selected_method=selected,best_external=region+' / '+winner['family'],validation=winner)
    metrics.append(dict(method='internal_baseline',eligible_day_ahead=True,**baseline_test))
    pd.DataFrame(selection_rows).to_csv(OUT/(prefix+'_validation.csv'),index=False)
    pd.DataFrame(metrics).to_csv(OUT/(prefix+'_metrics.csv'),index=False)
    pd.DataFrame(predictions).to_csv(OUT/(prefix+'_predictions.csv'),index=False)
    pd.DataFrame(decompositions).to_csv(OUT/(prefix+'_decomposition.csv'),index=False)
    artifact=dict(base_artifact=prior,models=models,categories=prior['categories'],start=prior['start'],
        training_end='2026-03-31',sales_target='Stories only',prediction_horizon='One day ahead with archived weather at fixed previous_day1 lead')
    joblib.dump(artifact,ROOT/('models/'+prefix+'_model.joblib'))
    result=dict(sales_target='Only Stories '+args.target+'; no external sales labels',
        protocol='Jan-Mar frozen coefficients; March11-31 training-only selection; Apr-Jun chronological retrospective test.',
        baseline_validation=baseline_validation,baseline_test=baseline_test,
        selected_by_region={r:models['selection_'+r] for r in REGIONS},metrics=metrics,
        source_manifest='data/external/context_v2/manifest.json',
        caveats=['Observed oracle is retrospective, not a deployable day-ahead model.',
          'Three proxy regions are not verified branch mappings. Region results are geography sensitivity, not location identification.',
          'External signals are feature inputs; all train/test sales targets are Stories.',
          'Forecast lead-time archive is separate from realized weather; provider ingestion latency and branch exposure remain unverified.',
          'Previously explored dataset and post-selection hypothesis design mean test results are exploratory.',
          'Calendar/observed-context feature contributions explain the model adjustment, not a causal decomposition of actual sales.',
          'Maronite/Orthodox/Muslim calendars do not identify individual or branch religion.',
          'Promotion schedule is observed in the dataset; historic campaign announcement timing is unverified.'])
    (OUT/(prefix+'_model_results.json')).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__':main()
