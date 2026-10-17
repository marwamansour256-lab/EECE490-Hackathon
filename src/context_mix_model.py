"""Category-specific external item-mix experts, trained only on Stories sales.

Keep the frozen internal category-volume model. Within each category choose a
static share control or regularized multinomial item-mix model using March11-31
validation. Externals can affect menu composition without forcing daily volume.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import poisson
from strict_model import features,metric,fit_engine,sku_parameters
from profit_policy import fill_floor_stock
from context_features import CALENDAR_FORECAST,ISSUED_FORECAST

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
FAMILIES={'branch_only':[],
 'calendar_mix':['ramadan','maronite_noon_fast','maronite_abstinence_calendar','orthodox_lent','triple_calendar_overlap','muslim_holiday','christian_holiday'],
 'weather_mix':ISSUED_FORECAST,
 'calendar_weather_mix':['ramadan','maronite_abstinence_calendar','orthodox_lent','muslim_holiday']+ISSUED_FORECAST}
CS=[.01,.1,1.]


def fit_model(data,numeric,c):
    categorical=['Branch']+(['weekday'] if numeric else [])
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical),
                            ('num',StandardScaler(),numeric)])
    model=make_pipeline(pre,LogisticRegression(C=c,max_iter=1000,solver='lbfgs'))
    use=data.Quantity.gt(0)
    model.fit(data.loc[use,categorical+numeric],data.loc[use,'Product'],logisticregression__sample_weight=data.loc[use,'Quantity'])
    return model


def predict_shares(model,data):
    classes=model.steps[-1][1].classes_
    probabilities=model.predict_proba(data)
    lookup={name:i for i,name in enumerate(classes)}
    return probabilities[np.arange(len(data)),[lookup[p] for p in data.Product]]


def weighted_log_score(data,shares):
    return float(-np.dot(data.Quantity,np.log(np.maximum(shares,1e-12)))/data.Quantity.sum())


def validation_business_score(data,shares):
    means=data.inner_category_mean.to_numpy()*np.asarray(shares)
    q=np.array([max(int(poisson.ppf(np.clip((p-c)/p,0,.999999),mu)),fill_floor_stock(mu,.90)) for mu,p,c in zip(means,data.inner_price,data.inner_cost)])
    sold=np.minimum(q,data.Quantity.to_numpy())
    return dict(validation_contribution=float(np.sum(data.inner_price.to_numpy()*sold-data.inner_cost.to_numpy()*q)),
                validation_fill_rate=float(sold.sum()/max(1,data.Quantity.sum())))


def frozen_share(training,target):
    totals=training.groupby(['Branch','Product']).Quantity.sum().rename('n').reset_index()
    totals['share']=(totals.n+1)/(totals.groupby('Branch').n.transform('sum')+totals.groupby('Branch').Product.transform('size'))
    return target[['Branch','Product']].merge(totals[['Branch','Product','share']],on=['Branch','Product'],how='left',validate='many_to_one')['share'].to_numpy()


def main():
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    prior=joblib.load(ROOT/'models/strict_half_model.joblib')
    context=pd.read_csv(ROOT/'data/external/context_v2/daily_features.csv',parse_dates=['Date'])
    context=context[context.region.eq('beirut')]
    dates=pd.DataFrame({'Date':pd.date_range('2026-01-01','2026-06-30')})
    frame=dates.merge(prior['catalog'],how='cross')
    actual=raw.groupby(['Date','Branch','Product']).Quantity.sum().rename('Quantity').reset_index()
    frame=frame.merge(actual,on=['Date','Branch','Product'],how='left',validate='one_to_one').fillna({'Quantity':0})
    frame=frame.merge(context,on='Date',validate='many_to_one')
    frame['weekday']=frame.Date.dt.dayofweek.astype(str)
    cat=frame.groupby(['Date','Branch','Category']).Quantity.sum().rename('actual').reset_index()
    inner_base=fit_engine(cat[cat.Date.lt('2026-03-11')],prior['selected_alpha'],prior['start'],prior['categories'])
    cat['inner_category_mean']=inner_base.predict(features(cat,prior['start'],prior['categories'])[0])
    frame=frame.merge(cat[['Date','Branch','Category','inner_category_mean']],on=['Date','Branch','Category'],validate='many_to_one')
    inner_economics=sku_parameters(raw[raw.Date.lt('2026-03-11')])[['Branch','Product','price','cost']].rename(columns={'price':'inner_price','cost':'inner_cost'})
    frame=frame.merge(inner_economics,on=['Branch','Product'],validate='many_to_one')
    selection=[];outputs=[];models={}
    for category,data in frame.groupby('Category',sort=True):
        data=data.reset_index(drop=True)
        initial=data.Date.lt('2026-03-11');validation=data.Date.between('2026-03-11','2026-03-31')
        training=data.Date.le('2026-03-31');test=data.Date.ge('2026-04-01')
        valid=data[validation].copy();future=data[test].copy()
        share=frozen_share(data[initial],valid)
        candidates=[dict(family='frozen_training_share',C=None,validation_log_loss=weighted_log_score(valid,share),**validation_business_score(valid,share))]
        for family,names in FAMILIES.items():
            for c in CS:
                fitted=fit_model(data[initial],names,c)
                share=predict_shares(fitted,valid)
                candidates.append(dict(family=family,C=c,validation_log_loss=weighted_log_score(valid,share),**validation_business_score(valid,share)))
        best=min(candidates,key=lambda r:r['validation_log_loss'])
        feasible=[c for c in candidates if c['validation_fill_rate']>=.90]
        profit_best=max(feasible,key=lambda r:r['validation_contribution']) if feasible else best
        if category not in ['Bakery','Breakfast','Desserts','Sandwiches']:profit_best=best
        internal_best=min([c for c in candidates if c['family']=='branch_only'],key=lambda r:r['validation_log_loss'])
        cached={}
        models[category]=dict(selection=best,profit_selection=profit_best,profit_guard_feasible=bool(feasible),fits={})
        for label,choice in [('selected_share',best),('profit_selected_share',profit_best),('internal_share',internal_best)]:
            key=choice['family']+' / '+str(choice['C'])
            if key not in cached:
                if choice['family']=='frozen_training_share':cached[key]=frozen_share(data[training],future)
                else:
                    fitted=fit_model(data[training],FAMILIES[choice['family']],choice['C'])
                    cached[key]=predict_shares(fitted,future)
                    models[category]['fits'][key]=dict(model=fitted,numeric=FAMILIES[choice['family']])
            future[label]=cached[key]
        models[category]['static']=data[training].groupby(['Branch','Product']).Quantity.sum().reset_index()
        # Strong original control: preceding28 TRAINING-day item shares, frozen.
        recent=data[data.Date.between('2026-03-04','2026-03-31')]
        future['baseline_share']=frozen_share(recent,future)
        for col in ['selected_share','profit_selected_share','internal_share','baseline_share']:
            group_sum=future.groupby(['Date','Branch'])[col].transform('sum')
            future[col]=future[col]/group_sum
        selection.append(dict(category=category,selected=best,profit_selected=profit_best,profit_guard_feasible=bool(feasible),candidates=candidates,
            test_log_loss=weighted_log_score(future,future.selected_share),
            internal_mix_test_log_loss=weighted_log_score(future,future.internal_share),
            control_test_log_loss=weighted_log_score(future,future.baseline_share)))
        outputs.append(future[['Date','Branch','Category','Product','Quantity','selected_share','profit_selected_share','internal_share','baseline_share']])
        print(category,best['family'],flush=True)
    output=pd.concat(outputs,ignore_index=True)
    unique=output[['Date','Branch','Category']].drop_duplicates()
    unique['category_mean']=prior['model'].predict(features(unique,prior['start'],prior['categories'])[0])
    output=output.merge(unique,on=['Date','Branch','Category'],validate='many_to_one')
    output=output.merge(prior['sku_parameters'][['Branch','Product','price','cost']],on=['Branch','Product'],validate='many_to_one')
    details=[];metrics=[]
    for method,col in [('selected_mix_experts','selected_share'),('profit_selected_mix','profit_selected_share'),('internal_mix_control','internal_share'),('frozen_share_control','baseline_share')]:
        d=output[['Date','Branch','Category','Product','Quantity','price','cost','category_mean']].copy()
        d['predicted_mean_units']=output.category_mean*output[col]
        d['method']=method
        d['prepare_units']=[max(int(poisson.ppf(np.clip((p-c)/p,0,.999999),mu)),fill_floor_stock(mu,.90)) for mu,p,c in zip(d.predicted_mean_units,d.price,d.cost)]
        d['sold_units']=np.minimum(d.Quantity,d.prepare_units)
        d['unsold_units']=d.prepare_units-d.sold_units
        d['unmet_units']=d.Quantity-d.sold_units
        d['contribution']=d.price*d.sold_units-d.cost*d.prepare_units
        for scope,subset in [('all_sku',d),('perishable_sku',d[d.Category.isin(['Bakery','Breakfast','Desserts','Sandwiches'])])]:
            metrics.append(dict(method=method,scope=scope,**metric(subset.Quantity,subset.predicted_mean_units)))
        details.append(d)
    detail=pd.concat(details,ignore_index=True)
    detail.to_csv(OUT/'context_v2_mix_predictions.csv',index=False)
    perishable=detail[detail.Category.isin(['Bakery','Breakfast','Desserts','Sandwiches'])]
    finance=perishable.groupby('method')[['Quantity','prepare_units','sold_units','unsold_units','unmet_units','contribution']].sum().reset_index()
    finance['fill_rate_pct']=100*finance.sold_units/finance.Quantity
    finance.to_csv(OUT/'context_v2_mix_finance.csv',index=False)
    result=dict(sales_target='Only Stories product quantities',protocol='Jan-Mar training, March11-31 internal selection, frozen coefficients on Apr-Jun test.',
        region='Beirut proxy fixed for this mix comparison, not selected by test performance',selection=selection,metrics=metrics,
        finance=finance.to_dict('records'),
        caveats=['Previously explored dataset; this is exploratory model development, not a fresh confirmatory test.',
          'Predictive selector uses training-period log score. Profit selector uses training-period contribution with90% realized validation fulfillment when feasible; failure to find a feasible candidate is disclosed. Neither selects using test outcomes.',
          'Internal-only fitted mix controls isolate the effect of external features from simply pooling more training history.',
          'Probabilities only redistribute a fixed internal category forecast; they do not inflate predicted category volume.',
          'All menu features describe public calendars; no personal religion or fasting inference.',
          'One-batch90% expected unit-fill, zero salvage, sales-as-demand and training economics; no actual labor/waste/current-policy comparison.'])
    (OUT/'context_v2_mix_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    joblib.dump(dict(models=models,base=prior),ROOT/'models/context_v2_mix_model.joblib')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(finance.to_string(index=False),flush=True)


if __name__=='__main__':main()
