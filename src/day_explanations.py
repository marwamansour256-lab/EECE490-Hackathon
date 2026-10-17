"""A daily evidence ledger plus an exact accounting bridge for every Stories day.

The accounting bridge identifies traffic/basket/unit-margin arithmetic. External
context and model contributions are separate evidence, never asserted causes.
"""
from pathlib import Path
from itertools import permutations
import json
import numpy as np
import pandas as pd
from strict_model import fit_engine,features
from context_features import SOURCES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'


def profit_bridge(before,after):
    before=np.asarray(before,dtype=float);after=np.asarray(after,dtype=float)
    result=np.zeros(3)
    for order in permutations(range(3)):
        state=before.copy()
        for k in order:
            old=np.prod(state);state[k]=after[k]
            result[k]+=np.prod(state)-old
    result/=6
    np.testing.assert_allclose(result.sum(),np.prod(after)-np.prod(before),atol=1e-7)
    return dict(traffic=result[0],basket_units=result[1],paid_margin_and_item_mix=result[2])


def evidence_for_day(date,ctx,events,association,branch='All branches'):
    items=[]
    def add(key,mechanism,status,action,falsifier,source=None):
        items.append(dict(id=key,mechanism=mechanism,evidence_status=status,business_action=action,
                          falsifier_or_missing_evidence=falsifier,source_url=source))
    if ctx['month_edge_hypothesis']:
        add('month_edge','Month-edge dates have an adjusted traffic association; salary timing is only a hypothesis.',
            'Exploratory association; six monthly cycles, no verified customer pay dates',
            'Check stock and staffing around month edges; verify future-cycle prediction before changing preparation.',
            'A payroll mechanism needs actual payment dates and customer/catchment exposure. Calendar negative controls also show structure.')
    if ctx['triple_calendar_overlap']:
        add('overlapping_fast_menu','Ramadan and two Christian fasting calendars overlap; meat-to-tuna mix is the tested outcome.',
            'Exploratory BH-supported menu association; not BY-supported at5%, not a customer religion label',
            'Review meat/fish preparation mix and margins; test future calendar periods before applying a multiplier.',
            'Timing/menu changes must match the mechanism; this overlap is one episode and overlaps conflict.',SOURCES['maronite_rules']['url'])
    if ctx['maronite_noon_fast']:
        add('noon_clock','Published Maronite fasting guidance has a noon endpoint, distinct from sunset.',
            'Calendar rule verified; no individual observance or universal sales effect established',
            'Inspect breakfast/lunch timing; do not automatically cut stock.',
            'A genuine clock mechanism should change morning relative to afternoon demand, not merely total daily sales.',SOURCES['maronite_rules']['url'])
    if ctx['ramadan']:
        add('sunset_clock','Compare equal two-hour windows before and after local sunset, not a fixed18:00 boundary.',
            'Calendar context; association tests are in context_v2_associations.csv',
            'Review evening batch timing and staffing against actual opening hours.',
            'No sunset-relative shift weakens a fasting-timing explanation; other operating changes may explain it.',SOURCES['prior_calendar']['url'])
    if abs(ctx['rain_forecast_surprise'])>=5:
        add('rain_surprise','Observed rain differs from the archived24-hour-lead forecast; the study finds an evening-share association.',
            'Retrospective association; surprise is unknown before the weather occurs',
            'Consider an intraday review of the remaining late batch, using live observations and measured lead time.',
            'Archive reanalysis is not live station data. Verify branch rain and that early-day surprise predicts later demand.',SOURCES['issued_weather']['url'])
    if ctx['rain_sum']>=10:
        add('rain_timing','Rain timing may redistribute orders around sunset without materially changing daily totals.',
            'Geography-sensitive association; future-rain negative-control warning remains',
            'Review sales timing and dine-in/takeaway staffing rather than assuming total demand falls.',
            'Batroun is a scenario, not a verified store location; distant grids or future-rain associations weaken attribution.',SOURCES['weather']['url'])
    if ctx['school_remote_transition_7d']:
        add('school_remote','A public-school remote-learning transition follows the March9 announcement.',
            'Announcement verified; seven-day feature is a research window, actual store exposure unknown',
            'Check school catchments and morning access before interpreting a demand change.',
            'Private schools differ; no branch-to-school exposure mapping or opening-hours logs.',SOURCES['school_remote']['url'])
    if ctx['muslim_holiday']:
        add('holiday_menu','Prior and expanded tests find lower sandwich demand on the sourced Muslim holiday dates.',
            'Repeated-date exploratory association; not a causal religious effect',
            'Review sandwich mix as a future pilot candidate; keep customer service and margins visible.',
            'Four holiday blocks and overlapping closures do not establish a universal response.',SOURCES['calendar']['url'])
    if branch in ['Harbour','All branches'] and pd.notna(ctx.get('reachability_min_drop_proxy_pct')) and ctx['reachability_min_drop_proxy_pct']>=3:
        add('internet_measurement_check','National probed-network reachability is lower relative to its preceding7-day reference.',
            'Exploratory Harbour association: global BH survives, BY does not; national signal cannot establish a store outage',
            'Check POS offline receipts, payment/connectivity logs and delayed sync before treating recorded sales as lost customer demand.',
            'Measurement coverage is separate from disruption. Actual branch ISP, POS access and missing-sales reconciliation are unavailable.',
            'https://api.ioda.inetintel.cc.gatech.edu/v2/')
    for event in events:
        if event['date']==date:
            add(event['id'],event['fact'],event.get('ledger_status','Verified event occurrence; store exposure/cause unresolved'),
                'Review the event mechanism against the branch/daypart footprint before any override.',
                event.get('missing','Branch exposure and timing are unverified'),event.get('url'))
    if not items:
        add('unresolved','No verified event-specific explanation meets the evidence threshold on this date.',
            'Unresolved; observed environmental context is still recorded',
            'Use the validated forecast and check operations if the unexplained gap is large.',
            'Opening hours, stockouts, campaigns, POS outages and local access records are missing.')
    return items


def main():
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    orders=raw.groupby('Transaction ID').agg(Date=('Date','first'),Branch=('Branch','first'))
    count=orders.groupby(['Date','Branch']).size().rename('orders')
    daily=raw.groupby(['Date','Branch']).agg(units=('Quantity','sum'),revenue=('Line Revenue (USD)','sum'),gross_profit=('Line Gross Profit (USD)','sum')).join(count).reset_index()
    daily['Category']='Orders'
    training=daily.Date.le('2026-03-31')
    order_model=fit_engine(daily[training].rename(columns={'orders':'actual'}),.001,'2026-01-01',['Orders'])
    daily['expected_orders']=order_model.predict(features(daily,'2026-01-01',['Orders'])[0])
    historical=daily[daily.Date.between('2026-03-04','2026-03-31')].copy()
    historical['weekday']=historical.Date.dt.dayofweek
    averages=historical.groupby(['Branch','weekday'])[['units','orders','gross_profit']].sum()
    averages['base_basket_units']=averages.units/averages.orders
    averages['base_unit_margin']=averages.gross_profit/averages.units
    daily['weekday']=daily.Date.dt.dayofweek
    daily=daily.merge(averages[['base_basket_units','base_unit_margin']],on=['Branch','weekday'],validate='many_to_one')
    daily['accounting_baseline_units']=daily.expected_orders*daily.base_basket_units
    daily['expected_gross_profit']=daily.accounting_baseline_units*daily.base_unit_margin
    network=daily.groupby('Date')[['units','orders','revenue','gross_profit','expected_orders','accounting_baseline_units','expected_gross_profit']].sum().reset_index()
    network['Branch']='All branches'
    network['base_basket_units']=network.accounting_baseline_units/network.expected_orders
    network['base_unit_margin']=network.expected_gross_profit/network.accounting_baseline_units
    all_days=pd.concat([daily,network],ignore_index=True)
    ctx_all=pd.read_csv(ROOT/'data/external/context_v2/daily_features.csv',parse_dates=['Date'])
    ctx=ctx_all[ctx_all.region.eq('beirut')].set_index('Date')
    internet_path=ROOT/'data/external/context_v2/internet_daily.csv'
    if internet_path.exists():
        internet=pd.read_csv(internet_path,parse_dates=['Date']).set_index('Date')
        ctx=ctx.join(internet[['reachability_mean','reachability_min_drop_proxy_pct','coverage_fraction']])
    models=json.loads((OUT/'context_v2_model_results.json').read_text(encoding='utf-8'))
    selected=models['selected_by_region']['beirut']['selected_method']
    prediction=pd.read_csv(OUT/'context_v2_predictions.csv',parse_dates=['Date'])
    method=models['selected_by_region']['beirut']['best_external'] if selected=='internal_baseline' else selected
    prediction=prediction[prediction.method.eq(method)]
    branch_pred=prediction.groupby(['Date','Branch'])[['baseline_mean','context_mean']].sum()
    net_pred=prediction.groupby('Date')[['baseline_mean','context_mean']].sum()
    decomp=pd.read_csv(OUT/'context_v2_decomposition.csv',parse_dates=['Date'])
    decomp=decomp[decomp.method.eq(method)]
    branch_decomp=decomp.groupby(['Date','Branch','group']).contribution_units.sum()
    net_decomp=decomp.groupby(['Date','group']).contribution_units.sum()
    events=json.loads((ROOT/'data/external/outlier_event_registry.json').read_text(encoding='utf-8'))
    team_path=ROOT/'data/external/external_events.json'
    if team_path.exists():
        team=json.loads(team_path.read_text(encoding='utf-8'))
        for event in team['events']:
            if event.get('validation_status') in ['not_found','unsourced'] or not event.get('source_url'):
                continue
            if event['event_id'].startswith('HOLIDAYS-') or 'AUB-COMMENCEMENT' in event['event_id']:
                continue
            events.append(dict(id=event['event_id'],date=event['event_start'][:10],
                fact='Teammate event candidate '+event['event_id']+'. Proposed mechanism: '+event['mechanism'],
                url=event['source_url'],missing='Branch exposure remains '+event.get('branch_exposure','unknown')+'; announcement-time and source-strength checks still apply.',
                ledger_status='Teammate registry: '+event.get('validation_status','unknown')+'; this ledger does not upgrade source status or establish a sales cause.'))
    association=pd.read_csv(OUT/'context_v2_associations.csv')
    order_result=json.loads((OUT/'context_v2_orders_model_results.json').read_text(encoding='utf-8'))
    order_method=order_result['selected_by_region']['beirut']['selected_method']
    order_candidate=order_result['selected_by_region']['beirut']['best_external'] if order_method=='internal_baseline' else order_method
    order_preds=pd.read_csv(OUT/'context_v2_orders_predictions.csv',parse_dates=['Date'])
    order_preds=order_preds[order_preds.method.eq(order_candidate)]
    order_by_branch=order_preds.groupby(['Date','Branch'])[['baseline_mean','context_mean']].sum()
    order_network=order_preds.groupby('Date')[['baseline_mean','context_mean']].sum()
    ledger=[]
    for row in all_days.sort_values(['Date','Branch']).itertuples():
        date=str(row.Date.date())
        before=[row.expected_orders,row.base_basket_units,row.base_unit_margin]
        after=[row.orders,row.units/row.orders,row.gross_profit/row.units]
        bridge=profit_bridge(before,after)
        pred=net_pred.loc[row.Date] if row.Branch=='All branches' else branch_pred.loc[(row.Date,row.Branch)]
        ds=net_decomp.loc[row.Date] if row.Branch=='All branches' else branch_decomp.loc[(row.Date,row.Branch)]
        if selected=='internal_baseline':
            model_mean=float(pred.baseline_mean);parts={};external_adjustment=0.
        else:
            model_mean=float(pred.context_mean);parts={str(k):float(v) for k,v in ds.items()};external_adjustment=float(pred.context_mean-pred.baseline_mean)
        np.testing.assert_allclose(float(pred.baseline_mean)+sum(parts.values()),model_mean,atol=1e-7)
        context=ctx.loc[row.Date]
        fields=['temp_mean','apparent_mean','heat_anomaly_7d','rain_sum','rain_morning','rain_evening','dust_mean','pm25_mean',
                'forecast_temp_max','forecast_rain_sum','rain_forecast_surprise','sunset_minute','daylight_hours',
                'ramadan','maronite_noon_fast','triple_calendar_overlap','maronite_abstinence_calendar','month_edge_hypothesis','school_remote_transition_7d','muslim_holiday']
        evidence=evidence_for_day(date,context.to_dict(),events,association,row.Branch)
        traffic=order_network.loc[row.Date] if row.Branch=='All branches' else order_by_branch.loc[(row.Date,row.Branch)]
        traffic_mean=float(traffic.baseline_mean if order_method=='internal_baseline' else traffic.context_mean)
        region_context={r:{k:float(g.iloc[0][k]) for k in ['rain_sum','apparent_mean','dust_mean','rain_forecast_surprise']} for r,g in ctx_all[ctx_all.Date.eq(row.Date)].groupby('region')}
        ledger.append(dict(date=date,branch=row.Branch,
            evaluation_status='training_period_retrospective_fit' if row.Date<=pd.Timestamp('2026-03-31') else 'chronological_test_previously_explored',
            actual=dict(orders=int(row.orders),units=float(row.units),revenue_usd=float(row.revenue),recorded_gross_profit_usd=float(row.gross_profit)),
            accounting_baseline=dict(orders=float(row.expected_orders),basket_units=float(row.base_basket_units),unit_margin_usd=float(row.base_unit_margin),gross_profit_usd=float(row.expected_gross_profit)),
            accounting_profit_gap_usd=float(row.gross_profit-row.expected_gross_profit),
            exact_accounting_bridge_usd={k:float(v) for k,v in bridge.items()},
            traffic_forecast=dict(method=order_method,baseline_orders=float(traffic.baseline_mean),predicted_orders=traffic_mean,
                external_adjustment_orders=traffic_mean-float(traffic.baseline_mean),unexplained_orders=float(row.orders-traffic_mean)),
            forecast=dict(method=selected,baseline_units=float(pred.baseline_mean),context_adjustment_units=external_adjustment,
                predicted_units=model_mean,unexplained_actual_minus_prediction_units=float(row.units-model_mean),
                model_adjustment_by_group_units=parts,contribution_reference='Difference from training-average standardized context, not causal effects'),
            context_primary_proxy={k:float(context[k]) for k in fields},alternative_geography_proxies=region_context,
            internet_measurement_context={k:float(context[k]) if pd.notna(context.get(k)) else None for k in ['reachability_mean','reachability_min_drop_proxy_pct','coverage_fraction']},
            event_and_mechanism_evidence=evidence,
            status='Accounting and model adjustments reconcile; external causes of actual sales remain unproven.',
            financial_warning='Recorded gross profit, not net profit; gaps are not recoverable savings; cost definitions and real operations are unverified.'))
    result=dict(researched_on='2026-10-17',days=181,scopes=['All branches','Garden','Harbour','Uptown'],records=ledger,
        geography='Beirut primary proxy; Batroun/Broummana sensitivity only. No verified branch mapping.',
        accounting_reference='Order-count model trained Jan-Mar; weekday basket/margin reference from final28 training dates. Jan-Mar ledger is retrospective/in-sample, not forecast verification.',
        confidence='Mechanisms are evidence-linked candidates, not exhaustive or causal explanations of every day.')
    (OUT/'daily_why_ledger.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    flat=[]
    for r in ledger:
        flat.append(dict(Date=r['date'],Branch=r['branch'],actual_orders=r['actual']['orders'],actual_units=r['actual']['units'],
            gross_profit_usd=r['actual']['recorded_gross_profit_usd'],profit_gap_usd=r['accounting_profit_gap_usd'],
            traffic_bridge_usd=r['exact_accounting_bridge_usd']['traffic'],basket_bridge_usd=r['exact_accounting_bridge_usd']['basket_units'],
            margin_mix_bridge_usd=r['exact_accounting_bridge_usd']['paid_margin_and_item_mix'],
            forecast_units=r['forecast']['predicted_units'],external_adjustment_units=r['forecast']['context_adjustment_units'],
            unexplained_units=r['forecast']['unexplained_actual_minus_prediction_units'],
            candidate_mechanisms='; '.join(e['id'] for e in r['event_and_mechanism_evidence']),evaluation_status=r['evaluation_status']))
    pd.DataFrame(flat).to_csv(OUT/'daily_why_ledger.csv',index=False)
    print('Saved',len(ledger),'fully reconciled day/scope dossiers across181 dates')


if __name__=='__main__':main()
