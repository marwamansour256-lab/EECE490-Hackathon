"""Date-first anomaly audit with mechanism checks and explicit unknown causes.

One-sided news matching is avoided: all 181 dates retain sales/weather context,
both positive and negative extremes are inspected, and dated event candidates
also include non-extreme days. Event links are evidence of occurrence, not cause.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson, nbinom
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'

EVENTS=[
 dict(id='hariri_access', date='2026-02-14', known_at='2026-02-13', family='Road access / commemoration',
      fact='ISF announced downtown street closures, with several restrictions from 06:00. Valentine date overlaps.',
      url='https://isf.gov.lb/ar/news/%D8%AA%D8%AF%D8%A7%D8%A8%D9%8A%D8%B1-%D8%B3%D9%8A%D8%B1-%D9%81%D9%8A-%D9%88%D8%B3%D8%B7-%D8%A8%D9%8A%D8%B1%D9%88%D8%AA-%D8%AA%D8%B2%D8%A7%D9%85%D9%86%D9%8B%D8%A7-%D9%85%D8%B9-%D8%A5%D8%AD%D9%8A%D8%A7/',
      mechanism='Access disruption should be stronger in exposed downtown stores, with a compatible time profile.',
      missing='Verified branch coordinates, road access and opening-hour logs.'),
 dict(id='feb21_strikes',date='2026-02-21',known_at='2026-02-21',family='Conflict before March escalation',
      fact='Reporting describes overnight strikes in eastern Lebanon. A national March2-only indicator misses prior disruption.',
      url='https://amp.dw.com/en/israel-says-it-hit-hezbollah-sites-in-eastern-lebanon/a-76068525',
      mechanism='Requires local exposure; also first Ramadan Saturday.',missing='Coordinates and exposure; cannot separate Ramadan and conflict on one date.'),
 dict(id='march28_conflict',date='2026-03-28',known_at='2026-03-28',family='Conflict / displacement',
      fact='Reuters footage documents damage after a night strike in Beirut southern suburbs.',
      url='https://www.reutersconnect.com/item/building-damaged-following-an-israeli-night-strike-in-beiruts-southern-suburbs/dGFnOnJldXRlcnMuY29tLDIwMjY6bmV3c21sX1ZBODIzOTI4MDMyMDI2UlAx',
      mechanism='Direct disruption and demand redistribution can have opposite signs.',missing='Alias mapping, closure logs, local incident times; positive sales do not prove displacement inflow.'),
 dict(id='dst',date='2026-03-29',known_at='2026-03-16',family='Clock change / daylight / Palm Sunday',
      fact='Government memo advances clocks one hour at midnight March28/29.',
      url='https://pcm.gov.lb/arabic/subpg.aspx?pageid=26984',
      mechanism='Clock effects should shift the hour profile. They cannot explain a March28 daytime spike that preceded the change.',
      missing='Timestamp/timezone semantics; religious event also overlaps.'),
 dict(id='good_friday',date='2026-04-03',known_at=None,family='Religious calendar / school closure',
      fact='Western Good Friday is on the official holiday calendar.',url='https://www.bdl.gov.lb/bankingholidays.php',
      mechanism='Holiday footfall or meal timing, not an assumed customer religion.',missing='School/store catchments and hours; event occurs once.'),
 dict(id='labour',date='2026-05-01',known_at=None,family='Civic holiday',fact='Labour Day is on the official 2026 calendar.',
      url='https://www.bdl.gov.lb/bankingholidays.php',mechanism='Leisure/commuting mix may shift; inspect branches and dayparts.',missing='Opening hours and verified branch catchments.'),
 dict(id='may13_access',date='2026-05-14',known_at='2026-05-13',family='Transport disruption / conflict',
      fact='Reuters reported strikes on cars on May13, including a highway south of Beirut; May14 sales decline follows this.',
      url='https://au.marketscreener.com/news/israeli-airstrikes-kill-eight-people-on-highway-south-of-beirut-lebanon-s-health-ministry-says-ce7f5bdfd08ef022',
      mechanism='Next-day access or confidence effect requires route exposure and timing.',missing='No route-to-store link; no separate estimate of exam vs conflict cause.'),
 dict(id='one_run',date='2026-05-23',known_at='2026-05-22',family='Sport event / road closures',
      fact='ISF announces One Run Global waterfront race from 06:30 to 12:00, May23.',
      url='https://isf.gov.lb/ar/news/%D8%AA%D8%AF%D8%A7%D8%A8%D9%8A%D8%B1-%D8%B3%D9%8A%D8%B1-%D9%8A%D9%88%D9%85-%D8%BA%D8%AF-%D8%A7%D9%84%D8%B3%D8%A8%D8%AA-%D8%A8%D9%85%D9%86%D8%A7%D8%B3%D8%A8%D8%A9-%D8%A5%D9%82%D8%A7%D9%85%D8%A9-%D9%85/',
      mechanism='Morning increase near runners or decrease behind blocked roads; useful non-peak event comparison.',missing='Branch distances and route exposure; includes both possible directions.'),
 dict(id='aub_graduation',date='2026-06-05',known_at=None,family='Graduation / family visits',
      fact='AUB graduate ceremony was scheduled for 19:30 June5; guests were invited from18:30.',
      url='https://www.aub.edu.lb/commencement/Pages/graduate-ceremony.aspx',
      additional_source='https://www.haigazian.edu.lb/2026/06/08/class-of-2026-bids-farewell-to-haigazian-university-president-haidostian-you-are-what-you-count-in-life/',
      additional_fact='Haigazian also held commencement June5, reported June8; retrospective evidence only.',
      mechanism='Near-campus evening sales should rise more than morning sales if graduation is the driver.',missing='Campus-to-branch mapping, event exposure and announcement vintage.'),
 dict(id='return_flows',date='2026-06-24',known_at='2026-06-24',family='Population movement / waterfront access',
      fact='Reuters footage records instructions to dismantle Beirut waterfront tents on June24.',
      url='https://www.reutersconnect.com/item/displaced-families-told-to-dismantle-beirut-tents-as-governor-says-shelters-secured/dGFnOnJldXRlcnMuY29tLDIwMjY6bmV3c21sX1ZBMTk4MzI0MDYyMDI2UlAx',
      mechanism='Receiving-area demand may change; this is not a uniform national war effect.',missing='No verified branch exposure; neither customers nor movement can be inferred from IDs.'),
]


def main():
    raw=pd.read_csv(ROOT/'data/stories_transactions.csv',parse_dates=['Date'])
    o=raw.groupby('Transaction ID').agg(Date=('Date','first'),Branch=('Branch','first'),Time=('Time','first'),
        order_type=('Order Type','first'),revenue=('Line Revenue (USD)','sum'),profit=('Line Gross Profit (USD)','sum'))
    o['hour']=o.Time.str[:2].astype(int)
    dates=pd.date_range(raw.Date.min(),raw.Date.max())
    x=pd.DataFrame({'const':1.,'trend':np.arange(len(dates))/30},index=dates)
    for k in range(1,7):x[f'dow{k}']=(dates.dayofweek==k).astype(float)
    rows=[]
    for branch in ['All branches',*sorted(raw.Branch.unique())]:
        b=o if branch=='All branches' else o[o.Branch.eq(branch)]
        for part,mask in [('all',np.ones(len(b),bool)),('before12',b.hour.lt(12)),('12to18',b.hour.between(12,17)),('18plus',b.hour.ge(18))]:
            y=b[mask].groupby('Date').size().reindex(dates,fill_value=0)
            agg=b[mask].groupby('Date').agg(revenue=('revenue','sum'),profit=('profit','sum'))
            for i in range(42,len(dates)):
                fit=sm.GLM(y.iloc[:i],x.iloc[:i],family=sm.families.Poisson()).fit()
                mu=float(fit.predict(x.iloc[[i]]).iloc[0]); actual=float(y.iloc[i])
                # NB variance fitted from training Pearson residuals; count-only
                # approximate predictive tails omit parameter uncertainty.
                dispersion=max(1.,float(fit.pearson_chi2/fit.df_resid))
                if dispersion<=1.000001:
                    dist=poisson(mu)
                else:
                    size=mu/(dispersion-1);dist=nbinom(size,size/(size+mu))
                tail=min(1.,2*min(dist.cdf(actual),dist.sf(actual-1)))
                hist=b[mask & b.Date.lt(dates[i]) & b.Date.ge(dates[i]-pd.Timedelta(days=28))]
                profit_per_order=float(hist.profit.mean())
                row=dict(Date=str(dates[i].date()),Branch=branch,daypart=part,orders=actual,expected_orders=mu,
                    difference_orders=actual-mu,residual_pct=100*(actual/mu-1),approx_p=tail,
                    profit=float(agg.loc[dates[i],'profit']) if dates[i] in agg.index else 0,
                    expected_profit=mu*profit_per_order,training_end=str(dates[i-1].date()),training_dispersion=dispersion)
                row['profit_gap']=row['profit']-row['expected_profit'];rows.append(row)
    daily=pd.DataFrame(rows)
    daily['approx_q_bh']=multipletests(daily.approx_p,method='fdr_bh')[1]
    daily.to_csv(OUT/'outlier_fingerprints.csv',index=False)
    all_day=daily[(daily.Branch=='All branches')&(daily.daypart=='all')].copy()
    extremes=pd.concat([all_day.nlargest(8,'residual_pct'),all_day.nsmallest(8,'residual_pct')]).drop_duplicates('Date')
    selected=sorted(set(extremes.Date)|{e['date'] for e in EVENTS})
    case_rows=[]
    for date in selected:
        day=daily[daily.Date.eq(date)]
        total=day[(day.Branch=='All branches')&(day.daypart=='all')].iloc[0]
        candidates=[e for e in EVENTS if e['date']==date]
        branches=day[day.Branch.ne('All branches')&day.daypart.eq('all')]
        parts=day[day.Branch.eq('All branches')&day.daypart.ne('all')]
        case=dict(date=date,orders=total.orders,expected_orders=total.expected_orders,residual_pct=total.residual_pct,
            approx_q_bh=total.approx_q_bh,profit_gap=total.profit_gap,
            branches=branches[['Branch','orders','expected_orders','residual_pct','profit_gap']].to_dict('records'),
            dayparts=parts[['daypart','orders','expected_orders','residual_pct']].to_dict('records'),
            external_candidates=candidates,attribution_status='unresolved: event occurrence is verified; sales cause is not identified')
        case_rows.append(case)
    results=dict(research_date='2026-10-17',protocol='All four scopes x four dayparts x139 dates, forecasts fit only earlier dates. News selected after anomalies is post-selection exploratory.',
        comparisons=len(daily),approx_bh_flags=int((daily.approx_q_bh<.05).sum()),cases=case_rows,
        caveats=['Count tail probabilities omit parameter uncertainty and may be miscalibrated; they are screening aids, not causal tests.',
          'Profit gap is actual contribution minus expected contribution from historical order economics; it is not recoverable profit.',
          'Search includes transport, public events, graduation, clock change, movement, conflict, weather, holidays and operations. No exhaustive claim is made.',
          'No verified date-specific external explanation found is recorded as unresolved, not filled with a plausible story.',
          'Monthly fuel, CPI and airport totals were not joined as daily causes: temporal granularity/provenance insufficient.',
          'POS downtime, paid campaigns, app rankings, competitor promotions and opening hours are unobserved alternative causes.'])
    (OUT/'outlier_dossiers.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    (ROOT/'data/external/outlier_event_registry.json').write_text(json.dumps(EVENTS,indent=2),encoding='utf-8')
    print(json.dumps({'comparisons':len(daily),'approx_bh_flags':results['approx_bh_flags'],'target_cases':len(case_rows)},indent=2))
    for date in ['2026-02-14','2026-03-28','2026-05-23','2026-06-05','2026-06-24']:
        c=next(c for c in case_rows if c['date']==date)
        print(date,round(c['residual_pct'],1),'branches',[(x['Branch'],round(x['residual_pct'],1)) for x in c['branches']],
            'dayparts',[(x['daypart'],round(x['residual_pct'],1)) for x in c['dayparts']])


if __name__=='__main__':main()
