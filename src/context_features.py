"""Mechanism-oriented external features, with separate predictive/observed streams.

Three geographic scenarios are retained; none identifies an anonymized branch.
Religious features describe public calendars, never individual customer religion.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/external/context_v2'
TZ='Asia/Beirut'
REGIONS=['beirut','batroun','broummana']
SOURCES={
 'maronite_rules':dict(url='https://bkerki.org/news-article/%D8%A7%D9%84%D8%B5%D9%88%D9%85-%D8%A7%D9%84%D9%83%D8%A8%D9%8A%D8%B1-%D8%B5%D9%84%D8%A7%D8%A9-%D9%88%D8%B5%D8%AF%D9%82%D8%A9-%D9%88%D8%B5%D9%8A%D8%A7%D9%85-%D8%B1%D8%B3%D8%A7%D9%84/',published_at='2026-02-09',
   fact='Pastoral guidance specifies fasting until noon, weekday exceptions, Holy Week, Friday abstinence and June21-28 Apostles abstinence; observance is unknown.'),
 'maronite_start':dict(url='https://bkerki.org/calendar/15-2-2026/',published_at='2026-02-15',fact='Entry into Lent on Feb15, followed by Ash Monday Feb16; Western Easter April5.'),
 'orthodox_calendar':dict(url='https://www.goarch.org/chapel/paschalion',published_at=None,fact='Great Lent begins Feb23 and Pascha is April12; not proof of local observance.'),
 'school_security':dict(url='https://mehe.gov.lb/ar/media-center/news/%D8%A8%D9%8A%D8%A7%D9%86-%D8%B5%D8%A7%D8%AF%D8%B1-%D8%B9%D9%86-%D9%88%D8%B2%D8%A7%D8%B1%D8%A9-%D8%A7%D9%84%D8%AA%D8%B1%D8%A8%D9%8A%D8%A9-%D9%88%D8%A7%D9%84%D8%AA%D8%B9%D9%84%D9%8A%D9%85-%D8%A7%D9%84%D8%B9%D8%A7%D9%84%D9%8A',published_at='2026-03-02',fact='National school closure announced March2. Publication hour is unknown; do not backdate it to opening.'),
 'school_remote':dict(url='https://www.mehe.gov.lb/ar/media-center/news/%D8%A7%D9%84%D9%88%D8%B2%D9%8A%D8%B1%D8%A9-%D9%83%D8%B1%D8%A7%D9%85%D9%8A-%D8%A3%D8%B5%D8%AF%D8%B1%D8%AA-%D8%AA%D8%B9%D9%85%D9%8A%D9%85%D8%A7-%D8%AD%D9%88%D9%84-%D8%B6%D9%85%D8%A7%D9%86-%D8%A7%D8%B3%D8%AA%D9%85%D8%B1%D8%A7%D8%B1%D9%8A%D8%A9-%D8%A7%D9%84%D8%AA%D8%B9%D9%84%D9%8A%D9%85-%D9%81%D9%8A-%D8%B8%D9%84-%D8%A7%D9%84%D8%B8%D8%B1%D9%88%D9%81-%D8%A7%D9%84%D8%A7%D8%B3%D8%AA%D8%AB%D9%86%D8%A7%D8%A6%D9%8A%D8%A9',published_at='2026-03-09',fact='Public-school remote learning from March10 in the following weeks; private-school arrangements vary. Seven-day transition window is a research specification, not the full closure duration.'),
 'weather':dict(url='https://open-meteo.com/en/docs/historical-weather-api',published_at=None,fact='Explicit ERA5 hourly reanalysis via Open-Meteo, explanatory observed stream.'),
 'issued_weather':dict(url='https://open-meteo.com/en/docs/previous-runs-api',published_at=None,fact='ECMWF IFS0.25 fixed previous_day1 lead, distinct from observed weather; earliest data availability/ingestion latency still requires operational verification.'),
 'air':dict(url='https://open-meteo.com/en/docs/air-quality-api',published_at=None,fact='CAMS global atmospheric-composition archive via Open-Meteo; model-based exposure proxy, not a street monitor or vintage-guaranteed issued air forecast.'),
 'displacement':dict(url='https://dtm.iom.int/lebanon',published_at=None,fact='Regional IOM displacement datasets identified but direct access returned403; not joined or interpolated into daily features.'),
 'calendar':dict(url='https://www.bdl.gov.lb/bankingholidays.php',published_at=None,fact='Public holiday dates; does not establish cafe closures.'),
 'prior_calendar':dict(url='reports/external_evidence.json',published_at=None,fact='Previously sourced Ramadan, Eid, Christian holidays and conflict dates; source-specific known-at controls apply.')}


def hourly(region,stream):
    payload=json.loads((CACHE/f'{region}_{stream}.json').read_text(encoding='utf-8'))
    frame=pd.DataFrame(payload['hourly'])
    frame['local_time']=pd.to_datetime(frame.pop('time'),unit='s',utc=True).dt.tz_convert(TZ)
    frame['Date']=frame.local_time.dt.tz_localize(None).dt.normalize()
    frame['hour']=frame.local_time.dt.hour
    return frame


def calendar_features(dates):
    x=pd.DataFrame({'Date':dates})
    d=x.Date
    x['ramadan']=d.between('2026-02-18','2026-03-19').astype(float)
    x['maronite_lent']=d.between('2026-02-16','2026-04-04').astype(float)
    x['orthodox_lent']=d.between('2026-02-23','2026-04-11').astype(float)
    x['fasting_overlap']=x.ramadan*x.maronite_lent
    x['triple_calendar_overlap']=x.fasting_overlap*x.orthodox_lent
    holy=d.between('2026-03-30','2026-04-04')
    exceptions=d.isin(pd.to_datetime(['2026-03-02','2026-03-09','2026-03-19','2026-03-25']))
    x['maronite_noon_fast']=(x.maronite_lent.astype(bool)&((d.dt.dayofweek<5)&~exceptions|holy)).astype(float)
    friday=(d.dt.dayofweek==4)&~d.between('2026-04-05','2026-05-24')&~d.between('2026-01-01','2026-01-06')&~d.eq('2026-02-13')
    first_week=d.between('2026-02-16','2026-02-20')
    apostles=d.between('2026-06-21','2026-06-28')
    x['maronite_abstinence_calendar']=(friday|first_week|holy|apostles).astype(float)
    x['friday_abstinence_outside_easter']=friday.astype(float)
    x['apostles_abstinence']=apostles.astype(float)
    x['muslim_holiday']=d.isin(pd.to_datetime(['2026-03-20','2026-03-21','2026-03-22','2026-03-23','2026-05-27','2026-05-28','2026-05-29','2026-06-16','2026-06-26'])).astype(float)
    x['christian_holiday']=d.isin(pd.to_datetime(['2026-01-06','2026-02-09','2026-03-25','2026-04-03','2026-04-05','2026-04-06','2026-04-10','2026-04-12','2026-04-13'])).astype(float)
    x['civic_holiday']=d.isin(pd.to_datetime(['2026-01-01','2026-05-01','2026-05-25'])).astype(float)
    holidays=set(d[x.muslim_holiday.add(x.christian_holiday).add(x.civic_holiday).gt(0)])
    def near_holiday(date):return any(date+pd.Timedelta(days=k) in holidays for k in [-1,1])
    x['holiday_eve']=d.map(lambda t:float(t+pd.Timedelta(days=1) in holidays))
    x['holiday_after']=d.map(lambda t:float(t-pd.Timedelta(days=1) in holidays))
    x['bridge_day']=[float(t.dayofweek in [0,4] and near_holiday(t)) for t in d]
    # Cyclic month position is a hypothesis, NOT a verified salary payment date.
    phase=(d.dt.day-1)/d.dt.days_in_month
    x['month_position_sin']=np.sin(2*np.pi*phase)
    x['month_position_cos']=np.cos(2*np.pi*phase)
    x['month_edge_hypothesis']=((d.dt.day<=3)|(d.dt.day>=d.dt.days_in_month-2)).astype(float)
    x['weekend']=(d.dt.dayofweek>=5).astype(float)
    x['school_closure_mar2']=(d=='2026-03-02').astype(float)
    x['school_remote_transition_7d']=d.between('2026-03-10','2026-03-16').astype(float)
    x['escalation_known']=(d>='2026-03-03').astype(float)
    x['truce_period_retro']=d.between('2026-04-17','2026-06-19').astype(float)
    x['june20_return_regime_retro']=(d>='2026-06-20').astype(float)
    x['world_cup_window']=(d>='2026-06-11').astype(float)
    x['promotion_harbour']=((d.dt.month.isin([3,5]))&d.dt.day.between(12,19)).astype(float)
    return x


def daily_context(region):
    w=hourly(region,'observed')
    g=w.groupby('Date')
    out=g.agg(temp_mean=('temperature_2m','mean'),temp_max=('temperature_2m','max'),
        apparent_mean=('apparent_temperature','mean'),apparent_max=('apparent_temperature','max'),
        humidity_mean=('relative_humidity_2m','mean'),rain_sum=('precipitation','sum'),wind_max=('wind_speed_10m','max'),
        solar_mean=('shortwave_radiation','mean'),cloud_mean=('cloud_cover','mean'))
    for label,lo,hi in [('morning',7,12),('afternoon',12,18),('evening',18,24)]:
        part=w[w.hour.ge(lo)&w.hour.lt(hi)].groupby('Date')
        out[f'rain_{label}']=part.precipitation.sum()
        out[f'apparent_{label}']=part.apparent_temperature.mean()
    out['temperature_change_1d']=out.temp_mean.diff()
    out['heat_anomaly_7d']=out.apparent_mean-out.apparent_mean.shift(1).rolling(7,min_periods=7).mean()
    out['rain_after_dry_spell']=((out.rain_sum>1)&(out.rain_sum.shift(1).rolling(3).sum()<1)).astype(float)
    out['rain_following_day']=out.rain_sum.shift(1)
    air=hourly(region,'air').groupby('Date')
    out['dust_mean']=air.dust.mean()
    out['pm25_mean']=air.pm2_5.mean()
    out['pm10_mean']=air.pm10.mean()
    out['aerosol_mean']=air.aerosol_optical_depth.mean()
    out['dust_change_1d']=out.dust_mean.diff()
    out['rain_dust_washout']=out.rain_sum*np.log1p(out.dust_mean.shift(1))
    fc=hourly(region,'issued_weather').rename(columns=lambda c:c.replace('_previous_day1',''))
    fg=fc.groupby('Date')
    out['forecast_temp_max']=fg.temperature_2m.max()
    out['forecast_rain_sum']=fg.precipitation.sum()
    out['forecast_wind_max']=fg.wind_speed_10m.max()
    out['forecast_morning_rain']=fc[fc.hour.between(7,11)].groupby('Date').precipitation.sum()
    out['forecast_evening_rain']=fc[fc.hour.ge(18)].groupby('Date').precipitation.sum()
    out['forecast_heat_anomaly']=out.forecast_temp_max-out.temp_max.shift(1).rolling(7).mean()
    out['rain_forecast_surprise']=out.rain_sum-out.forecast_rain_sum
    out['temperature_forecast_surprise']=out.temp_max-out.forecast_temp_max
    for col in ['dust_mean','pm25_mean','rain_sum','apparent_mean']:
        out['lag2_'+col]=out[col].shift(2)
    p=json.loads((CACHE/f'{region}_observed.json').read_text(encoding='utf-8'))
    solar=pd.DataFrame(p['daily'])
    solar['Date']=pd.to_datetime(solar.time,unit='s',utc=True).dt.tz_localize(None).dt.normalize()
    for col in ['sunrise','sunset']:
        ts=pd.to_datetime(solar[col],unit='s',utc=True).dt.tz_convert(TZ)
        solar[col+'_minute']=ts.dt.hour*60+ts.dt.minute+ts.dt.second/60
        solar[col+'_utc']=pd.to_datetime(solar[col],unit='s',utc=True).astype(str)
    solar['daylight_hours']=solar.daylight_duration/3600
    out=out.reset_index().merge(solar[['Date','sunrise_minute','sunset_minute','sunset_utc','daylight_hours']],on='Date',validate='one_to_one')
    out=out[out.Date.between('2026-01-01','2026-06-30')].reset_index(drop=True)
    out=out.merge(calendar_features(out.Date),on='Date',validate='one_to_one')
    out['forecast_rain_weekend']=out.forecast_rain_sum*out.weekend
    out['ramadan_daylight_hours']=out.ramadan*out.daylight_hours
    out['region']=region
    if out.select_dtypes(include='number').isna().any().any():
        raise ValueError('Uncovered context dates or missing features')
    return out


CALENDAR_FORECAST=['ramadan','maronite_noon_fast','maronite_abstinence_calendar','orthodox_lent','fasting_overlap',
 'muslim_holiday','christian_holiday','civic_holiday','month_position_sin','month_position_cos','bridge_day','daylight_hours',
 'escalation_known','school_remote_transition_7d']
ISSUED_FORECAST=['forecast_temp_max','forecast_rain_sum','forecast_wind_max','forecast_morning_rain','forecast_evening_rain','forecast_rain_weekend']
OBSERVED=['heat_anomaly_7d','apparent_mean','humidity_mean','rain_sum','rain_morning','rain_evening','rain_after_dry_spell',
 'rain_following_day','wind_max','solar_mean','dust_mean','pm25_mean','aerosol_mean','dust_change_1d','rain_dust_washout',
 'rain_forecast_surprise','temperature_forecast_surprise']


def main():
    frames=[daily_context(r) for r in REGIONS]
    combined=pd.concat(frames,ignore_index=True)
    coast=frames[0].set_index('Date').apparent_mean
    mountain=frames[2].set_index('Date').apparent_mean
    gap=coast-mountain
    combined['coast_mountain_comfort_gap']=combined.Date.map(gap)
    combined['weekend_comfort_gap']=combined.coast_mountain_comfort_gap*combined.weekend
    combined.to_csv(CACHE/'daily_features.csv',index=False)
    catalog=dict(researched_on='2026-10-17',sales_target='Stories transactions only',
        sources=SOURCES,calendar_predictors=CALENDAR_FORECAST,issued_weather_predictors=ISSUED_FORECAST,
        observed_explanatory_features=OBSERVED,
        known_at_limits=['Lunar observance uses previously sourced declarations; calendars cannot identify religion.',
          'Date-only announcements are conservatively excluded from same-day forecast changes.',
          'Holiday eve/after and one-off news events are explanation-only here unless original announcement vintage is verified.',
          'Archived previous_day1 weather is fixed-lead data, not a specific issuance timestamp; operational ingestion latency is not reconstructed.',
          'Actual weather, air quality, forecast errors and retrospective regime labels are never presented as day-ahead inputs.',
          'Geography is a three-region sensitivity analysis, not discovered branch locations.'])
    (CACHE/'feature_catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    print('Cached daily context:',len(combined),'rows;',len(combined.columns),'columns; three alternative regional scenarios')


if __name__=='__main__':main()
