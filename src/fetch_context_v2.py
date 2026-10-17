"""Cache public context for Stories; never downloads another business's sales.

Explicit models and UTC Unix timestamps avoid ambiguous DST offsets. Actual
weather/air quality are explanatory data. Previous-run weather is a separate
archived forecast stream, never silently substituted with realized weather.
"""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'data/external/context_v2'
REGIONS = {'beirut':(33.89,35.50), 'batroun':(34.2553,35.6581), 'broummana':(33.8828,35.6206)}


def fetch(task):
    name, host, params, purpose = task
    target = DEST/(name+'.json')
    url = host+'?'+urlencode(params)
    try:
        if target.exists():
            payload = target.read_bytes()
            cached = True
        else:
            with urlopen(Request(url, headers={'User-Agent':'StoriesHackathonResearch/1.0'}), timeout=90) as response:
                payload = response.read()
            parsed = json.loads(payload)
            if parsed.get('error'):
                raise ValueError(parsed.get('reason'))
            target.write_bytes(payload)
            cached = False
        parsed = json.loads(payload)
        hourly = parsed.get('hourly', {})
        stats = {k:dict(count=len(v), null_count=sum(x is None for x in v)) for k,v in hourly.items() if k != 'time'}
        return dict(name=name, status='ok', url=url, purpose=purpose, reused_cache=cached,
                    sha256=hashlib.sha256(payload).hexdigest(), hourly_count=len(hourly.get('time',[])),
                    returned_latitude=parsed.get('latitude'), returned_longitude=parsed.get('longitude'), fields=stats)
    except (HTTPError, URLError, ValueError, TimeoutError) as exc:
        return dict(name=name,status='unavailable',url=url,purpose=purpose,error=str(exc))


def main():
    DEST.mkdir(exist_ok=True, parents=True)
    tasks=[]
    for region,(lat,lon) in REGIONS.items():
        common=dict(latitude=lat,longitude=lon,start_date='2025-12-15',end_date='2026-06-30',timezone='UTC',timeformat='unixtime')
        tasks.append((region+'_observed','https://archive-api.open-meteo.com/v1/archive',dict(common,
            models='era5', hourly='temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,shortwave_radiation,cloud_cover',
            daily='sunrise,sunset,daylight_duration'), 'Explanatory ERA5 reanalysis and astronomical clock; geography scenario, not verified branch exposure'))
        tasks.append((region+'_issued_weather','https://previous-runs-api.open-meteo.com/v1/forecast',dict(common,
            models='ecmwf_ifs025',hourly='temperature_2m_previous_day1,precipitation_previous_day1,wind_speed_10m_previous_day1'),
            'Fixed 24-hour-lead archived predictions; check coverage and publication semantics before operational use'))
        tasks.append((region+'_air','https://air-quality-api.open-meteo.com/v1/air-quality',dict(common,
            domains='cams_global',hourly='pm10,pm2_5,dust,aerosol_optical_depth'),
            'CAMS global atmospheric-composition archive via Open-Meteo; retrospective context, not issued day-ahead air-quality forecasts'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(fetch,tasks))
    manifest=dict(retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        documentation=['https://open-meteo.com/en/docs/historical-weather-api','https://open-meteo.com/en/docs/previous-runs-api','https://open-meteo.com/en/docs/air-quality-api'],
        sales_target='Only data/stories_transactions.csv; no external sales targets',
        geography='Three alternative proxy regions. No branch is assigned a region without company confirmation.',
        attribution='Weather: ERA5/ECMWF via Open-Meteo. Air quality: CAMS global atmospheric composition via Open-Meteo.',records=records)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for r in records:
        print(r['name'],r['status'],r.get('hourly_count'),r.get('error',''),flush=True)


if __name__=='__main__':main()
