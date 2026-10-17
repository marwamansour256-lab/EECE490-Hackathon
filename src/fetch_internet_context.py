"""National internet reachability as a measurement/access hypothesis, not sales.

Public IODA signal queried monthly to respect its time-range limits. A national
network signal cannot establish that a particular POS or customer lost access.
"""
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen,Request
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import subprocess
import base64
from urllib.error import URLError

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/external/context_v2'


def fetch(month):
    start=int(datetime(2026,month,1,tzinfo=timezone.utc).timestamp())
    end=int(datetime(2026,month+1,1,tzinfo=timezone.utc).timestamp())
    url='https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw/country/LB?'+urlencode({'from':start,'until':end,'datasource':'ping-slash24','maxPoints':800})
    path=DEST/f'internet_LB_2026_{month:02d}.json'
    try:
        if path.exists():payload=path.read_bytes()
        else:
            try:
                with urlopen(Request(url,headers={'User-Agent':'StoriesHackathonResearch/1.0'}),timeout=60) as response:payload=response.read()
                path.write_bytes(payload)
            except URLError as exc:
                if 'CERTIFICATE_VERIFY_FAILED' not in str(exc):raise
                # Use Windows' NORMAL trusted-root validation, never disable TLS
                # certificate/hostname verification or accept arbitrary roots.
                script="Invoke-WebRequest -Uri '"+url.replace("'","''")+"' -OutFile '"+str(path).replace("'","''")+"' -TimeoutSec 35 -ErrorAction Stop"
                encoded=base64.b64encode(script.encode('utf-16le')).decode('ascii')
                subprocess.run(['powershell','-NoProfile','-NonInteractive','-EncodedCommand',encoded],check=True,capture_output=True,timeout=45)
                payload=path.read_bytes()
        data=json.loads(payload)
        return dict(month=month,status='downloaded_requires_coverage_check',url=url,sha256=hashlib.sha256(payload).hexdigest(),top_level_keys=list(data) if isinstance(data,dict) else 'list')
    except Exception as exc:
        return dict(month=month,status='unavailable',url=url,error=str(exc))


def main():
    DEST.mkdir(exist_ok=True,parents=True)
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,range(1,7)))
    manifest=dict(source='Georgia Tech IODA active-probing national reachability',
        documentation='https://api.ioda.inetintel.cc.gatech.edu/v2/',
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),records=records,
        limitation='Research context only. Missing samples are not outages. National reachability is not proof of POS downtime or local network exposure.')
    (DEST/'internet_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
