"""Review the evidence and financial bridge for any Stories date and branch."""
from pathlib import Path
import argparse
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date',required=True)
    p.add_argument('--branch',choices=['All branches','Harbour','Garden','Uptown'],default='All branches')
    args=p.parse_args()
    data=json.loads((ROOT/'reports/daily_why_ledger.json').read_text(encoding='utf-8'))
    matches=[r for r in data['records'] if r['date']==args.date and r['branch']==args.branch]
    if not matches:p.error('No record for that date/branch; range is2026-01-01 through2026-06-30')
    print(json.dumps(matches[0],indent=2,allow_nan=False))


if __name__=='__main__':main()
