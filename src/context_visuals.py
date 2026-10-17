"""Evidence and daily accounting figures for the context extension."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    data=json.loads((ROOT/'reports/daily_why_ledger.json').read_text(encoding='utf-8'))['records']
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.5))
    for ax,date in zip(axes,['2026-05-01','2026-06-24']):
        r=next(r for r in data if r['date']==date and r['branch']=='All branches')
        baseline=r['accounting_baseline']['gross_profit_usd']
        changes=list(r['exact_accounting_bridge_usd'].values())
        positions=[baseline]+changes+[r['actual']['recorded_gross_profit_usd']]
        current=baseline
        ax.bar(0,baseline,color='#7a8791',width=.65)
        for i,change in enumerate(changes,start=1):
            ax.bar(i,abs(change),bottom=min(current,current+change),color='#176b63' if change>=0 else '#c4713d',width=.65)
            ax.plot([i-1+.325,i-.325],[current,current],color='#90999f',linewidth=.8)
            ax.text(i,max(current,current+change)+20,f'{change:+.0f}',ha='center',fontsize=10)
            current+=change
        ax.bar(4,positions[-1],color='#263b4f',width=.65)
        ax.set_xticks(range(5),['Reference','Traffic','Basket\nunits','Margin /\nmix','Actual'])
        ax.set_ylabel('Recorded gross profit (USD)')
        ax.set_ylim(0,max(baseline,positions[-1])*1.2)
        ax.set_title(date+f' | difference {r["accounting_profit_gap_usd"]:+.0f} USD',loc='left')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Explain the money before assigning an external cause',x=.04,ha='left',fontsize=16)
    fig.text(.04,.02,'Exact accounting decomposition, not causal attribution or recoverable profit.\nReference: Jan-Mar order model and final 28 training-day weekday basket/margin; actual costs/operations unverified.',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.94))
    folder=ROOT/'reports/figures';folder.mkdir(exist_ok=True,parents=True)
    fig.savefig(folder/'daily_profit_bridges.png',dpi=180);plt.close(fig)
    rows=pd.read_csv(ROOT/'reports/context_v2_associations.csv')
    month=rows[(rows.region=='beirut')&(rows.outcome=='network_orders')&(rows.feature=='month_edge_hypothesis')].iloc[0]
    rain=rows[(rows.region=='beirut')&(rows.outcome=='evening_order_share_pp')&(rows.feature=='rain_forecast_surprise')].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,row,scale,title,xlabel in [(axes[0],month,1/month.feature_sd,'Month-edge order association','Additional network orders/day'),
        (axes[1],rain,10/rain.feature_sd,'Unexpected-rain timing association','Evening order-share difference (percentage points /10mm)')]:
        value=row.coefficient_per_sd*scale
        lo,hi=row.ci_low*scale,row.ci_high*scale
        ax.errorbar(value,0,xerr=[[value-lo],[hi-value]],fmt='o',color='#176b63',capsize=6,markersize=8)
        ax.axvline(0,color='#7a8791',linewidth=.8)
        ax.set_yticks([]);ax.set_ylim(-1,1)
        ax.set_title(title,loc='left',fontsize=12)
        ax.set_xlabel(xlabel,fontsize=10)
        ax.text(.02,.85,f'Estimate: {value:+.2f}\nBY-adjusted q: {row.q_by:.4f}',transform=ax.transAxes,fontsize=10)
        ax.spines[['top','right','left']].set_visible(False)
    fig.text(.03,.025,'Exploratory adjusted associations with HAC intervals; not causal effects. Payroll dates and store exposure unknown.\nRain surprise uses retrospective observed weather; it is unavailable before the day. Previously explored sample.',fontsize=9)
    fig.tight_layout(rect=(0,.15,1,1))
    fig.savefig(folder/'external_mechanism_evidence.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
