"""Standalone presentation figures from the audited replay and outlier dossiers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/figures'


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    summary = pd.read_csv(ROOT/'reports/two_stage_summary.csv')
    scenarios = ['free_flexibility', 'base_costed', 'cost_stress', 'high_cost']
    labels = ['No extra batch cost', r'\$0.25 / batch + \$0.10 / unit', r'\$1 / batch + \$0.25 / unit', r'\$3 / batch + \$0.50 / unit']
    fig, ax = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(4)
    for offset, model, label, color in [(-.18, 'strong_baseline', 'Seasonal baseline', '#176b63'), (.18, 'internal', 'Poisson regression', '#be6e35')]:
        values = summary[summary.forecast.eq(model)].set_index('scenario').loc[scenarios, 'incremental_contribution'].to_numpy()
        ax.barh(positions+offset, values, height=.32, label=label, color=color)
        for y, value in zip(positions+offset, values):
            ax.text(value+(25 if value >= 0 else -25), y, f'${value:+,.0f}', va='center', ha='left' if value >= 0 else 'right', fontsize=10)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.axvline(0, color='#555555', linewidth=.8)
    ax.set_xlim(-2500, 1100)
    ax.set_xlabel('June simulated contribution difference: two-stage minus one-batch (USD)')
    ax.set_title('In this replay, extra preparation pays only at low costs', loc='left', fontsize=16, pad=18)
    ax.legend(loc='lower right')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.text(.02, .02, 'Retrospective replay, 3 branches. 90% expected unit-fill floor: per stage vs per day.\nImmediate replenishment; no capacity constraint. These are scenarios, not measured company savings.', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(OUT/'two_stage_cost_boundary.png', dpi=180)
    plt.close(fig)

    cases = json.loads((ROOT/'reports/outlier_dossiers.json').read_text(encoding='utf-8'))['cases']
    wanted = ['2026-02-14', '2026-03-28', '2026-05-23', '2026-06-05', '2026-06-24']
    values = []
    for date in wanted:
        c = next(c for c in cases if c['date'] == date)
        b = {r['Branch']: r['residual_pct'] for r in c['branches']}
        h = {r['daypart']: r['residual_pct'] for r in c['dayparts']}
        values.append([b['Garden'], b['Harbour'], b['Uptown'], h['before12'], h['12to18'], h['18plus']])
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    im = ax.imshow(values, cmap='RdBu', vmin=-45, vmax=45, aspect='auto')
    ax.set_xticks(range(6), ['Garden', 'Harbour', 'Uptown', 'Before noon', 'Noon-18:00', 'After 18:00'])
    ax.set_yticks(range(5), ['Feb 14 | access closures', 'Mar 28 | conflict reports', 'May 23 | morning race', 'Jun 5 | graduations', 'Jun 24 | waterfront changes'])
    ax.axvline(2.5, color='white', linewidth=4)
    for i in range(5):
        for j in range(6):
            ax.text(j, i, f'{values[i,j]:+.1f}%', ha='center', va='center', color='white' if abs(values[i,j]) > 27 else '#17212c')
    fig.colorbar(im, ax=ax, label='Orders versus rolling expected orders (%)', shrink=.8)
    ax.set_title('Check the branch and time pattern before explaining an outlier', loc='left', fontsize=15, pad=18)
    fig.text(.02, .025, 'Left: whole-day branch residuals. Right: network daypart residuals. Independent models; cells do not add.\nEvents are verified candidates, not identified causes. Actual branch locations are unknown.', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(OUT/'outlier_mechanism_fingerprints.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
