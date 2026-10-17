"""Presentation figures for the primary frozen transaction-only experiment."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics = pd.read_csv(ROOT/'reports/strict_half_metrics.csv')
    finance = pd.read_csv(ROOT/'reports/strict_half_finance.csv').set_index('method')
    names = ['regularized_poisson', 'seasonal_growth', 'seasonal_frozen']
    labels = ['Trained Poisson', 'Seasonal + growth', 'Frozen seasonal']
    colors = ['#176b63', '#b86e32', '#6b7685']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), gridspec_kw={'width_ratios':[1., 1.4]})
    values = metrics[metrics.level.eq('branch_category')].set_index('method').loc[names, 'wape_pct']
    axes[0].barh(labels, values, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 34)
    for i, value in enumerate(values):
        axes[0].text(value+.4, i, f'{value:.2f}%', va='center')
    axes[0].set_xlabel('Category WAPE (%) | lower is better')
    axes[0].set_title('Forecast accuracy', loc='left', fontsize=13)
    axes[0].spines[['top','right','left']].set_visible(False)
    for name, label, color in zip(names, labels, colors):
        row = finance.loc[name]
        axes[1].scatter(row.fill_rate_pct, row.simulated_contribution, color=color, s=100)
        axes[1].annotate(f'{label}\nUSD {row.simulated_contribution:,.0f} | {row.fill_rate_pct:.2f}%',
                         (row.fill_rate_pct, row.simulated_contribution), xytext=(9, 8), textcoords='offset points', fontsize=10)
    axes[1].set_xlim(87, 96.2)
    axes[1].set_ylim(9000, 14500)
    axes[1].set_xlabel('Recorded demand fulfilled (%)')
    axes[1].set_ylabel('Simulated contribution (USD)')
    axes[1].set_title('Profit and service must be assessed together', loc='left', fontsize=13)
    axes[1].spines[['top','right']].set_visible(False)
    axes[1].grid(alpha=.15)
    fig.suptitle('Frozen half/half test: train Jan-Mar, forecast all Apr-Jun', fontsize=16, x=.04, ha='left')
    fig.text(.04, .02, 'Transaction CSV only. No test lags or external features. One-batch 90% EXPECTED unit-fill rule.\nInventory costs, demand and waste are scenario assumptions; this is not measured company savings.', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    output = ROOT/'reports/figures/strict_half_profit_service.png'
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
