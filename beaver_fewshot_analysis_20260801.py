from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'beaver_results_across_settings_models_20260801.csv'
PRICING = ROOT / 'model_pricing.csv'
OUT_DIR = ROOT / 'analysis_outputs_20260801'
OUT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(CSV)
    pricing = pd.read_csv(PRICING)
    pricing = pricing.rename(columns={'model_name': 'model'})
    merged = df.merge(pricing, on='model', how='left')
    merged['avg_input_cost_usd'] = (
        merged['avg_prompt_tokens'] / 1_000_000
    ) * merged['cost_per_million_input_tokens_usd']
    merged['avg_output_cost_usd'] = (
        merged['avg_completion_tokens'] / 1_000_000
    ) * merged['cost_per_million_output_tokens_usd']
    merged['avg_total_cost_usd'] = merged['avg_input_cost_usd'] + merged['avg_output_cost_usd']
    merged['exact_match_rate'] = merged['exact_matches'] / merged['total_evaluated']
    merged['cost_per_exact_match_usd'] = merged['avg_total_cost_usd'] / merged['exact_matches']
    merged['utility_accuracy_per_cost'] = merged['exact_match_rate'] / merged['avg_total_cost_usd']
    merged['utility_matches_per_cost'] = merged['exact_matches'] / merged['avg_total_cost_usd']

    cost_min = merged['avg_total_cost_usd'].min()
    cost_max = merged['avg_total_cost_usd'].max()
    acc_min = merged['exact_matches'].min()
    acc_max = merged['exact_matches'].max()
    merged['cost_norm'] = (merged['avg_total_cost_usd'] - cost_min) / (cost_max - cost_min)
    merged['accuracy_norm'] = (merged['exact_matches'] - acc_min) / (acc_max - acc_min)
    merged['distance_to_top_left'] = (merged['cost_norm'] ** 2 + (1 - merged['accuracy_norm']) ** 2) ** 0.5
    return merged


def plot_accuracy_by_setting(df):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.barplot(data=df, x='setting', y='exact_matches', hue='model', palette='deep', ax=ax)
    ax.set_title('Exact matches by setting and model (few-shot BEAVER DW)')
    ax.set_xlabel('Setting')
    ax.set_ylabel('Exact matches / 100')
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'accuracy_by_setting_20260801.png', dpi=200)
    plt.close(fig)


def plot_subtask_heatmap(df):
    metric_cols = [
        'avg_table_retrieval_f1',
        'avg_column_mapping_f1',
        'avg_join_key_f1',
        'avg_domain_knowledge_f1',
        'avg_query_decomposition_score',
    ]
    heat_df = df[['model', 'setting'] + metric_cols].copy()
    heat_df['label'] = heat_df['model'] + ' / s' + heat_df['setting'].astype(str)
    heat_df = heat_df.set_index('label')
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    sns.heatmap(heat_df[metric_cols], cmap='viridis', annot=True, fmt='.2f', linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('Subtask score heatmap')
    ax.set_xlabel('Subtask metric')
    ax.set_ylabel('Model / setting')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'subtask_heatmap_20260801.png', dpi=200)
    plt.close(fig)


def plot_cost_components(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    sns.barplot(data=df, x='model', y='avg_input_cost_usd', hue='setting', ax=axes[0], palette='Blues_d')
    axes[0].set_title('Average prompt-token cost by model/setting')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Input cost (USD)')
    axes[0].tick_params(axis='x', rotation=25)

    sns.barplot(data=df, x='model', y='avg_output_cost_usd', hue='setting', ax=axes[1], palette='Greens_d')
    axes[1].set_title('Average completion-token cost by model/setting')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Output cost (USD)')
    axes[1].tick_params(axis='x', rotation=25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'cost_components_20260801.png', dpi=200)
    plt.close(fig)


def plot_total_cost_vs_accuracy(df):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sns.scatterplot(data=df, x='avg_total_cost_usd', y='exact_matches', hue='model', style='setting', s=90, ax=ax)

    ideal_x = 0
    ideal_y = df['exact_matches'].max()
    for _, row in df.iterrows():
        ax.plot(
            [ideal_x, row['avg_total_cost_usd']],
            [ideal_y, row['exact_matches']],
            linestyle='--',
            linewidth=0.8,
            color='0.35',
            alpha=0.22,
        )

    top_utility = df.sort_values('distance_to_top_left').head(1)
    for _, row in top_utility.iterrows():
        ax.annotate(
            f"best utility\n{row['model']} / s{int(row['setting'])}",
            (row['avg_total_cost_usd'], row['exact_matches']),
            xytext=(6, 10),
            textcoords='offset points',
            fontsize=8,
            color='0.25',
            alpha=0.85,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.7', alpha=0.6),
        )

    ax.set_title('Exact matches vs derived total cost with top-left utility guide')
    ax.set_xlabel('Derived average total cost (USD)')
    ax.set_ylabel('Exact matches / 100')
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'cost_vs_accuracy_20260801.png', dpi=200)
    plt.close(fig)


def write_report(df):
    rows = df.sort_values(['setting', 'model']).reset_index(drop=True)
    summary = (
        rows[['model', 'setting', 'exact_matches', 'avg_table_retrieval_f1', 'avg_column_mapping_f1', 'avg_join_key_f1', 'avg_domain_knowledge_f1', 'avg_query_decomposition_score', 'avg_latency_sec', 'avg_total_tokens', 'avg_input_cost_usd', 'avg_output_cost_usd', 'avg_total_cost_usd']]
        .round(6)
        .to_markdown(index=False)
    )
    utility_rank = (
        df[['model', 'setting', 'exact_matches', 'avg_total_cost_usd', 'exact_match_rate', 'utility_accuracy_per_cost', 'utility_matches_per_cost', 'distance_to_top_left']]
        .sort_values('utility_accuracy_per_cost', ascending=False)
        .round(6)
        .to_markdown(index=False)
    )

    md = f'''# BEAVER few-shot comparative analysis (20260801)

## Scope
- Source CSV: [beaver_results_across_settings_models_20260801.csv](beaver_results_across_settings_models_20260801.csv)
- Pricing source: [model_pricing.csv](model_pricing.csv)
- Dataset: few-shot BEAVER DW runs only.
- Models included: Claude Haiku 4.5, Claude Sonnet 4.5, GPT-5.4-mini, and GPT-5.5.
- All runs use the same 100-question evaluation slice.

## Main findings
- The large OpenAI model remains the strongest in absolute correctness at the top end of the ladder.
- Claude Sonnet 4.5 is the best Claude model in the current sweep and is notably stronger than Claude Haiku 4.5.
- The cost signal is dominated by the prompt-input side at low settings and by completion output side for some larger runs.
- A simple utility view, defined as exact-match rate divided by average total cost, changes the ranking: the best cost-adjusted systems are the lower-cost setting-2 runs, especially Claude Haiku 4.5 and GPT-5.4-mini.
- In the updated scatter, the dashed lines point back toward the top-left ideal corner (highest accuracy, lowest cost), and the normalized distance-to-ideal is a compact visual proxy for cost-adjusted utility.

## Utility ranking
The utility metric used here is `exact_match_rate / avg_total_cost_usd`, so larger values mean more correctness per dollar. The current top-ranked points are:

{utility_rank}

## Evidence snapshot
{summary}

## Cost derivation
Average total cost was derived from the pricing sheet by:
- prompt cost = avg_prompt_tokens / 1,000,000 × cost per million input tokens
- completion cost = avg_completion_tokens / 1,000,000 × cost per million output tokens
- total cost = prompt cost + completion cost

This means the reported average cost does not depend on the blank `avg_cost_usd` column in the experiment CSV; it is derived explicitly from the pricing lookup.

## Figures
### Accuracy by setting
![](analysis_outputs_20260801/accuracy_by_setting_20260801.png)

### Subtask heatmap
![](analysis_outputs_20260801/subtask_heatmap_20260801.png)

### Cost components
![](analysis_outputs_20260801/cost_components_20260801.png)

### Cost vs accuracy
![](analysis_outputs_20260801/cost_vs_accuracy_20260801.png)
'''
    (ROOT / 'beaver_analysis_20260801.md').write_text(md, encoding='utf-8')


def main():
    df = load_data()
    plot_accuracy_by_setting(df)
    plot_subtask_heatmap(df)
    plot_cost_components(df)
    plot_total_cost_vs_accuracy(df)
    write_report(df)
    print('Wrote report and graphics to root and analysis_outputs_20260801.')


if __name__ == '__main__':
    main()
