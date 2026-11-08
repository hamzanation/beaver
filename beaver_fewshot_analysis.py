from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'beaver_results_across_settings_models_20260722.csv'
OUT = ROOT / 'analysis_outputs'
OUT.mkdir(exist_ok=True)


def load_df():
    df = pd.read_csv(CSV)
    df = df.sort_values(['setting', 'model']).reset_index(drop=True)
    return df


def plot_accuracy_by_setting(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x='setting', y='exact_matches', hue='model', ax=ax, palette='deep')
    ax.set_title('Exact matches by setting and model (few-shot DW)')
    ax.set_xlabel('Setting')
    ax.set_ylabel('Exact matches / 100')
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'accuracy_by_setting.png', dpi=200)
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
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.heatmap(heat_df[metric_cols], cmap='viridis', annot=True, fmt='.2f', linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('Subtask score heatmap')
    ax.set_xlabel('Subtask metric')
    ax.set_ylabel('Model / setting')
    fig.tight_layout()
    fig.savefig(OUT / 'subtask_heatmap.png', dpi=200)
    plt.close(fig)


def plot_resource_tradeoff(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(data=df, x='avg_latency_sec', y='exact_matches', hue='model', style='setting', s=90, ax=axes[0])
    axes[0].set_title('Accuracy vs latency')
    axes[0].set_xlabel('Average latency (sec)')
    axes[0].set_ylabel('Exact matches')

    sns.scatterplot(data=df, x='avg_total_tokens', y='exact_matches', hue='model', style='setting', s=90, ax=axes[1])
    axes[1].set_title('Accuracy vs total tokens')
    axes[1].set_xlabel('Average total tokens')
    axes[1].set_ylabel('Exact matches')

    for ax in axes:
        ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.tight_layout()
    fig.savefig(OUT / 'resource_tradeoff.png', dpi=200)
    plt.close(fig)


def summarize(df):
    out = []
    out.append('# BEAVER few-shot comparative analysis')
    out.append('')
    out.append('## Dataset and scope')
    out.append('- Source CSV: `beaver_results_across_settings_models_20260722.csv`')
    out.append('- Scope: few-shot BEAVER DW runs only; all rows evaluated on 100 questions.')
    out.append('- Metrics included: exact match, subtask scores, average latency, total tokens, and the judge-based decomposition score.')
    out.append('')
    out.append('## Main observations')
    out.append('- The biggest jump is from setting 0 to 1 for the small model family, especially `gpt-5.4-mini` (5 -> 13 exact matches).')
    out.append('- At setting 2, the small model narrows the gap but still trails the larger model in final exact matches (18 vs 26).')
    out.append('- The structural subtasks improve sharply with better settings: table retrieval, column mapping, and joins go from weak to strong.')
    out.append('- Domain knowledge and query decomposition remain comparatively weak, which suggests reasoning composition is still the hard bottleneck.')
    out.append('- Latency increases noticeably for the larger model, while the small model stays much faster in the same setting ladder.')
    out.append('')
    out.append('## Evidence snapshot')
    out.append(df[['model', 'setting', 'exact_matches', 'avg_table_retrieval_f1', 'avg_column_mapping_f1', 'avg_join_key_f1', 'avg_domain_knowledge_f1', 'avg_query_decomposition_score', 'avg_latency_sec', 'avg_total_tokens']].to_markdown(index=False))
    out.append('')
    out.append('## Figures')
    out.append('![](analysis_outputs/accuracy_by_setting.png)')
    out.append('')
    out.append('![](analysis_outputs/subtask_heatmap.png)')
    out.append('')
    out.append('![](analysis_outputs/resource_tradeoff.png)')
    return '\n'.join(out)


def main():
    df = load_df()
    plot_accuracy_by_setting(df)
    plot_subtask_heatmap(df)
    plot_resource_tradeoff(df)
    text = summarize(df)
    (ROOT / 'beaver_analysis.md').write_text(text, encoding='utf-8')
    print('Wrote:', OUT / 'accuracy_by_setting.png')
    print('Wrote:', OUT / 'subtask_heatmap.png')
    print('Wrote:', OUT / 'resource_tradeoff.png')
    print('Wrote:', ROOT / 'beaver_analysis.md')


if __name__ == '__main__':
    main()
