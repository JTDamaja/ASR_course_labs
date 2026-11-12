import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- загрузка ----------
per = pd.read_csv('per_item_results.csv')

# порядок условий для наглядности (от чистого к самому шумному)
order = ['clean', 'speed_0.9x', 'speed_1.1x', 'noise_20db', 'noise_10db',
         'reverb_room_a', 'noise_0db']
present = [c for c in order if c in per['condition'].unique()]
per['condition'] = pd.Categorical(per['condition'], categories=present, ordered=True)
per = per.sort_values(['item_id', 'condition'])

# короткие подписи для осей
short = {
    'clean': 'clean',
    'speed_0.9x': '0.9x',
    'speed_1.1x': '1.1x',
    'noise_20db': '20dB',
    'noise_10db': '10dB',
    'reverb_room_a': 'reverb',
    'noise_0db': '0dB',
}

# ---------- 1. WER и CER по условиям (boxplot по уттеренсам) ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, metric in zip(axes, ['wer', 'cer']):
    data = [per[per['condition'] == c][metric].values for c in present]
    ax.boxplot(data, tick_labels=[short[c] for c in present])
    ax.set_title(f'{metric.upper()} по условиям (5 уттеренсов)')
    ax.set_ylabel(metric.upper())
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('plot_wer_cer_by_condition.png', dpi=150)
plt.close()

# ---------- 2. Средние WER/CER столбиками ----------
agg = per.groupby('condition', observed=True).agg(
    mean_wer=('wer', 'mean'), mean_cer=('cer', 'mean')).reindex(present)
x = np.arange(len(agg))
w = 0.38
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - w/2, agg['mean_wer'], w, label='mean WER', color='#d9534f')
ax.bar(x + w/2, agg['mean_cer'], w, label='mean CER', color='#5bc0de')
ax.set_xticks(x)
ax.set_xticklabels([short[c] for c in agg.index], rotation=30)
ax.set_ylabel('Значение')
ax.set_title('Средние WER и CER по типам аугментации')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_mean_wer_cer_bars.png', dpi=150)
plt.close()

# ---------- 3. WER vs CER scatter (видно расхождение метрик — задание 7) ----------
fig, ax = plt.subplots(figsize=(7, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(present)))
for c, col in zip(present, colors):
    sub = per[per['condition'] == c]
    ax.scatter(sub['wer'], sub['cer'], label=short[c], s=70, alpha=0.85, color=col)
lim = max(per['wer'].max(), per['cer'].max()) + 0.03
ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='WER = CER')
ax.set_xlabel('WER')
ax.set_ylabel('CER')
ax.set_title('WER vs CER: точки выше диагонали — CER > WER')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_wer_vs_cer_scatter.png', dpi=150)
plt.close()

# ---------- 4. RTF и длительность аудио по условиям ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, metric, title in zip(
        axes,
        ['audio_seconds', 'rtf'],
        ['Длительность аудио, с', 'RTF (decode/audio)']):
    data = [per[per['condition'] == c][metric].values for c in present]
    ax.boxplot(data, tick_labels=[short[c] for c in present])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('plot_duration_rtf.png', dpi=150)
plt.close()

# ---------- 5. «Паутина»: WER по каждому уттеренсу (задание 5/8) ----------
pivot = per.pivot_table(index='item_id', columns='condition',
                        values='wer', observed=True)
fig, ax = plt.subplots(figsize=(10, 4.5))
for item_id, row in pivot.iterrows():
    ax.plot([short[c] for c in pivot.columns], row.values, marker='o', label=str(item_id)[:12])
ax.set_ylabel('WER')
ax.set_title('WER по каждому уттеренсу в разных условиях')
ax.legend(fontsize=7, title='item_id')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('plot_wer_per_item.png', dpi=150)
plt.close()

print("Готово. Сохранены файлы:")
for f in ['plot_wer_cer_by_condition.png','plot_mean_wer_cer_bars.png',
          'plot_wer_vs_cer_scatter.png','plot_duration_rtf.png','plot_wer_per_item.png']:
    print("  -", f)