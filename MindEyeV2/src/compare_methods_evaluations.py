import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def is_interactive():
    import __main__
    return not hasattr(__main__, '__file__')


# ── Configurazione ──────────────────────────────────────────────────────
model_name = 'final_subj01_pretrained_40sess_24bs'
tables_dir = 'tables'

# Mappa: nome visualizzato -> suffisso del file.
# Il suffisso DEVE combaciare con model_name_plus_suffix prodotto da
# final_evaluations.ipynb (cioe' il basename di --all_recons_path senza '.pt')
methods = {
    'MindEye2 (baseline)': 'all_recons',
    'SDXL Enhanced':       'all_enhancedrecons',
    'ControlNet (Canny)':  'all_controlnet_canny',
    # Aggiungi altre righe qui, ad es. se valuti anche la variante HED:
    # 'ControlNet (HED)': 'all_controlnet_hed',
}

# Metodo di riferimento per calcolare il miglioramento relativo (%)
baseline_label = 'MindEye2 (baseline)'

# Metriche per cui un valore PIU' BASSO indica una ricostruzione migliore
# (EffNet-B e SwAV sono distanze, non similarita')
LOWER_IS_BETTER = {'EffNet-B', 'SwAV'}


# ── Carica e unisci le tabelle per metodo ───────────────────────────────
frames, missing = [], []
for label, suffix in methods.items():
    path = f'{tables_dir}/{model_name}_{suffix}.csv'
    if not os.path.exists(path):
        missing.append(path)
        continue
    df = pd.read_csv(path, sep='\t')
    df['Method'] = label
    frames.append(df)

if missing:
    print('ATTENZIONE: tabelle non trovate (esegui prima final_evaluations.ipynb):')
    for m in missing:
        print(f'  - {m}')

assert frames, 'Nessuna tabella trovata. Esegui final_evaluations.ipynb per almeno un metodo.'

long_df = pd.concat(frames, ignore_index=True)
wide_df = long_df.pivot(index='Metric', columns='Method', values='Value')
wide_df = wide_df[[m for m in methods if m in wide_df.columns]]  # ordine coerente con 'methods'
print(wide_df)


# ── Miglioramento relativo rispetto al baseline ─────────────────────────
# Per le metriche 'lower is better' invertiamo il segno del delta,
# cosi' che un valore POSITIVO significhi sempre 'meglio', a prescindere
# dalla direzione naturale della metrica.
comparison = wide_df.copy()

if baseline_label in comparison.columns:
    for label in wide_df.columns:
        if label == baseline_label:
            continue
        delta_pct = (comparison[label] - comparison[baseline_label]) / comparison[baseline_label].abs() * 100
        sign = comparison.index.map(lambda m: -1 if m in LOWER_IS_BETTER else 1)
        comparison[f'{label} - Δ% vs baseline'] = (delta_pct * sign).round(2)

print(comparison)


# ── Salva il confronto completo ──────────────────────────────────────────
os.makedirs(tables_dir, exist_ok=True)
out_path = f'{tables_dir}/{model_name}_method_comparison.csv'
comparison.to_csv(out_path, sep='\t')
print(f'Salvato: {out_path}')


# ── Grafico: una mini bar chart per metrica ──────────────────────────────
metrics = list(wide_df.index)
n = len(metrics)
ncols = 4
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = np.array(axes).flatten()

colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

for ax, metric in zip(axes, metrics):
    values = wide_df.loc[metric]
    ax.bar(values.index, values.values, color=colors[:len(values)])
    direction = '↓ meglio' if metric in LOWER_IS_BETTER else '↑ meglio'
    ax.set_title(f'{metric}\n({direction})', fontsize=9)
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

for ax in axes[n:]:
    ax.axis('off')

fig.suptitle(f'Confronto metodi — {model_name}', fontsize=13)
fig.tight_layout()
fig.savefig(f'{tables_dir}/{model_name}_method_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Prova a unire anche le caption metrics, se presenti ──────────────────
caption_frames = []
for label, suffix in methods.items():
    path = f'{tables_dir}/{model_name}_{suffix}_caption_metrics.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t')
        df['Method'] = label
        caption_frames.append(df)

# if caption_frames:
#     cap_long = pd.concat(caption_frames, ignore_index=True)
#     cap_wide = cap_long.pivot(index='Metric', columns='Method', values='Value')
#     display(cap_wide)
# else:
#     print('Nessuna tabella di caption metrics trovata (normale se non hai ancora '
#           'implementato l\'estensione descritta sopra).')
# DOPO
if caption_frames:
    cap_long = pd.concat(caption_frames, ignore_index=True)
    cap_wide = cap_long.pivot(index='Metric', columns='Method', values='Value')
    print(cap_wide)
else:
    print('Nessuna tabella di caption metrics trovata...') 

