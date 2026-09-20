"""Détecte les hallucinations sur FLEURS."""
import pandas as pd
from pathlib import Path

df = pd.read_csv('results/tables/fleurs/whisper_base_results.csv')
df['ref_len'] = df['reference'].str.split().str.len()
df['hyp_len'] = df['hypothesis'].str.split().str.len()
df['ratio'] = df['hyp_len'] / df['ref_len'].clip(lower=1)

# Cas suspects : ratio > 3 ou < 0.3
suspects = df[(df['ratio'] > 3) | (df['ratio'] < 0.3)]
print(f"{len(suspects)} cas suspects sur {len(df)}")
for _, r in suspects.head(10).iterrows():
    print(f"\n{r['audio_path']}")
    print(f"  Ref ({r['ref_len']} mots): {r['reference'][:100]}")
    print(f"  Hyp ({r['hyp_len']} mots): {r['hypothesis'][:100]}")