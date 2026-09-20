"""Comparaison finale medium vs large-v3."""
import pandas as pd
import jiwer
import re

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return " ".join(text.split())

medium = pd.read_csv("results/tables/fleurs/perturbed_medium_results.csv", sep="\t")
large = pd.read_csv("results/tables/fleurs/perturbed_large_v3_results.csv", sep="\t")

for name, df in [("medium", medium), ("large-v3", large)]:
    df["ref_norm"] = df["reference"].apply(normalize)
    df["hyp_norm"] = df["hypothesis"].apply(normalize)

print(f"{'='*80}")
print(f"COMPARAISON medium vs large-v3")
print(f"{'='*80}\n")

print(f"{'Perturbation':<15} {'Sévérité':<12} {'medium':<12} {'large-v3':<12} {'Δ':<10}")
print(f"{'-'*80}")

for pert in ["white_noise", "reverb", "clipping", "speed"]:
    for sev in sorted(medium[medium["perturbation"] == pert]["severity"].unique()):
        m = medium[(medium["perturbation"] == pert) & (medium["severity"] == sev)]
        l = large[(large["perturbation"] == pert) & (large["severity"] == sev)]
        
        wer_m = jiwer.wer(m["ref_norm"].tolist(), m["hyp_norm"].tolist())
        wer_l = jiwer.wer(l["ref_norm"].tolist(), l["hyp_norm"].tolist())
        delta = wer_l - wer_m
        
        print(f"{pert:<15} {sev:<12} {wer_m:<12.4f} {wer_l:<12.4f} {delta:+.4f}")