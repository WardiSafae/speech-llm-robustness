"""Recalcule le WER avec normalisation (lowercase, ponctuation)."""
import pandas as pd
import re
import jiwer
from pathlib import Path

def normalize(text):
    """Lowercase, retire ponctuation, normalise espaces."""
    text = str(text).lower()
    # Retire ponctuation (garde lettres latines + arabes + chiffres)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # Normalise espaces
    text = " ".join(text.split())
    return text

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path, sep=",")
    #print(f"File: {p} | Columns found: {df.columns.tolist()}")
    df["ref_norm"] = df["reference"].apply(normalize)
    df["hyp_norm"] = df["hypothesis"].apply(normalize)

    print(f"{'='*70}")
    print(f"WER avec normalisation — {Path(csv_path).name}")
    print(f"{'='*70}")
    print(f"{'Langue':<8} {'N':<5} {'WER_brut':<12} {'WER_norm':<12} {'CER_norm':<12}")
    print(f"{'-'*70}")

    for lang in ["en", "fr", "ar", "ALL"]:
        if lang == "ALL":
            sub = df
        else:
            sub = df[df["language"] == lang]
        if len(sub) == 0:
            continue

        wer_raw = jiwer.wer(sub["reference"].tolist(), sub["hypothesis"].tolist())
        wer_norm = jiwer.wer(sub["ref_norm"].tolist(), sub["hyp_norm"].tolist())
        cer_norm = jiwer.cer(sub["ref_norm"].tolist(), sub["hyp_norm"].tolist())

        print(f"{lang:<8} {len(sub):<5} {wer_raw:<12.4f} {wer_norm:<12.4f} {cer_norm:<12.4f}")

if __name__ == "__main__":
    for csv in [
        "results/tables/fleurs/whisper_base_results.csv",
        "results/tables/fleurs/whisper_medium_results.csv",
    ]:
        p = Path(csv)
        if p.exists():
            compute_metrics(str(p))
            print()