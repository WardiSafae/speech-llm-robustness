"""
Analyse par longueur : identifie à partir de quelle longueur Whisper hallucine.

Usage :
    python scripts/analyze_length_threshold.py \
        --results results/tables/fleurs/perturbed_medium_results.csv \
        --output results/analysis/length
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return " ".join(text.split())


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ref_norm"] = df["reference"].apply(normalize)
    df["hyp_norm"] = df["hypothesis"].apply(normalize)
    df["ref_len"] = df["ref_norm"].apply(lambda s: len(s.split()))
    df["hyp_len"] = df["hyp_norm"].apply(lambda s: len(s.split()))
    df["len_ratio"] = df["hyp_len"] / df["ref_len"].clip(lower=1)
    df["wer"] = df.apply(
        lambda r: jiwer.wer(r["ref_norm"], r["hyp_norm"]) if r["ref_norm"] else 1.0,
        axis=1,
    )
    df["hallucination"] = df["len_ratio"] > 2.0
    return df


def compute_stats_by_bucket(df: pd.DataFrame, buckets: list[int]) -> pd.DataFrame:
    """
    Statistiques par tranche de longueur.
    buckets = [0, 5, 10, 15, 20, 25, 30, 40, 100]
    """
    df = df.copy()
    df["bucket"] = pd.cut(df["ref_len"], bins=buckets, right=False)

    stats = df.groupby("bucket", observed=True).agg(
        n=("wer", "count"),
        wer_mean=("wer", "mean"),
        wer_median=("wer", "median"),
        wer_max=("wer", "max"),
        hal_rate=("hallucination", "mean"),
    ).round(4)

    return stats


def find_critical_threshold(df: pd.DataFrame) -> dict:
    """
    Trouve le seuil critique par tranche de longueur.
    
    Utilise des tranches de 5 mots pour avoir assez d'échantillons.
    """
    df = df.copy()
    df["bucket"] = pd.cut(df["ref_len"], bins=range(0, 100, 5), right=False)

    by_bucket = df.groupby("bucket", observed=True).agg(
        n=("hallucination", "count"),
        hal_rate=("hallucination", "mean"),
    ).reset_index()

    # Filtrer les tranches avec assez d'échantillons
    by_bucket = by_bucket[by_bucket["n"] >= 30]

    # Convertir bucket en borne inférieure (numérique)
    by_bucket["bucket_left"] = by_bucket["bucket"].apply(lambda b: b.left)

    # Trouver le seuil où hal_rate dépasse le seuil
    threshold_10pct = None
    threshold_5pct = None
    threshold_1pct = None

    for _, row in by_bucket.sort_values("bucket_left").iterrows():
        if row["hal_rate"] >= 0.10 and threshold_10pct is None:
            threshold_10pct = int(row["bucket_left"])
        if row["hal_rate"] >= 0.05 and threshold_5pct is None:
            threshold_5pct = int(row["bucket_left"])
        if row["hal_rate"] >= 0.01 and threshold_1pct is None:
            threshold_1pct = int(row["bucket_left"])

    # Si aucun seuil trouvé, prendre la longueur du max d'hallucinations
    if threshold_10pct is None:
        # Longueur moyenne des hallucinations
        hals = df[df["hallucination"]]
        if len(hals) > 0:
            threshold_10pct = int(hals["ref_len"].min())
            threshold_5pct = int(hals["ref_len"].min())
            threshold_1pct = int(hals["ref_len"].min())

    return {
        "threshold_1pct": threshold_1pct,
        "threshold_5pct": threshold_5pct,
        "threshold_10pct": threshold_10pct,
        "n_hallucinations": int(df["hallucination"].sum()),
        "hallucination_lengths": sorted(df[df["hallucination"]]["ref_len"].tolist()),
    }


def plot_wer_vs_length(df: pd.DataFrame, output_dir: Path):
    """Scatter : WER vs longueur de la référence, coloré par langue."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"en": "#1f77b4", "fr": "#ff7f0e", "ar": "#2ca02c"}
    for lang in ["en", "fr", "ar"]:
        sub = df[df["language"] == lang]
        ax.scatter(sub["ref_len"], sub["wer"], alpha=0.2, s=8,
                   color=colors[lang], label=lang.upper())

    # Moyenne mobile
    df_sorted = df.sort_values("ref_len")
    rolling = df_sorted["wer"].rolling(100, min_periods=20).mean()
    ax.plot(df_sorted["ref_len"], rolling, color="black", linewidth=2,
            label="Moyenne mobile (100)")

    ax.set_xlabel("Longueur de la référence (mots)")
    ax.set_ylabel("WER")
    ax.set_title("WER vs longueur de la référence (par langue)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Seuil WER=0.5")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / "wer_vs_length.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_hallucination_rate(df: pd.DataFrame, output_dir: Path):
    """Histogramme : taux d'hallucination par tranche de longueur."""
    fig, ax = plt.subplots(figsize=(12, 6))

    buckets = [0, 5, 10, 15, 20, 25, 30, 40, 100]
    df = df.copy()
    df["bucket"] = pd.cut(df["ref_len"], bins=buckets, right=False)

    for lang in ["en", "fr", "ar"]:
        sub = df[df["language"] == lang]
        stats = sub.groupby("bucket", observed=True)["hallucination"].mean()
        x = [f"{b.left}-{b.right}" for b in stats.index]
        ax.plot(x, stats.values * 100, marker="o",
                label=lang.upper(), linewidth=2)

    ax.set_xlabel("Longueur de la référence (mots)")
    ax.set_ylabel("Taux d'hallucination (%)")
    ax.set_title("Taux d'hallucination par tranche de longueur et langue",
                 fontsize=13, fontweight="bold")
    ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="Seuil 10%")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / "hallucination_rate_by_length.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_wer_by_bucket(df: pd.DataFrame, output_dir: Path):
    """Bar chart : WER moyen par tranche de longueur et langue."""
    buckets = [0, 5, 10, 15, 20, 25, 30, 100]
    df = df.copy()
    df["bucket"] = pd.cut(df["ref_len"], bins=buckets, right=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    bucket_labels = [f"{b.left}-{b.right}" for b in df["bucket"].cat.categories]
    x = np.arange(len(bucket_labels))
    width = 0.25
    colors = {"en": "#1f77b4", "fr": "#ff7f0e", "ar": "#2ca02c"}

    for i, lang in enumerate(["en", "fr", "ar"]):
        sub = df[df["language"] == lang]
        stats = sub.groupby("bucket", observed=True)["wer"].mean()
        values = [stats.get(b, 0) for b in df["bucket"].cat.categories]
        ax.bar(x + i * width, values, width, label=lang.upper(), color=colors[lang])

    ax.set_xlabel("Longueur de la référence (mots)")
    ax.set_ylabel("WER moyen")
    ax.set_title("WER moyen par tranche de longueur et langue",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(bucket_labels, rotation=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "wer_by_length_bucket.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="results/analysis/length")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Chargement de {results_path}")
    df = pd.read_csv(results_path, sep="\t")
    df = enrich(df)
    print(f"   {len(df)} échantillons\n")

    # --- Distribution des longueurs ---
    print("📊 Distribution des longueurs de référence :")
    print(f"   Min : {df['ref_len'].min()}")
    print(f"   Max : {df['ref_len'].max()}")
    print(f"   Moyenne : {df['ref_len'].mean():.1f}")
    print(f"   Médiane : {df['ref_len'].median():.1f}")
    print()

    # --- Seuil critique ---
    thresholds = find_critical_threshold(df)
    print("🎯 Seuil critique d'hallucination :")
    print(f"   Nombre total d'hallucinations : {thresholds['n_hallucinations']}")
    print(f"   Longueurs des hallucinations : {thresholds['hallucination_lengths']}")
    print(f"   Taux 1 %  : à partir de {thresholds['threshold_1pct']} mots")
    print(f"   Taux 5 %  : à partir de {thresholds['threshold_5pct']} mots")
    print(f"   Taux 10 % : à partir de {thresholds['threshold_10pct']} mots")
    print()

    # --- Stats par tranche ---
    buckets = [0, 5, 10, 15, 20, 25, 30, 40, 100]
    stats = compute_stats_by_bucket(df, buckets)
    print("📊 Statistiques par tranche de longueur :")
    print(stats.to_string())
    print()

    # Par langue
    for lang in ["en", "fr", "ar"]:
        sub = df[df["language"] == lang]
        stats_lang = compute_stats_by_bucket(sub, buckets)
        print(f"--- {lang.upper()} ---")
        print(stats_lang.to_string())
        print()

    # Sauvegardes
    stats.to_csv(output_dir / "stats_by_length.csv")
    for lang in ["en", "fr", "ar"]:
        sub = df[df["language"] == lang]
        compute_stats_by_bucket(sub, buckets).to_csv(
            output_dir / f"stats_by_length_{lang}.csv"
        )

    # Hallucinations
    hallucinations = df[df["hallucination"]].copy()
    hallucinations[["audio_path", "language", "perturbation",
                    "ref_len", "hyp_len", "len_ratio", "wer"]].to_csv(
        output_dir / "hallucinations_by_length.csv", sep="\t", index=False
    )
    print(f"💾 {len(hallucinations)} hallucinations sauvegardées\n")

    # Figures
    print("🎨 Génération des figures...")
    plot_wer_vs_length(df, output_dir)
    plot_hallucination_rate(df, output_dir)
    plot_wer_by_bucket(df, output_dir)

    print(f"\n✅ Analyse terminée dans {output_dir}")


if __name__ == "__main__":
    main()
    