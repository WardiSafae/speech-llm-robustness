"""
Analyse par échantillon : identifie les audios les plus difficiles.

Usage :
    python scripts/analyze_samples.py \
        --results results/tables/fleurs/perturbed_medium_results.csv \
        --output results/analysis/samples \
        --top-k 20
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return " ".join(text.split())


def compute_wer_row(ref: str, hyp: str) -> float:
    """WER pour un seul échantillon."""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 1.0 if hyp_n else 0.0
    return jiwer.wer(ref_n, hyp_n)


def compute_cer_row(ref: str, hyp: str) -> float:
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 1.0 if hyp_n else 0.0
    return jiwer.cer(ref_n, hyp_n)


def categorize_wer(wer: float) -> str:
    """Catégorise le WER d'un échantillon."""
    if wer < 0.10:
        return "🟢 Excellent"
    elif wer < 0.25:
        return "🟡 Bon"
    elif wer < 0.50:
        return "🟠 Moyen"
    elif wer < 0.80:
        return "🔴 Mauvais"
    else:
        return "⚫ Catastrophique"


# ---------------------------------------------------------------------------
# Analyse principale
# ---------------------------------------------------------------------------

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute WER/CER par ligne + catégorie."""
    df = df.copy()
    df["wer_sample"] = df.apply(lambda r: compute_wer_row(r["reference"], r["hypothesis"]), axis=1)
    df["cer_sample"] = df.apply(lambda r: compute_cer_row(r["reference"], r["hypothesis"]), axis=1)
    df["category"] = df["wer_sample"].apply(categorize_wer)
    df["root"] = df["audio_path"].apply(lambda p: Path(p).stem.split("__")[0])
    df["ref_len"] = df["reference"].apply(lambda s: len(normalize(s).split()))
    df["hyp_len"] = df["hypothesis"].apply(lambda s: len(normalize(s).split()))
    df["len_ratio"] = df["hyp_len"] / df["ref_len"].clip(lower=1)
    return df


def top_difficult_samples(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Top-K échantillons avec le WER le plus élevé."""
    return df.nlargest(top_k, "wer_sample")


def top_difficult_by_language(df: pd.DataFrame, top_k: int = 5) -> dict:
    """Top-K par langue."""
    result = {}
    for lang in ["en", "fr", "ar"]:
        sub = df[df["language"] == lang]
        result[lang] = sub.nlargest(top_k, "wer_sample")
    return result


def top_hallucinations(df: pd.DataFrame, min_ratio: float = 2.5) -> pd.DataFrame:
    """Détecte les hallucinations (hypothèse beaucoup plus longue que la référence)."""
    return df[df["len_ratio"] > min_ratio].sort_values("len_ratio", ascending=False)


def top_by_perturbation(df: pd.DataFrame) -> pd.DataFrame:
    """WER moyen par perturbation (agrégé par échantillon)."""
    stats = df.groupby("perturbation").agg(
        wer_mean=("wer_sample", "mean"),
        wer_std=("wer_sample", "std"),
        wer_max=("wer_sample", "max"),
        n=("wer_sample", "count"),
    ).round(4)
    return stats.sort_values("wer_mean", ascending=False)


def worst_samples_per_root(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Pour chaque racine (audio original), trouve la pire perturbation.
    Révèle quels audios sont intrinsèquement difficiles.
    """
    idx = df.groupby("root")["wer_sample"].idxmax()
    worst = df.loc[idx].sort_values("wer_sample", ascending=False).head(top_k)
    return worst[["root", "language", "perturbation", "severity",
                  "wer_sample", "reference", "hypothesis"]]


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_top_samples(df: pd.DataFrame, title: str, n: int = 10):
    print_section(title)
    for i, row in df.head(n).iterrows():
        print(f"[{i+1}] {row['root']} | {row['language']} | {row['perturbation']}={row['severity']}")
        print(f"    WER={row['wer_sample']:.3f} | CER={row['cer_sample']:.3f} | "
              f"len_ratio={row['len_ratio']:.2f}")
        print(f"    REF : {row['reference'][:120]}")
        print(f"    HYP : {row['hypothesis'][:120]}")
        print()


def print_hallucinations(df: pd.DataFrame, n: int = 5):
    print_section("🚨 Hallucinations détectées (hypothèse trop longue)")
    if len(df) == 0:
        print("✅ Aucune hallucination détectée.")
        return
    for i, row in df.head(n).iterrows():
        print(f"[{i+1}] {row['root']} | {row['language']} | {row['perturbation']}={row['severity']}")
        print(f"    len_ratio = {row['len_ratio']:.2f} (ref={row['ref_len']}, hyp={row['hyp_len']})")
        print(f"    REF : {row['reference'][:120]}")
        print(f"    HYP : {row['hypothesis'][:200]}")
        print()


def print_perturbation_stats(stats: pd.DataFrame):
    print_section("📊 Statistiques par perturbation")
    print(stats.to_string())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_wer_distribution(df: pd.DataFrame, output_dir: Path):
    """Distribution des WER par langue."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, lang in zip(axes, ["en", "fr", "ar"]):
        sub = df[df["language"] == lang]
        ax.hist(sub["wer_sample"], bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.axvline(sub["wer_sample"].mean(), color="red", linestyle="--",
                   label=f"moyenne = {sub['wer_sample'].mean():.3f}")
        ax.axvline(sub["wer_sample"].median(), color="green", linestyle=":",
                   label=f"médiane = {sub['wer_sample'].median():.3f}")
        ax.set_title(f"{lang.upper()} — Distribution des WER", fontsize=13, fontweight="bold")
        ax.set_xlabel("WER par échantillon")
        ax.set_ylabel("Nombre d'échantillons")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Distribution des WER par échantillon",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / "wer_distribution_by_lang.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_top_failures(df: pd.DataFrame, output_dir: Path, top_k: int = 30):
    """Bar chart des pires échantillons."""
    top = df.nlargest(top_k, "wer_sample")

    fig, ax = plt.subplots(figsize=(14, max(6, top_k * 0.3)))
    colors = {"en": "#1f77b4", "fr": "#ff7f0e", "ar": "#2ca02c"}
    bar_colors = [colors[l] for l in top["language"]]

    labels = [f"{r['root']} ({r['perturbation']})" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["wer_sample"], color=bar_colors, edgecolor="black")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("WER")
    ax.set_title(f"Top {top_k} échantillons les plus difficiles",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Légende
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors[l], label=l.upper()) for l in ["en", "fr", "ar"]]
    ax.legend(handles=legend, loc="lower right")

    plt.tight_layout()
    out = output_dir / "top_failures.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_wer_vs_ref_length(df: pd.DataFrame, output_dir: Path):
    """WER vs longueur de la référence (par langue)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, lang in zip(axes, ["en", "fr", "ar"]):
        sub = df[df["language"] == lang]
        ax.scatter(sub["ref_len"], sub["wer_sample"], alpha=0.3, s=15, color="#1f77b4")
        ax.set_title(f"{lang.upper()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Longueur de la référence (mots)")
        ax.set_ylabel("WER")
        ax.grid(alpha=0.3)

    plt.suptitle("WER vs longueur de la référence",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / "wer_vs_length.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_category_distribution(df: pd.DataFrame, output_dir: Path):
    """Distribution des catégories (Excellent/Bon/Moyen/...) par langue."""
    categories_order = ["🟢 Excellent", "🟡 Bon", "🟠 Moyen", "🔴 Mauvais", "⚫ Catastrophique"]

    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    for lang in ["en", "fr", "ar"]:
        sub = df[df["language"] == lang]
        for cat in categories_order:
            count = (sub["category"] == cat).sum()
            pct = 100 * count / len(sub)
            data.append({"language": lang.upper(), "category": cat, "pct": pct})

    df_plot = pd.DataFrame(data)
    pivot = df_plot.pivot(index="language", columns="category", values="pct")
    pivot = pivot.reindex(columns=[c for c in categories_order if c in pivot.columns])

    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn_r", edgecolor="black")
    ax.set_title("Distribution des catégories WER par langue",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("% des échantillons")
    ax.set_xlabel("Langue")
    ax.legend(title="Catégorie", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "category_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="results/analysis/samples")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger et enrichir
    print(f"📂 Chargement de {results_path}")
    df = pd.read_csv(results_path, sep="\t")
    df = enrich_dataframe(df)
    print(f"   {len(df)} échantillons enrichis\n")

    # --- Statistiques globales ---
    print_section("📊 Statistiques globales")
    print(f"WER moyen     : {df['wer_sample'].mean():.4f}")
    print(f"WER médian    : {df['wer_sample'].median():.4f}")
    print(f"WER 95e pct   : {df['wer_sample'].quantile(0.95):.4f}")
    print(f"WER max       : {df['wer_sample'].max():.4f}")
    print(f"CER moyen     : {df['cer_sample'].mean():.4f}")

    # Par langue
    print_section("📊 Statistiques par langue")
    for lang in ["en", "fr", "ar"]:
        sub = df[df["language"] == lang]
        print(f"{lang.upper()} : WER moy={sub['wer_sample'].mean():.4f} | "
              f"médian={sub['wer_sample'].median():.4f} | "
              f"max={sub['wer_sample'].max():.4f}")

    # --- Top difficiles ---
    top_all = top_difficult_samples(df, args.top_k)
    print_top_samples(top_all, f"🎯 Top {args.top_k} échantillons les plus difficiles", n=args.top_k)

    # --- Top par langue ---
    top_by_lang = top_difficult_by_language(df, top_k=5)
    for lang, sub in top_by_lang.items():
        print_top_samples(sub, f"🎯 Top 5 les plus difficiles — {lang.upper()}", n=5)

    # --- Hallucinations ---
    hallucinations = top_hallucinations(df, min_ratio=2.5)
    print_hallucinations(hallucinations, n=5)

    # --- Pires par racine ---
    print_section("🎯 Pires cas par audio original (top 10)")
    worst = worst_samples_per_root(df, top_k=10)
    print(worst.to_string(index=False))

    # --- Stats par perturbation ---
    stats = top_by_perturbation(df)
    print_perturbation_stats(stats)

    # --- Sauvegardes CSV ---
    top_all.to_csv(output_dir / f"top{args.top_k}_difficult.csv", sep="\t", index=False)
    hallucinations.to_csv(output_dir / "hallucinations.csv", sep="\t", index=False)
    worst.to_csv(output_dir / "worst_per_root.csv", sep="\t", index=False)
    stats.to_csv(output_dir / "stats_by_perturbation.csv", sep="\t")
    print(f"\n💾 CSV sauvegardés dans {output_dir}")

    # --- Figures ---
    print("\n🎨 Génération des figures...")
    plot_wer_distribution(df, output_dir)
    plot_top_failures(df, output_dir, top_k=30)
    plot_wer_vs_ref_length(df, output_dir)
    plot_category_distribution(df, output_dir)

    print(f"\n✅ Analyse terminée. Résultats dans {output_dir}")


if __name__ == "__main__":
    main()
    