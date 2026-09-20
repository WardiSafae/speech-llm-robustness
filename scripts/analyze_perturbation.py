"""
Analyse de robustesse par langue et génération de figures.

Usage :
    python scripts/analyze_perturbation.py \
        --results results/tables/fleurs/perturbed_medium_results.csv \
        --output results/figures/perturbation
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
# Normalisation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase + retire la ponctuation (garde lettres latines + arabes)."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def compute_wer_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule une matrice WER :
        index   = (perturbation, severity)
        columns = language + 'ALL'
    """
    df = df.copy()
    df["ref_norm"] = df["reference"].apply(normalize)
    df["hyp_norm"] = df["hypothesis"].apply(normalize)

    rows = []
    for (pert, sev), group in df.groupby(["perturbation", "severity"]):
        row = {"perturbation": pert, "severity": sev}
        for lang in ["en", "fr", "ar"]:
            sub = group[group["language"] == lang]
            if len(sub) > 0:
                row[lang] = jiwer.wer(sub["ref_norm"].tolist(), sub["hyp_norm"].tolist())
            else:
                row[lang] = np.nan
        row["ALL"] = jiwer.wer(group["ref_norm"].tolist(), group["hyp_norm"].tolist())
        row["n"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_baseline(df: pd.DataFrame) -> dict:
    """Baseline = speed=1.0 (non perturbé)."""
    baseline = df[(df["perturbation"] == "speed") & (df["severity"].astype(str) == "1.0")]
    baseline = baseline.copy()
    baseline["ref_norm"] = baseline["reference"].apply(normalize)
    baseline["hyp_norm"] = baseline["hypothesis"].apply(normalize)

    result = {}
    for lang in ["en", "fr", "ar", "ALL"]:
        sub = baseline if lang == "ALL" else baseline[baseline["language"] == lang]
        if len(sub) > 0:
            result[lang] = jiwer.wer(sub["ref_norm"].tolist(), sub["hyp_norm"].tolist())
    return result


def compute_delta_wer(matrix: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """ΔWER = WER - baseline, pour chaque langue et global."""
    delta = matrix.copy()
    for col in ["en", "fr", "ar", "ALL"]:
        delta[f"{col}_delta"] = delta[col] - baseline.get(col, 0.0)
    return delta


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_wer_by_perturbation(delta: pd.DataFrame, output_dir: Path):
    """Bar chart : ΔWER par perturbation × sévérité × langue."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    perturbations = ["white_noise", "reverb", "clipping", "speed"]
    colors = {"en": "#1f77b4", "fr": "#ff7f0e", "ar": "#2ca02c"}

    for ax, pert in zip(axes, perturbations):
        sub = delta[delta["perturbation"] == pert].sort_values("severity")
        x = np.arange(len(sub))
        width = 0.25

        for i, lang in enumerate(["en", "fr", "ar"]):
            ax.bar(x + i * width, sub[f"{lang}_delta"], width,
                   label=lang.upper(), color=colors[lang])

        ax.set_title(f"{pert}", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(sub["severity"], rotation=0)
        ax.set_ylabel("ΔWER")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("ΔWER par perturbation et langue (whisper-medium)",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = output_dir / "delta_wer_by_perturbation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_wer_heatmap(matrix: pd.DataFrame, output_dir: Path):
    """Heatmap : WER par (perturbation, sévérité) × langue."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    languages = ["en", "fr", "ar"]

    for ax, lang in zip(axes, languages):
        pivot = matrix.pivot_table(
            index="perturbation", columns="severity", values=lang, aggfunc="mean"
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds", ax=ax,
                    cbar_kws={"label": "WER"}, vmin=0, vmax=0.6)
        ax.set_title(f"{lang.upper()} — WER", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sévérité")
        ax.set_ylabel("Perturbation")

    plt.suptitle("Heatmap WER par langue et perturbation",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / "wer_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_delta_heatmap(delta: pd.DataFrame, output_dir: Path):
    """Heatmap : ΔWER par (perturbation, sévérité) × langue."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    languages = ["en", "fr", "ar"]

    for ax, lang in zip(axes, languages):
        pivot = delta.pivot_table(
            index="perturbation", columns="severity",
            values=f"{lang}_delta", aggfunc="mean"
        )
        sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="RdBu_r", ax=ax,
                    center=0, cbar_kws={"label": "ΔWER"}, vmin=-0.1, vmax=0.6)
        ax.set_title(f"{lang.upper()} — ΔWER", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sévérité")
        ax.set_ylabel("Perturbation")

    plt.suptitle("Heatmap ΔWER par langue et perturbation",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / "delta_wer_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_degradation_curves(delta: pd.DataFrame, output_dir: Path):
    """Courbes : ΔWER vs sévérité pour chaque perturbation × langue."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    perturbations = ["white_noise", "reverb", "clipping", "speed"]
    colors = {"en": "#1f77b4", "fr": "#ff7f0e", "ar": "#2ca02c"}

    for ax, pert in zip(axes, perturbations):
        sub = delta[delta["perturbation"] == pert].copy()

        # Trier par sévérité numérique
        def to_num(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return float(re.search(r"[\d.]+", str(s)).group())

        sub["sev_num"] = sub["severity"].apply(to_num)
        sub = sub.sort_values("sev_num")

        for lang in ["en", "fr", "ar"]:
            ax.plot(sub["sev_num"], sub[f"{lang}_delta"], marker="o",
                    label=lang.upper(), color=colors[lang], linewidth=2)

        ax.set_title(f"{pert}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sévérité")
        ax.set_ylabel("ΔWER")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Courbes de dégradation ΔWER vs sévérité",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = output_dir / "degradation_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_language_disparity(matrix: pd.DataFrame, output_dir: Path):
    """Bar chart : WER baseline par langue (disparité initiale)."""
    baseline = matrix[(matrix["perturbation"] == "speed") & (matrix["severity"] == "1.0")]
    if len(baseline) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    langs = ["en", "fr", "ar"]
    values = [baseline[lang].values[0] for lang in langs]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(langs, values, color=colors, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("WER")
    ax.set_title("WER baseline par langue (conditions propres)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "baseline_by_language.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def plot_perturbation_ranking(delta: pd.DataFrame, output_dir: Path):
    """Bar chart : perturbations classées par ΔWER moyen."""
    rankings = []
    for pert in ["white_noise", "reverb", "clipping", "speed"]:
        sub = delta[delta["perturbation"] == pert]
        mean_delta = sub["ALL_delta"].mean()
        rankings.append({"perturbation": pert, "mean_delta": mean_delta})

    rankings = sorted(rankings, key=lambda x: x["mean_delta"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["perturbation"] for r in rankings]
    values = [r["mean_delta"] for r in rankings]

    colors = ["#d62728" if v > 0.15 else "#ff7f0e" if v > 0.05 else "#2ca02c"
              for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="black")

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("ΔWER moyen (toutes sévérités)")
    ax.set_title("Classement des perturbations par impact",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "perturbation_ranking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="CSV des résultats (whisper-medium)")
    parser.add_argument("--output", default="results/figures/perturbation",
                        help="Dossier de sortie")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger
    print(f"📂 Chargement de {results_path}")
    df = pd.read_csv(results_path, sep="\t")
    print(f"   {len(df)} échantillons")
    print(f"   Langues : {df['language'].unique().tolist()}")
    print(f"   Perturbations : {df['perturbation'].unique().tolist()}")
    print()

    # Baseline
    baseline = compute_baseline(df)
    print("📊 Baseline (speed=1.0) :")
    for lang, wer in baseline.items():
        print(f"   {lang.upper():<5} : {wer:.4f}")
    print()

    # Matrices WER
    matrix = compute_wer_matrix(df)
    delta = compute_delta_wer(matrix, baseline)

    # Tableau par langue
    print("📋 WER par perturbation et langue :")
    print(f"{'Perturbation':<15} {'Sévérité':<12} "
          f"{'EN':<10} {'FR':<10} {'AR':<10} {'ALL':<10}")
    print("-" * 72)
    for _, row in matrix.sort_values(["perturbation", "severity"]).iterrows():
        print(f"{row['perturbation']:<15} {str(row['severity']):<12} "
              f"{row['en']:<10.4f} {row['fr']:<10.4f} "
              f"{row['ar']:<10.4f} {row['ALL']:<10.4f}")
    print()

    # Sauvegardes CSV
    matrix.to_csv(output_dir / "wer_matrix.csv", index=False)
    delta.to_csv(output_dir / "delta_wer_matrix.csv", index=False)
    print(f"💾 {output_dir / 'wer_matrix.csv'}")
    print(f"💾 {output_dir / 'delta_wer_matrix.csv'}")
    print()

    # Figures
    print("🎨 Génération des figures...")
    plot_wer_by_perturbation(delta, output_dir)
    plot_wer_heatmap(matrix, output_dir)
    plot_delta_heatmap(delta, output_dir)
    plot_degradation_curves(delta, output_dir)
    plot_language_disparity(matrix, output_dir)
    plot_perturbation_ranking(delta, output_dir)

    print()
    print(f"✅ Figures sauvegardées dans {output_dir}")


if __name__ == "__main__":
    main()