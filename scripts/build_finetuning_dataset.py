"""
Construit le dataset de fine-tuning ciblé pour whisper-large-v3.

Composition :
    - 30% FR + reverb 0.5  (cas "Merci.")
    - 20% FR + white_noise snr5
    - 20% EN + white_noise snr5
    - 15% EN + reverb 0.3
    - 15% AR + white_noise snr5

Usage :
    python scripts/build_finetuning_dataset.py \
        --source results/tables/fleurs/perturbed_large_v3_results.csv \
        --manifest data/processed/fleurs_perturbed/manifest_perturbed.tsv \
        --output data/finetuning \
        --n_train 1000 \
        --n_val 100
"""

import argparse
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration du dataset ciblé
# ---------------------------------------------------------------------------

# Distribution cible : (perturbation, severité, langue, %)
TARGET_DISTRIBUTION = [
    ("reverb", "0.5", "fr", 0.30),          # 300 samples (cas "Merci.")
    ("white_noise", "5.0", "fr", 0.20),     # 200 samples (FR difficile)
    ("white_noise", "5.0", "en", 0.20),     # 200 samples (EN hallucine)
    ("reverb", "0.3", "en", 0.15),          # 150 samples
    ("white_noise", "5.0", "ar", 0.15),     # 150 samples
]


def select_target_samples(
    results_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    n_train: int = 1000,
    n_val: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sélectionne les échantillons cibles selon la distribution.
    """
    random.seed(seed)


    # 🔧 FIX : normaliser severity en string
    results_df = results_df.copy()
    results_df["severity"] = results_df["severity"].astype(str)
    
    # 🔧 FIX : ajuster les valeurs cibles pour matcher
    # "5.0" peut devenir "5" selon le CSV
    # On utilise une fonction de normalisation
    def normalize_sev(s):
        """Normalise '5.0' et '5' au même format."""
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return str(f)
        except (ValueError, TypeError):
            return str(s)
    
    results_df["severity_norm"] = results_df["severity"].apply(normalize_sev)


    all_selected = []
    for pert, sev, lang, ratio in TARGET_DISTRIBUTION:
        n_target = int(n_train * ratio)
        sev_norm = normalize_sev(sev)  # ← Normaliser aussi la cible


        # Filtrer les résultats correspondants
        subset = results_df[
            (results_df["perturbation"] == pert)
            & (results_df["severity_norm"] == sev_norm)
            & (results_df["language"] == lang)
        ].copy()

        if len(subset) == 0:
            print(f"⚠️  Pas d'échantillons pour {pert}={sev}, {lang}")
            continue

        # Mélanger et sélectionner
        subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_available = len(subset)
        n_take = min(n_target, n_available)

        selected = subset.head(n_take).copy()
        selected["target_category"] = f"{lang}_{pert}_{sev}"

        all_selected.append(selected)
        print(f"   {pert}={sev}, {lang} : {n_take}/{n_available}")

    # Concaténer
    df_train = pd.concat(all_selected, ignore_index=True)

    # Mélanger globalement
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split train/val
    n_total = len(df_train)
    n_val_actual = min(n_val, int(n_total * 0.1))

    df_val = df_train.head(n_val_actual)
    df_train = df_train.iloc[n_val_actual:]

    print(f"\n📊 Dataset final :")
    print(f"   Train : {len(df_train)}")
    print(f"   Val   : {len(df_val)}")
    print(f"   Total : {n_total}")

    return df_train, df_val


def prepare_for_whisper(df: pd.DataFrame, audio_root: Path) -> pd.DataFrame:
    """
    Prépare le dataframe au format Whisper (audio_path, transcript).
    """
    out = pd.DataFrame({
        "audio_path": df["audio_path"].apply(
            lambda p: str(audio_root / Path(p).name).replace("\\", "/")
        ),
        "transcript": df["reference"],
        "language": df["language"],
        "perturbation": df["perturbation"],
        "severity": df["severity"],
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="CSV des résultats large-v3")
    parser.add_argument("--manifest", required=True,
                        help="Manifest des perturbations")
    parser.add_argument("--output", default="data/finetuning")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger
    print(f"📂 Chargement de {args.source}")
    results_df = pd.read_csv(args.source, sep="\t")
    print(f"   {len(results_df)} échantillons\n")

    # Sélectionner
    print("🎯 Sélection des échantillons cibles :")
    df_train, df_val = select_target_samples(
        results_df, results_df,
        n_train=args.n_train,
        n_val=args.n_val,
    )

    # Préparer pour Whisper
    audio_root = Path("data/processed/fleurs_perturbed")
    df_train_final = prepare_for_whisper(df_train, audio_root)
    df_val_final = prepare_for_whisper(df_val, audio_root)

    # Sauvegarder
    train_path = output_dir / "train_manifest.tsv"
    val_path = output_dir / "val_manifest.tsv"

    df_train_final.to_csv(train_path, sep="\t", index=False)
    df_val_final.to_csv(val_path, sep="\t", index=False)

    print(f"\n💾 Train : {train_path} ({len(df_train_final)} entrées)")
    print(f"💾 Val   : {val_path} ({len(df_val_final)} entrées)")

    # Statistiques
    print(f"\n📊 Distribution train :")
    print(df_train_final.groupby(["language", "perturbation"]).size().to_string())


if __name__ == "__main__":
    main()