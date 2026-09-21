"""
Version améliorée : augmente le dataset avec des perturbations aléatoires.

Objectif : 2000-3000 échantillons pour un vrai fine-tuning.

Usage :
    python scripts/build_finetuning_dataset_v2.py \
        --source results/tables/fleurs/perturbed_large_v3_results.csv \
        --output data/finetuning \
        --n_train 2000 \
        --n_val 200 \
        --augment
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa


# Distribution cible (avec augmentation)
TARGET_DISTRIBUTION = [
    # (perturbation, severity, language, %)
    ("reverb", "0.5", "fr", 0.25),          # 500 samples
    ("reverb", "0.5", "en", 0.15),          # 300 samples
    ("white_noise", "5", "fr", 0.20),       # 400 samples
    ("white_noise", "5", "en", 0.15),       # 300 samples
    ("white_noise", "5", "ar", 0.15),       # 300 samples
    ("reverb", "0.3", "en", 0.10),          # 200 samples
]


def normalize_sev(s):
    """Normalise '5.0' et '5' au même format."""
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return str(s)


# ---------------------------------------------------------------------------
# Perturbations audio
# ---------------------------------------------------------------------------

def _white_noise(audio, snr_db=10.0):
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    return audio + np.random.normal(0, rms_noise, audio.shape)


def _reverb(audio, sr, room_scale=0.5):
    ir = np.random.randn(int(sr * room_scale)) * np.exp(-np.linspace(0, 6, int(sr * room_scale)))
    return np.convolve(audio, ir, mode="same")[: len(audio)]


def augment_audio(audio_path, perturbation, severity, output_path, seed=None):
    """
    Génère une version augmentée avec un peu de bruit aléatoire en plus.
    Différente à chaque appel (seed=None).
    """
    if seed is not None:
        np.random.seed(seed)

    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # Ajout de bruit aléatoire (petit) pour varier
    noise_level = np.random.uniform(0.005, 0.02)
    audio = audio + np.random.normal(0, noise_level, audio.shape)

    # Appliquer la perturbation principale
    if perturbation == "white_noise":
        audio = _white_noise(audio, snr_db=float(severity))
    elif perturbation == "reverb":
        audio = _reverb(audio, sr, room_scale=float(severity))

    sf.write(str(output_path), audio, sr)
    return output_path


# ---------------------------------------------------------------------------
# Construction du dataset
# ---------------------------------------------------------------------------

def build_dataset(
    source_csv: Path,
    output_dir: Path,
    n_train: int = 2000,
    n_val: int = 200,
    audio_root: Path = Path("data/processed/fleurs_perturbed"),
):
    """
    Construit le dataset en générant n_train échantillons par catégorie.
    Utilise l'augmentation pour atteindre la cible.
    """
    random.seed(42)
    np.random.seed(42)

    output_dir.mkdir(parents=True, exist_ok=True)
    aug_dir = output_dir / "augmented"
    aug_dir.mkdir(exist_ok=True)

    df = pd.read_csv(source_csv, sep="\t")
    df["severity_norm"] = df["severity"].apply(normalize_sev)

    all_selected = []
    for pert, sev, lang, ratio in TARGET_DISTRIBUTION:
        n_target = int(n_train * ratio)
        sev_norm = normalize_sev(sev)

        # Échantillons de base
        subset = df[
            (df["perturbation"] == pert)
            & (df["severity_norm"] == sev_norm)
            & (df["language"] == lang)
        ].copy()

        if len(subset) == 0:
            print(f"⚠️  Pas d'échantillons pour {pert}={sev}, {lang}")
            continue

        # Mélanger
        subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)
        n_available = len(subset)

        n_to_augment = max(0, n_target - n_available)
        print(f"   {pert}={sev}, {lang} : {n_available} dispos, "
              f"{n_target} visés (+{n_to_augment} augmentés)")

        selected_rows = []

        # 1. Originaux
        for i in range(min(n_available, n_target)):
            selected_rows.append(subset.iloc[i].to_dict())

        # 2. Augmentés
        if n_to_augment > 0:
            for i in range(n_to_augment):
                orig = subset.iloc[i % n_available]

                # Résoudre le chemin
                orig_path = Path(orig["audio_path"])
                if not orig_path.exists():
                    orig_path = audio_root / orig_path.name

                aug_name = f"{orig_path.stem}__aug{i:05d}.wav"
                aug_path = aug_dir / aug_name

                try:
                    augment_audio(orig_path, pert, sev, aug_path, seed=42 + i)

                    row = orig.to_dict()
                    row["audio_path"] = str(aug_path).replace("\\", "/")
                    selected_rows.append(row)
                except Exception as e:
                    print(f"      ⚠️  Augmentation échouée : {e}")
                    selected_rows.append(orig.to_dict())

        selected = pd.DataFrame(selected_rows)
        selected["target_category"] = f"{lang}_{pert}_{sev}"
        all_selected.append(selected)

    if len(all_selected) == 0:
        raise ValueError("❌ Aucun échantillon sélectionné")

    # Concaténer et mélanger
    df_train = pd.concat(all_selected, ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    n_total = len(df_train)
    n_val_actual = min(n_val, int(n_total * 0.1))
    df_val = df_train.head(n_val_actual)
    df_train = df_train.iloc[n_val_actual:]

    print(f"\n📊 Dataset final :")
    print(f"   Train : {len(df_train)}")
    print(f"   Val   : {len(df_val)}")
    print(f"   Total : {n_total}")

    # Préparer format Whisper (chemins absolus pour Colab)
    df_train_final = pd.DataFrame({
        "audio_path": df_train["audio_path"].apply(
            lambda p: str(Path(p).resolve()).replace("\\", "/")
        ),
        "transcript": df_train["reference"],
        "language": df_train["language"],
    })

    df_val_final = pd.DataFrame({
        "audio_path": df_val["audio_path"].apply(
            lambda p: str(Path(p).resolve()).replace("\\", "/")
        ),
        "transcript": df_val["reference"],
        "language": df_val["language"],
    })

    train_path = output_dir / "train_manifest.tsv"
    val_path = output_dir / "val_manifest.tsv"

    df_train_final.to_csv(train_path, sep="\t", index=False, encoding="utf-8")
    df_val_final.to_csv(val_path, sep="\t", index=False, encoding="utf-8")

    print(f"\n💾 {train_path} ({len(df_train_final)})")
    print(f"💾 {val_path} ({len(df_val_final)})")

    # Distribution
    print(f"\n📊 Distribution finale :")
    print(df_train_final.groupby("language").size().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="data/finetuning")
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=200)
    args = parser.parse_args()

    build_dataset(
        source_csv=Path(args.source),
        output_dir=Path(args.output),
        n_train=args.n_train,
        n_val=args.n_val,
    )


if __name__ == "__main__":
    main()