"""
Compare les performances de plusieurs modèles Whisper sur le même manifest.

Usage :
    python scripts/compare_models.py --manifest data/raw/manifest_neural.tsv
"""
import argparse
from pathlib import Path
import pandas as pd
import jiwer


MODELS = {
    "whisper-small":  "whisper_results.csv",
    "whisper-base":   "whisper_base_results.csv",
    "whisper-medium": "whisper_medium_results.csv",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/raw/manifest_neural.tsv")
    parser.add_argument("--results-dir", default="results/tables")
    args = parser.parse_args()

    # Charger le manifest pour filtrer
    manifest_df = pd.read_csv(args.manifest, sep="\t")
    neural_paths = set(manifest_df["audio_path"].str.replace("\\", "/"))
    print(f"Manifest : {args.manifest} ({len(neural_paths)} échantillons)\n")

    # Tableau
    header = f"{'Modèle':<18} {'Langue':<8} {'N':<4} {'WER':<10} {'CER':<10} {'Latence moy.':<12}"
    print(header)
    print("-" * len(header))

    results = {}
    for model_name, csv_name in MODELS.items():
        csv_path = Path(args.results_dir) / csv_name
        if not csv_path.exists():
            print(f"{model_name:<18} (fichier manquant: {csv_name})")
            continue

        df = pd.read_csv(csv_path)
        df["audio_path"] = df["audio_path"].str.replace("\\", "/")
        df = df[df["audio_path"].isin(neural_paths)]

        if "language" not in df.columns:
            df["language"] = df["audio_path"].apply(
                lambda p: Path(p).stem.split("_")[1]
            )

        for lang in ["en", "fr", "ar"]:
            sub = df[df["language"] == lang]
            if len(sub) == 0:
                continue
            wer = jiwer.wer(sub["reference"].tolist(), sub["hypothesis"].tolist())
            cer = jiwer.cer(sub["reference"].tolist(), sub["hypothesis"].tolist())
            lat = sub["latency_s"].mean()
            print(f"{model_name:<18} {lang:<8} {len(sub):<4} {wer:<10.4f} {cer:<10.4f} {lat:<12.2f}")

        # Global
        wer = jiwer.wer(df["reference"].tolist(), df["hypothesis"].tolist())
        cer = jiwer.cer(df["reference"].tolist(), df["hypothesis"].tolist())
        lat = df["latency_s"].mean()
        print(f"{model_name:<18} {'ALL':<8} {len(df):<4} {wer:<10.4f} {cer:<10.4f} {lat:<12.2f}")
        print()

        results[model_name] = {
            "global_wer": round(wer, 4),
            "global_cer": round(cer, 4),
            "global_latency": round(lat, 2),
        }

    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:<18} WER={r['global_wer']:.4f} | CER={r['global_cer']:.4f} | Lat={r['global_latency']:.2f}s")


if __name__ == "__main__":
    main()