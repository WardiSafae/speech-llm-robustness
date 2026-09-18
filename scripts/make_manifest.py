"""Génère un manifest.tsv à partir des WAV dans un dossier."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", default="data/processed")
    parser.add_argument("--output", default="data/processed/manifest.tsv")
    parser.add_argument("--default_transcript", default="")
    args = parser.parse_args()

    rows = []
    for wav in sorted(Path(args.wav_dir).glob("*.wav")):
        rows.append({
            "audio_path": str(wav).replace("\\", "/"),  # forward slashes pour portabilité
            "transcript": args.default_transcript,
        })

    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"✅ {len(rows)} entrées écrites dans {args.output}")


if __name__ == "__main__":
    main()