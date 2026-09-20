"""
Calcul des métriques d'évaluation pour la robustesse des Speech-LLMs :
    - WER (Word Error Rate)
    - CER (Character Error Rate)
    - Latence d'inférence

Usage :
    python -m src.evaluation.metrics --model whisper --data data/processed \
        --output results/tables/ --manifest data/processed/manifest.tsv
"""

import argparse
import csv
import time
from pathlib import Path

import jiwer
import pandas as pd

from src.models.model_wrappers import load_model, MODEL_REGISTRY
from src.utils.io import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """Word Error Rate (0.0 = parfait)."""
    return jiwer.wer(references, hypotheses)


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    """Character Error Rate (0.0 = parfait)."""
    return jiwer.cer(references, hypotheses)


# ---------------------------------------------------------------------------
# Chargement des références
# ---------------------------------------------------------------------------

def load_references(manifest_path: Path) -> dict[str, str]:
    """
    Charge un manifest TSV avec colonnes 'audio_path' et 'transcript'.
    Renvoie un dict {chemin_normalisé: transcription}.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable : {manifest_path}")

    df = pd.read_csv(manifest_path, sep="\t")
    if "audio_path" not in df.columns or "transcript" not in df.columns:
        raise ValueError("Le manifest doit contenir les colonnes 'audio_path' et 'transcript'.")

    refs = {}
    for _, row in df.iterrows():
        # Normaliser les chemins (forward slashes partout)
        key = str(row["audio_path"]).replace("\\", "/").lower()
        refs[key] = str(row["transcript"])
    return refs


def _normalize_path(p: Path) -> str:
    return str(p).replace("\\", "/").lower()


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    language: str | None = None,
) -> dict:
    """
    Évalue un modèle sur les WAV listés dans le manifest.

    Returns:
        dict résumé : {model, wer, cer, avg_latency_s, n_samples}
    """
    logger.info(f"=== Évaluation de '{model_name}' ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    references_map = load_references(manifest_path)
    logger.info(f"{len(references_map)} références chargées depuis {manifest_path}")

    model = load_model(model_name)

    references: list[str] = []
    hypotheses: list[str] = []
    latencies: list[float] = []
    paths_used: list[str] = []

    for wav_path in sorted(data_dir.glob("**/*.wav")):
        key = _normalize_path(wav_path)
        if key not in references_map:
            logger.debug(f"Pas de référence pour {wav_path}, ignoré.")
            continue

        ref = references_map[key]
        start = time.perf_counter()
        try:
            hyp = model.transcribe(wav_path, language=language)
        except Exception as e:
            logger.error(f"Erreur sur {wav_path} : {e}")
            hyp = ""
        latency = time.perf_counter() - start

        references.append(ref)
        hypotheses.append(hyp)
        latencies.append(latency)
        paths_used.append(str(wav_path))

        logger.info(f"  {wav_path.name:35s} | {latency:5.2f}s | ref={ref!r} hyp={hyp!r}")

    if not references:
        logger.error("Aucune référence trouvée — arrêt.")
        return {}

    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)
    avg_latency = sum(latencies) / len(latencies)

    logger.info(
        f"→ {model_name} : WER={wer:.3f} | CER={cer:.3f} | "
        f"Latence moy.={avg_latency:.2f}s | N={len(references)}"
    )

    # --- CSV détaillé par modèle ---
    detail_csv = output_dir / f"{model_name}_results.csv"

    languages = [
        Path(p).stem.split("_")[1] if "_" in Path(p).stem else "?"
        for p in paths_used
    ]

    pd.DataFrame({
        "audio_path": paths_used,
        "reference": references,
        "hypothesis": hypotheses,
        "latency_s": latencies,
        "language": languages,     # ← NOUVELLE COLONNE
    }).to_csv(detail_csv, index=False, encoding="utf-8")
    logger.info(f"Détail sauvegardé : {detail_csv}")

    # --- Summary global (append) ---
    summary_csv = output_dir / "summary.csv"
    row = {
        "model": model_name,
        "language": language or "auto",
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "avg_latency_s": round(avg_latency, 3),
        "n_samples": len(references),
    }
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Summary mis à jour : {summary_csv}")

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation des Speech-LLMs.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data", required=True, help="Dossier contenant les WAV à évaluer")
    parser.add_argument("--manifest", required=True, help="Chemin du manifest TSV")
    parser.add_argument("--output", required=True, help="Dossier de sortie des résultats")
    parser.add_argument("--language", default=None, help="Code langue (ex: 'fr', 'en', 'ar')")
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        manifest_path=Path(args.manifest),
        language=args.language,
    )


if __name__ == "__main__":
    main()
    