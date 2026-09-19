"""
Chargement et préparation des jeux de données audio pour l'évaluation
de la robustesse des Speech-LLMs.

Trois modes disponibles :

    1. --mode local       : utilise un dossier de WAV + un manifest existant
    2. --mode synthetic   : génère des WAV de test (TTS local via pyttsx3 ou sinus)
    3. --mode streaming   : télécharge FLEURS via HuggingFace (avec retry + limit)

Usage :
    # Mode local
    python -m src.data.load_datasets --mode local \\
        --input data/raw --output data/processed

    # Mode synthetic
    python -m src.data.load_datasets --mode synthetic \\
        --output data/raw --n 2 --languages en,fr,ar

    # Mode streaming (nécessite internet)
    python -m src.data.load_datasets --mode streaming \\
        --dataset fleurs --config en_us --output data/raw --n 2
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.utils.io import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SUPPORTED_MODES = ["local", "synthetic", "streaming"]
SUPPORTED_DATASETS = ["fleurs", "librispeech", "common_voice"]

LANG_CONFIGS = {
    # langue -> (nom dataset, config, code ISO)
    "en": ("google/fleurs", "en_us", "en"),
    "fr": ("google/fleurs", "fr_fr", "fr"),
    "ar": ("google/fleurs", "ar_eg", "ar"),
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Sauvegarde un WAV en 16 kHz mono."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Rééchantillonner si nécessaire
    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except ImportError:
            logger.warning(f"librosa absent, conservation sr={sr} pour {path.name}")
            sf.write(str(path), audio, sr)
            return
        sr = 16000
    sf.write(str(path), audio, sr)


def _generate_sine(duration_s: float = 2.0, freq_hz: float = 440.0,
                   sr: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _generate_speech_tts(text: str, output_path: Path, lang: str = "en") -> bool:
    """Génère un WAV avec la voix correspondant à la langue."""
    try:
        import pyttsx3
    except ImportError:
        logger.warning("pyttsx3 non installé, fallback sinus.")
        return False

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")

        # Mapping langue → substring à chercher dans voice.languages[0] ou voice.name
        lang_patterns = {
            "en": ["en-US", "en-GB", "english"],
            "fr": ["fr-FR", "fr-CA", "french"],
            "ar": ["ar-SA", "ar-EG", "arabic"],
        }
        patterns = lang_patterns.get(lang, ["en-US"])

        selected = None
        for voice in voices:
            voice_langs = " ".join(str(l) for l in voice.languages).lower()
            voice_name = voice.name.lower()
            for pat in patterns:
                if pat.lower() in voice_langs or pat.lower() in voice_name:
                    selected = voice
                    break
            if selected:
                break

        if selected is None:
            logger.warning(f"  → Aucune voix {lang} trouvée, voix par défaut")
        else:
            engine.setProperty("voice", selected.id)
            logger.info(f"  → Voix sélectionnée : {selected.name}")

        engine.setProperty("rate", 150)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

        if not output_path.exists() or output_path.stat().st_size == 0:
            logger.warning(f"pyttsx3 n'a pas généré {output_path}")
            return False

        # Rééchantillonner en 16 kHz
        audio, sr = sf.read(str(output_path))
        _save_wav(output_path, audio, sr)
        return True
    except Exception as e:
        logger.warning(f"Erreur pyttsx3 : {e}")
        return False


# ---------------------------------------------------------------------------
# Mode 1 — LOCAL
# ---------------------------------------------------------------------------

def load_local(input_dir: Path, output_dir: Path, language: str | None = None) -> None:
    """
    Copie/normalise les WAV d'un dossier local vers output_dir, et copie le manifest.
    Utile pour préparer les données avant évaluation.
    """
    logger.info(f"[LOCAL] Copie de {input_dir} → {output_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Dossier source introuvable : {input_dir}")

    wav_files = list(input_dir.glob("**/*.wav"))
    if not wav_files:
        logger.warning(f"Aucun WAV dans {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for wav in wav_files:
        # Charger + rééchantillonner en 16 kHz
        audio, sr = sf.read(str(wav), always_2d=False)
        dest = output_dir / wav.name
        _save_wav(dest, audio, sr)
        n += 1

    logger.info(f"[LOCAL] {n} WAV traités dans {output_dir}")

    # Copier le manifest s'il existe
    manifest_src = input_dir / "manifest.tsv"
    if manifest_src.exists():
        manifest_dst = output_dir / "manifest.tsv"
        manifest_dst.write_text(manifest_src.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"[LOCAL] Manifest copié : {manifest_dst}")


# ---------------------------------------------------------------------------
# Mode 2 — SYNTHETIC
# ---------------------------------------------------------------------------

SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "fr": "Le vif renard brun saute par-dessus le chien paresseux.",
    "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
}


def load_synthetic(output_dir: Path, n: int = 2, languages: list[str] | None = None) -> None:
    """
    Génère des WAV de test sans téléchargement :
        - Tente TTS local (pyttsx3) si disponible
        - Sinon, génère des sinus de fréquences différentes
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = languages or list(SENTENCES.keys())

    logger.info(f"[SYNTHETIC] Génération de {n} échantillons × {len(languages)} langues")

    rows = []
    for lang in languages:
        if lang not in SENTENCES:
            logger.warning(f"Langue '{lang}' non supportée, ignorée.")
            continue

        for i in range(n):
            out_path = output_dir / f"synth_{lang}_{i:02d}.wav"
            text = SENTENCES[lang]

            # Essayer TTS, sinon sinus
            success = _generate_speech_tts(text, out_path, lang=lang)
            if not success:
                # Fréquences différentes par langue pour distinguer
                freqs = {"en": 440.0, "fr": 523.25, "ar": 659.25}
                audio = _generate_sine(freq_hz=freqs.get(lang, 440.0))
                _save_wav(out_path, audio, 16000)

            rows.append({
                "audio_path": str(out_path).replace("\\", "/"),
                "transcript": text,
                "language": lang,
            })
            logger.info(f"  ✓ {out_path.name}")

    # Écrire un manifest
    manifest_path = output_dir / "manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False, encoding="utf-8")
    logger.info(f"[SYNTHETIC] Manifest écrit : {manifest_path} ({len(rows)} entrées)")


# ---------------------------------------------------------------------------
# Mode 3 — STREAMING (FLEURS)
# ---------------------------------------------------------------------------

def _stream_fleurs(config: str, n: int, max_retries: int = 3) -> list[dict]:
    """
    Télécharge n échantillons FLEURS en streaming, avec retry.
    Retourne une liste de dicts {audio_array, sampling_rate, transcript}.
    """
    from datasets import load_dataset, Audio

    samples = []
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentative {attempt}/{max_retries} pour FLEURS/{config}...")
            ds = load_dataset("google/fleurs", config, split="test", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))

            for i, s in enumerate(ds.take(n)):
                audio_bytes = s["audio"]["bytes"]
                array, sr = sf.read(io.BytesIO(audio_bytes))
                samples.append({
                    "audio_array": array,
                    "sampling_rate": sr,
                    "transcript": s.get("transcription") or s.get("raw_transcription", ""),
                })
            return samples
        except Exception as e:
            logger.warning(f"Erreur tentative {attempt} : {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info(f"Attente {wait}s avant retry...")
                time.sleep(wait)

    raise RuntimeError(f"Échec après {max_retries} tentatives pour FLEURS/{config}")


def load_streaming(
    dataset: str,
    config: str,
    output_dir: Path,
    n: int = 2,
    language: str | None = None,
) -> None:
    """Télécharge un dataset via HuggingFace en mode streaming."""
    if dataset != "fleurs":
        raise NotImplementedError(f"Dataset '{dataset}' non supporté en mode streaming (seul FLEURS pour l'instant).")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[STREAMING] FLEURS/{config} — {n} échantillons")

    samples = _stream_fleurs(config, n)
    rows = []
    for i, s in enumerate(samples):
        out_path = output_dir / f"{config}_{i:03d}.wav"
        _save_wav(out_path, s["audio_array"], s["sampling_rate"])
        rows.append({
            "audio_path": str(out_path).replace("\\", "/"),
            "transcript": s["transcript"],
            "language": language or config.split("_")[0],
        })
        logger.info(f"  ✓ {out_path.name} → {s['transcript'][:60]}")

    # Manifest append (ne pas écraser celui du mode synthetic)
    manifest_path = output_dir / "manifest.tsv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, sep="\t")
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        combined.to_csv(manifest_path, sep="\t", index=False, encoding="utf-8")
    else:
        pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False, encoding="utf-8")

    logger.info(f"[STREAMING] Manifest mis à jour : {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chargement des jeux de données audio (local / synthetic / streaming)."
    )
    parser.add_argument("--mode", required=True, choices=SUPPORTED_MODES,
                        help="Mode de chargement")
    parser.add_argument("--input", default=None,
                        help="[local] Dossier source des WAV")
    parser.add_argument("--output", required=True,
                        help="Dossier de sortie")
    parser.add_argument("--dataset", default="fleurs", choices=SUPPORTED_DATASETS,
                        help="[streaming] Dataset HF")
    parser.add_argument("--config", default=None,
                        help="[streaming] Config du dataset (ex: en_us)")
    parser.add_argument("--n", type=int, default=2,
                        help="[synthetic/streaming] Nombre d'échantillons par langue")
    parser.add_argument("--languages", default="en,fr,ar",
                        help="[synthetic] Langues séparées par virgule (en,fr,ar)")
    parser.add_argument("--language", default=None,
                        help="[streaming] Code langue ISO (en, fr, ar)")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.mode == "local":
        if not args.input:
            parser.error("--input requis en mode local")
        load_local(Path(args.input), output_dir, language=args.language)

    elif args.mode == "synthetic":
        langs = [x.strip() for x in args.languages.split(",") if x.strip()]
        load_synthetic(output_dir, n=args.n, languages=langs)

    elif args.mode == "streaming":
        if not args.config:
            parser.error("--config requis en mode streaming (ex: en_us)")
        load_streaming(
            dataset=args.dataset,
            config=args.config,
            output_dir=output_dir,
            n=args.n,
            language=args.language,
        )


if __name__ == "__main__":
    main()