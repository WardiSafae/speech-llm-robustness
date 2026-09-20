"""
Génération de perturbations acoustiques contrôlées pour évaluer la robustesse
des Speech-LLMs.

Perturbations supportées :
    - bruit blanc (white_noise)
    - bruit de rue (street_noise)
    - réverbération (reverb)
    - clipping (clipping)
    - variation de vitesse (speed)

Usage :
    python -m src.augmentation.augment --input data/raw --output data/processed \
        --types noise,reverb,clipping,speed
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from src.utils.io import setup_logger

logger = setup_logger(__name__)

SUPPORTED_PERTURBATIONS = ["white_noise", "street_noise", "reverb", "clipping", "speed"]

ALIASES = {
    "noise": "white_noise",     # alias pour compatibilité CLI
    "noise_white": "white_noise",
    "noise_street": "street_noise",
}


def _white_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape)
    return audio + noise


def _street_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    # bruit rose approximé (1/f) comme proxy du bruit de rue
    white = np.random.normal(0, 1, audio.shape)
    pink = np.cumsum(white)
    pink = pink / (np.max(np.abs(pink)) + 1e-9)
    rms_signal = np.sqrt(np.mean(audio ** 2))
    pink = pink * rms_signal / (10 ** (snr_db / 20))
    return audio + pink


def _reverb(audio: np.ndarray, sr: int, room_scale: float = 0.3) -> np.ndarray:
    ir = np.random.randn(int(sr * room_scale)) * np.exp(-np.linspace(0, 6, int(sr * room_scale)))
    return np.convolve(audio, ir, mode="same")[: len(audio)]


def _clipping(audio: np.ndarray, threshold_ratio: float = 0.7) -> np.ndarray:
    """
    Simule un clipping par saturation : les pics > threshold_ratio * max
    sont écrêtés (aplatis).
    """
    threshold = threshold_ratio * np.max(np.abs(audio))
    return np.clip(audio, -threshold, threshold)


def _speed(audio: np.ndarray, sr: int, rate: float = 1.1) -> tuple[np.ndarray, int]:
    return librosa.effects.time_stretch(audio, rate=rate), sr


def apply_perturbation(audio_path: Path, perturbation: str, output_dir: Path) -> Path:
    perturbation = ALIASES.get(perturbation, perturbation)
    if perturbation not in SUPPORTED_PERTURBATIONS:
        raise ValueError(f"Perturbation '{perturbation}' non supportée. Choix : {SUPPORTED_PERTURBATIONS}")

    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    if perturbation == "white_noise":
        out = _white_noise(audio)
    elif perturbation == "street_noise":
        out = _street_noise(audio)
    elif perturbation == "reverb":
        out = _reverb(audio, sr)
    elif perturbation == "clipping":
        out = _clipping(audio)
    elif perturbation == "speed":
        out, sr = _speed(audio, sr)
    else:
        raise ValueError(perturbation)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}__{perturbation}.wav"
    sf.write(str(out_path), out, sr)
    return out_path

def apply_perturbation_with_severity(
    audio_path: Path,
    perturbation: str,
    severity: float,
    output_dir: Path,
) -> Path:
    """
    Applique une perturbation avec un niveau de sévérité contrôlé.

    Args:
        audio_path: fichier audio source
        perturbation: white_noise, street_noise, reverb, clipping, speed
        severity: 
            - white_noise / street_noise : SNR en dB (5, 10, 15)
            - reverb : room_scale (0.1, 0.3, 0.5)
            - clipping : threshold_ratio (0.7, 0.5, 0.3)
            - speed : rate (0.9, 1.0, 1.1)
        output_dir: dossier de sortie
    """
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    if perturbation in ("white_noise", "street_noise"):
        snr_db = severity
        if perturbation == "white_noise":
            out = _white_noise(audio, snr_db=snr_db)
        else:
            out = _street_noise(audio, snr_db=snr_db)
        suffix = f"snr{int(snr_db)}"

    elif perturbation == "reverb":
        room_scale = severity
        out = _reverb(audio, sr, room_scale=room_scale)
        suffix = f"room{room_scale}"

    elif perturbation == "clipping":
        threshold_ratio = severity
        out = _clipping(audio, threshold_ratio=threshold_ratio)
        suffix = f"clip{threshold_ratio}"

    elif perturbation == "speed":
        rate = severity
        out, sr = _speed(audio, sr, rate=rate)
        suffix = f"spd{rate}"

    else:
        raise ValueError(f"Perturbation '{perturbation}' non supportée.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}__{perturbation}_{suffix}.wav"
    sf.write(str(out_path), out, sr)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Génération de perturbations acoustiques.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--types", required=True)
    parser.add_argument("--severity-levels", action="store_true",
                        help="Appliquer 3 niveaux de sévérité par perturbation")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    perturbations = [p.strip() for p in args.types.split(",")]

    # Grille de sévérité par perturbation
    SEVERITY_GRID = {
        "white_noise": [5.0, 10.0, 15.0],       # SNR dB
        "street_noise": [5.0, 10.0, 15.0],
        "reverb": [0.1, 0.3, 0.5],              # room scale
        "clipping": [0.7, 0.5, 0.3],            # threshold ratio
        "speed": [0.9, 1.0, 1.1],               # rate
    }

    if args.severity_levels:
        for audio_file in input_dir.glob("**/*.wav"):
            for p in perturbations:
                for sev in SEVERITY_GRID.get(p, []):
                    try:
                        out = apply_perturbation_with_severity(audio_file, p, sev, output_dir)
                        logger.info(f"✓ {out.name}")
                    except Exception as e:
                        logger.error(f"✗ {audio_file.name} [{p}={sev}] : {e}")
                        
    n = 0
    for audio_file in input_dir.glob("**/*.wav"):
        for p in perturbations:
            try:
                out = apply_perturbation(audio_file, p, output_dir)
                logger.info(f"✓ {out.name}")
                n += 1
            except Exception as e:
                logger.error(f"✗ {audio_file.name} [{p}] : {e}")
    logger.info(f"{n} fichiers perturbés générés dans {output_dir}")


if __name__ == "__main__":
    main()