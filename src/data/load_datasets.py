"""
Chargement et préparation des jeux de données audio utilisés pour l'évaluation
de la robustesse des Speech-LLMs.

Jeux de données ciblés :
    - Common Voice (Mozilla)
    - LibriSpeech
    - FLEURS
    - Arabic Speech Corpus

Usage :
    python -m src.data.load_datasets --dataset common_voice --lang fr,ar,en
"""

import argparse
from pathlib import Path


SUPPORTED_DATASETS = ["common_voice", "librispeech", "fleurs", "arabic_speech_corpus"]


def load_dataset(name: str, langs: list[str], output_dir: Path) -> None:
    """
    Télécharge et prépare un jeu de données audio.

    Args:
        name: nom du jeu de données (voir SUPPORTED_DATASETS)
        langs: liste des codes de langue à récupérer
        output_dir: dossier de sortie pour les données brutes
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{name}' non supporté. Choix possibles : {SUPPORTED_DATASETS}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implémenter le téléchargement via `datasets` (Hugging Face) ou API dédiée
    raise NotImplementedError(
        f"Le chargement de '{name}' pour les langues {langs} reste à implémenter."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chargement des jeux de données audio.")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--lang", required=True, help="Codes de langue séparés par des virgules, ex: fr,ar,en")
    parser.add_argument("--output", default="data/raw", help="Dossier de sortie")
    args = parser.parse_args()

    langs = args.lang.split(",")
    load_dataset(args.dataset, langs, Path(args.output))


if __name__ == "__main__":
    main()
