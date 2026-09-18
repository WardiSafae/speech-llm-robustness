"""
Fine-tuning ciblé des Speech-LLMs sur les conditions difficiles identifiées
lors de la phase d'évaluation (bruit, accents problématiques).

Usage :
    python -m src.finetuning.finetune --model whisper --config configs/finetune_whisper.yaml
"""

import argparse
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def finetune(model_name: str, config: dict) -> None:
    # TODO: implémenter la boucle de fine-tuning (Hugging Face Trainer ou boucle manuelle)
    raise NotImplementedError(f"Fine-tuning de '{model_name}' à implémenter avec la config : {config}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning ciblé d'un Speech-LLM.")
    parser.add_argument("--model", required=True, help="Nom du modèle à fine-tuner")
    parser.add_argument("--config", required=True, help="Fichier de configuration YAML")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    finetune(args.model, config)


if __name__ == "__main__":
    main()
