"""
Fine-tuning ciblé des Speech-LLMs avec LoRA (Low-Rank Adaptation).

Supporte : Whisper, Wav2Vec2, Qwen2-Audio.
Le fine-tuning est param-efficient : on n'entraîne que ~1-2 % des paramètres.

Usage :
    python -m src.finetuning.finetune --config configs/finetune_whisper.yaml
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.utils.io import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Charge un fichier YAML de configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator pour Whisper : pad les inputs et les labels."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Retirer le token de début si présent
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class SpeechDataset(Dataset):
    """Dataset audio + transcription à partir d'un manifest TSV."""

    def __init__(self, manifest_path: Path, processor, language: str, sr: int = 16000):
        self.df = pd.read_csv(manifest_path, sep="\t")
        self.processor = processor
        self.language = language
        self.sr = sr

        if "audio_path" not in self.df.columns or "transcript" not in self.df.columns:
            raise ValueError("Le manifest doit contenir 'audio_path' et 'transcript'")

        # Filtrer par langue si colonne présente
        if "language" in self.df.columns:
            self.df = self.df[self.df["language"] == language].reset_index(drop=True)

        logger.info(f"Dataset : {len(self.df)} échantillons (langue={language})")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Charger l'audio
        audio, sr = sf.read(str(row["audio_path"]), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sr:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.sr)

        # Features
        inputs = self.processor(
            audio,
            sampling_rate=self.sr,
            return_tensors="np",
        )
        input_features = inputs.input_features[0]

        # Labels
        labels = self.processor.tokenizer(row["transcript"]).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Modèle + LoRA
# ---------------------------------------------------------------------------

def build_lora_model(config: dict):
    """Charge Whisper + applique LoRA sur la self-attention du décodeur uniquement."""
    from peft import LoraConfig, get_peft_model, TaskType

    model_name = config["model"]["name"]
    logger.info(f"Chargement de {model_name}...")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Configurer le modèle pour la génération
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = config["model"]["language"]
    model.generation_config.task = config["model"]["task"]
    model.generation_config.forced_decoder_ids = None

    if config["lora"]["enabled"]:
        logger.info(f"Application de LoRA (r={config['lora']['r']})...")

        # 🔧 Filtrer les modules cibles : UNIQUEMENT self_attn du décodeur
        target_projs = config["lora"]["target_modules"]  # ["q_proj", "v_proj", "k_proj", "out_proj"]
        target_modules = []

        for name, _ in model.named_modules():
            # Cibler uniquement les projections voulues
            if not any(name.endswith(f".{proj}") for proj in target_projs):
                continue
            # Exclure le cross-attention (encoder_attn) — conflit input_ids
            if "encoder_attn" in name:
                continue
            # Exclure l'encodeur (on ne fine-tune que le décodeur)
            if "model.encoder" in name:
                continue
            target_modules.append(name)

        logger.info(f"Modules LoRA ciblés : {len(target_modules)}")
        for m in target_modules[:5]:
            logger.info(f"  • {m}")
        if len(target_modules) > 5:
            logger.info(f"  ... et {len(target_modules) - 5} autres")

        if not target_modules:
            raise ValueError(
                "Aucun module LoRA trouvé. Vérifiez `target_modules` dans le YAML."
            )

        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=config["lora"]["lora_dropout"],
            bias=config["lora"]["bias"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(config: dict) -> dict:
    """Boucle principale de fine-tuning."""
    from src.models.model_wrappers import load_model

    model, processor = build_lora_model(config)

    # Dataset
    manifest = Path(config["data"]["manifest"])
    lang = config["model"]["language"]

    full_dataset = SpeechDataset(manifest, processor, language=lang)

    if len(full_dataset) < 2:
        raise ValueError(f"Pas assez d'échantillons pour la langue '{lang}' ({len(full_dataset)})")

    # Split train/eval
    eval_size = max(1, int(len(full_dataset) * config["data"].get("eval_split", 0.2)))
    train_size = len(full_dataset) - eval_size

    train_ds, eval_ds = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Split : train={train_size}, eval={eval_size}")

    # Collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Argument d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        max_steps=config["training"].get("max_steps", -1),
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=config["training"]["fp16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        report_to=config["training"].get("report_to", "none"),
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    logger.info("=== Démarrage du fine-tuning ===")
    train_result = trainer.train()
    logger.info(f"=== Fine-tuning terminé : {train_result.metrics} ===")

    # Sauvegarde finale
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    # Sauvegarder les métriques
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)
    logger.info(f"Métriques sauvegardées : {metrics_path}")

    return train_result.metrics


# ---------------------------------------------------------------------------
# Évaluation avant/après
# ---------------------------------------------------------------------------

def evaluate_before_after(config: dict) -> dict:
    """
    Compare les performances AVANT et APRÈS fine-tuning sur le même manifest.
    Détecte automatiquement si le modèle fine-tuné est LoRA ou standard.
    """
    import jiwer
    from src.models.model_wrappers import load_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    manifest = Path(config["evaluation"]["manifest"])
    output_dir = Path(config["evaluation"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest, sep="\t")
    if "language" in df.columns:
        df = df[df["language"] == config["model"]["language"]].reset_index(drop=True)

    if len(df) == 0:
        logger.warning("Aucun échantillon à évaluer.")
        return {}

    base_model_name = config["model"]["name"]
    finetuned_path = Path(config["training"]["output_dir"]) / "final"
    lang = config["model"]["language"]

    # --- AVANT ---
    logger.info("=== Évaluation AVANT fine-tuning ===")
    base_model = load_model("whisper", device="cpu")
    base_model.model.generation_config.language = lang
    base_model.model.generation_config.task = config["model"]["task"]

    refs, hyps_before = [], []
    for _, row in df.iterrows():
        ref = row["transcript"]
        hyp = base_model.transcribe(Path(row["audio_path"]), language=lang)
        refs.append(ref)
        hyps_before.append(hyp)

    wer_before = jiwer.wer(refs, hyps_before)
    cer_before = jiwer.cer(refs, hyps_before)
    logger.info(f"AVANT : WER={wer_before:.4f} | CER={cer_before:.4f}")

    # --- APRÈS ---
    logger.info("=== Évaluation APRÈS fine-tuning ===")
    if not finetuned_path.exists():
        logger.warning(f"Modèle fine-tuné introuvable : {finetuned_path}")
        return {"before": {"wer": wer_before, "cer": cer_before}, "after": None}

    # Charger selon LoRA ou standard
    processor = WhisperProcessor.from_pretrained(str(finetuned_path))

    if config["lora"]["enabled"]:
        # Charger base + adapter LoRA
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base, str(finetuned_path))
        logger.info("Modèle LoRA chargé")
    else:
        # Charger directement le modèle fine-tuné
        model = WhisperForConditionalGeneration.from_pretrained(str(finetuned_path))
        logger.info("Modèle standard fine-tuné chargé")

    model.eval()
    model.generation_config.language = lang
    model.generation_config.task = config["model"]["task"]
    model.generation_config.forced_decoder_ids = None

    hyps_after = []
    for _, row in df.iterrows():
        audio, sr = sf.read(str(row["audio_path"]), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                language=lang,
                task=config["model"]["task"],
                max_new_tokens=128,
            )
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        hyps_after.append(hyp)

    wer_after = jiwer.wer(refs, hyps_after)
    cer_after = jiwer.cer(refs, hyps_after)
    logger.info(f"APRÈS : WER={wer_after:.4f} | CER={cer_after:.4f}")

    # --- Résumé ---
    summary = {
        "language": lang,
        "model": base_model_name,
        "lora_enabled": config["lora"]["enabled"],
        "n_samples": len(df),
        "before": {"wer": round(wer_before, 4), "cer": round(cer_before, 4)},
        "after": {"wer": round(wer_after, 4), "cer": round(cer_after, 4)},
        "improvement": {
            "wer_abs": round(wer_before - wer_after, 4),
            "wer_rel_pct": round((wer_before - wer_after) / wer_before * 100, 2) if wer_before > 0 else 0.0,
            "cer_abs": round(cer_before - cer_after, 4),
            "cer_rel_pct": round((cer_before - cer_after) / cer_before * 100, 2) if cer_before > 0 else 0.0,
        },
    }

    with open(output_dir / "before_after.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Résumé sauvegardé : {output_dir / 'before_after.json'}")
    logger.info(
        f"Amélioration : WER {wer_before:.3f} → {wer_after:.3f} "
        f"({summary['improvement']['wer_rel_pct']:+.1f}%)"
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA des Speech-LLMs.")
    parser.add_argument("--config", required=True, help="Fichier YAML de configuration")
    parser.add_argument("--eval-only", action="store_true",
                        help="Évaluer avant/après sans entraîner")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.eval_only:
        evaluate_before_after(config)
    else:
        metrics = finetune(config)
        evaluate_before_after(config)


if __name__ == "__main__":
    main()