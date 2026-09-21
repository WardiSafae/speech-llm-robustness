"""
Fine-tuning de whisper-large-v3 avec LoRA.

Résout le bug 'multiple values for input_ids' en filtrant les modules.

Usage :
    python scripts/finetune_whisper.py --config configs/finetune_large_v3.yaml
"""

import argparse
from pathlib import Path

import yaml
import torch
import soundfile as sf
import numpy as np
import librosa
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.io import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def prepare_dataset(manifest_path: Path, processor, language: str = None):
    """Charge le manifest et prépare les features Whisper."""
    import pandas as pd

    df = pd.read_csv(manifest_path, sep="\t")
    if language:
        df = df[df["language"] == language].reset_index(drop=True)

    logger.info(f"Dataset : {len(df)} échantillons")

    def prepare_example(example):
        audio, sr = sf.read(example["audio_path"], always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)

        inputs = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        )
        example["input_features"] = inputs.input_features[0]

        labels = processor.tokenizer(example["transcript"]).input_ids
        example["labels"] = labels
        return example

    ds = Dataset.from_pandas(df)
    ds = ds.map(prepare_example, remove_columns=ds.column_names)
    return ds


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# LoRA config (résolution du bug)
# ---------------------------------------------------------------------------

def build_lora_model(config):
    """
    Charge whisper-large-v3 et applique LoRA sur la self-attention du décodeur.
    
    Résout le bug 'multiple values for input_ids' en excluant encoder_attn.
    """
    model_name = config["model"]["name"]
    logger.info(f"Chargement de {model_name}...")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Configurer pour la génération
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = config["model"].get("language", "fr")
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if config["lora"]["enabled"]:
        # Filtrer les modules : UNIQUEMENT self_attn du décodeur
        target_projs = config["lora"]["target_modules"]
        target_modules = []

        for name, _ in model.named_modules():
            if not any(name.endswith(f".{proj}") for proj in target_projs):
                continue
            if "encoder_attn" in name:  # ← Éviter le cross-attention
                continue
            if "model.encoder" in name:  # ← Éviter l'encodeur
                continue
            target_modules.append(name)

        logger.info(f"Modules LoRA ciblés : {len(target_modules)}")
        for m in target_modules[:3]:
            logger.info(f"   • {m}")

        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=target_modules,
            lora_dropout=config["lora"]["dropout"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Modèle + LoRA
    model, processor = build_lora_model(config)
    model = model.to("cuda")

    # Datasets
    train_ds = prepare_dataset(Path(config["data"]["train_manifest"]), processor)
    val_ds = prepare_dataset(Path(config["data"]["val_manifest"]), processor)

    # Collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        num_train_epochs=config["training"]["epochs"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    # Entraînement
    logger.info("=== Démarrage du fine-tuning ===")
    trainer.train()

    # Sauvegarde
    output_dir = Path(config["training"]["output_dir"]) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info(f"✅ Modèle sauvegardé dans {output_dir}")


if __name__ == "__main__":
    main()
    