"""
Test minimal de fine-tuning Whisper-tiny sans LoRA.
Valide juste que la boucle fonctionne.
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"  # éviter Xet

from pathlib import Path
# Ajouter la racine du projet au PYTHONPATH
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from src.finetuning.finetune import SpeechDataset, DataCollatorSpeechSeq2SeqWithPadding


def main():
    model_name = "openai/whisper-tiny"
    print(f"Chargement de {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "fr"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Gradient checkpointing (économie RAM)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Dataset
    dataset = SpeechDataset(Path("data/raw/manifest.tsv"), processor, language="fr")
    if len(dataset) < 2:
        print(f"❌ Pas assez d'échantillons ({len(dataset)})")
        return

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/whisper-tiny-test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_steps=5,
        logging_steps=1,
        save_strategy="no",              # ← PAS de sauvegarde
        eval_strategy="no",              # ← PAS d'éval
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        predict_with_generate=False,     # ← désactiver pour aller vite
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    print("=== Démarrage du test ===")
    result = trainer.train()
    print(f"✅ Fine-tuning OK : {result.metrics}")


if __name__ == "__main__":
    main()