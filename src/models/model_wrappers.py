"""
Wrappers unifiés pour les Speech-LLMs évalués dans ce projet, afin d'exposer
une interface commune (`.transcribe(audio_path) -> str`) quel que soit le
modèle sous-jacent.

Modèles ciblés : Whisper, Wav2Vec2-XLSR-53, Qwen2-Audio, SpeechT5.

Compatible transformers >= 5.0 (pas de forced_decoder_ids, pas de warnings).

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2AudioForConditionalGeneration,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.utils.io import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Utilitaires audio
# ---------------------------------------------------------------------------

def _load_audio(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Charge un WAV et le rééchantillonne si nécessaire (soundfile only)."""
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Interface de base
# ---------------------------------------------------------------------------

class SpeechLLMWrapper(ABC):
    """Interface commune pour tous les Speech-LLMs évalués."""

    MODEL_ID: str = ""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self._base_gen_config: Optional[GenerationConfig] = None

    @abstractmethod
    def load(self) -> None:
        """Charge les poids et le processor en mémoire."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: Path, language: Optional[str] = None, **kwargs) -> str:
        """Transcrit un fichier audio en texte."""
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

class WhisperWrapper(SpeechLLMWrapper):
    """Whisper (encodeur-décodeur, multilingue)."""

    MODEL_ID = "openai/whisper-small"

    def load(self) -> None:
        logger.info(f"Chargement de {self.MODEL_ID}...")
        self.processor = WhisperProcessor.from_pretrained(self.MODEL_ID)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()

        # Config de base sans forced_decoder_ids (déprécié en transformers v5)
        self._base_gen_config = GenerationConfig.from_pretrained(self.MODEL_ID)
        self._base_gen_config.forced_decoder_ids = None
        self._base_gen_config.begin_suppress_tokens = None
        self._base_gen_config.suppress_tokens = None

        logger.info("Modèle Whisper chargé ✓")

    def transcribe(self, audio_path: Path, language: Optional[str] = None,
                   task: str = "transcribe", max_new_tokens: int = 128,
                   **kwargs) -> str:
        self._ensure_loaded()
        audio = _load_audio(audio_path, target_sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)

        # Clone propre de la config
        gen_config = GenerationConfig.from_dict(self._base_gen_config.to_dict())
        gen_config.task = task
        if language is not None:
            gen_config.language = language
        gen_config.max_new_tokens = max_new_tokens

        with torch.no_grad():
            ids = self.model.generate(**inputs, generation_config=gen_config)

        return self.processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ---------------------------------------------------------------------------
# Wav2Vec2
# ---------------------------------------------------------------------------

class Wav2Vec2Wrapper(SpeechLLMWrapper):
    """Wav2Vec2-XLSR-53 (CTC, multilingue)."""

    MODEL_ID = "facebook/wav2vec2-large-xlsr-53-french"

    def load(self) -> None:
        logger.info(f"Chargement de {self.MODEL_ID}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()
        logger.info("Modèle Wav2Vec2 chargé ✓")

    def transcribe(self, audio_path: Path, language: Optional[str] = None, **kwargs) -> str:
        self._ensure_loaded()
        audio = _load_audio(audio_path, target_sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]


# ---------------------------------------------------------------------------
# Qwen2-Audio
# ---------------------------------------------------------------------------

class Qwen2AudioWrapper(SpeechLLMWrapper):
    """Qwen2-Audio 7B (Speech-LLM instructible). ⚠️ ~14 Go RAM en fp16."""

    MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

    def load(self) -> None:
        logger.info(f"Chargement de {self.MODEL_ID} (~14 Go RAM)...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16, device_map=self.device
        )
        self.model.eval()
        logger.info("Modèle Qwen2-Audio chargé ✓")

    def transcribe(self, audio_path: Path, language: Optional[str] = None, **kwargs) -> str:
        self._ensure_loaded()
        audio = _load_audio(audio_path, target_sr=16000)
        prompt = "Transcris cet audio." if language is None else f"Transcris cet audio en {language}."
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audios=[audio], return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=256)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "whisper": WhisperWrapper,
    "wav2vec2": Wav2Vec2Wrapper,
    "qwen2_audio": Qwen2AudioWrapper,
}


def load_model(name: str, device: str = "cpu") -> SpeechLLMWrapper:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle '{name}' non supporté. Choix : {list(MODEL_REGISTRY)}")
    wrapper = MODEL_REGISTRY[name](model_name=name, device=device)
    wrapper.load()
    return wrapper
