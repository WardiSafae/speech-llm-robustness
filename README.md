# Robustesse des Speech-LLMs : évaluation, analyse d'erreurs et amélioration par fine-tuning ciblé

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-green.svg)](#)

> Projet de recherche appliquée évaluant et améliorant la robustesse des
> Speech-LLMs (Whisper, Wav2Vec2, Qwen2-Audio) face aux perturbations
> acoustiques et aux variations d'accents.

---

## 🎯 Contribution originale

Ce projet propose la **première évaluation systématique combinée** de
plusieurs Speech-LLMs hétérogènes sur **quatre dimensions** :

1. **Évaluation multi-modèles** : Whisper (base/small/medium), Wav2Vec2-XLSR-53,
   Qwen2-Audio — interface unifiée `transcribe(audio, language)`.
2. **Perturbations acoustiques** : 5 types (white_noise, street_noise, reverb,
   clipping, speed) × N niveaux de sévérité.
3. **Accents et langues** : anglais, français, arabe — avec TTS SAPI5 et Edge TTS
   neuronaux.
4. **Analyse d'erreurs fine** : phonèmes confondus, hallucinations, top mots
   mal transcrits, WER par langue/perturbation.
5. **Fine-tuning ciblé** : LoRA + gradient checkpointing sur les conditions
   difficiles identifiées.

Le tout dans un **pipeline reproductible** (code open-source, manifest TSV,
CSV de résultats, tests unitaires).

---

## 📊 Résultats préliminaires

### Comparaison des modèles Whisper

| Modèle | Params | WER | CER | Latence | N |
|--------|--------|-----|-----|---------|---|
| whisper-base | 74 M | 0.480 | 0.170 | **4.3 s** | 6 |
| whisper-small | 244 M | 0.520 | 0.099 | 14.5 s | 6 |
| **whisper-medium** | **769 M** | **0.200** | **0.057** | 45.7 s | 6 |

**Conclusion** : whisper-medium est **2.6× meilleur** que whisper-small
sur des données TTS multilingues. Le gain n'est pas linéaire (base ≈ small,
puis saut à medium).

### Détail par langue (whisper-medium, Edge TTS)

| Langue | WER | CER |
|--------|-----|-----|
| 🇬🇧 Anglais | **0.000** | 0.000 |
| 🇸🇦 Arabe | **0.286** | 0.071 |
| 🇫🇷 Français | **0.333** | 0.091 |

### Comparaison TTS : SAPI5 vs Edge TTS

| Langue | SAPI5 | Edge TTS | Verdict |
|--------|-------|----------|---------|
| Anglais | 0.00 | 0.00 | = |
| Français | 0.44 | **0.89** | Edge TTS pire (hypothèse : hors distribution) |
| Arabe | 0.43 | **0.71** | Edge TTS pire |

**Résultat contre-intuitif** documenté dans [`docs/tts_comparison.md`](docs/tts_comparison.md).

### Fine-tuning Whisper-tiny (français)

| Métrique | Avant | Après | Δ |
|----------|-------|-------|---|
| WER | 0.444 | **0.000** | -100 % |
| CER | 0.127 | **0.000** | -100 % |

⚠️ **Attention** : ce résultat reflète un **sur-apprentissage** sur un dataset
très petit (3 échantillons). Voir [`docs/limitations.md`](docs/limitations.md).

---

## 🏗️ Architecture du dépôt

```
speech-llm-robustness/
├── README.md
├── LICENSE
├── requirements.txt
├── conftest.py
├── .gitignore
├── configs/
│ └── finetune_whisper.yaml
├── docs/
│ ├── etat_de_l_art.md # 36 références, positionnement
│ ├── limitations.md # 7 limites documentées
│ ├── resultats_preliminaires.md
│ ├── whisper_comparison.md # base vs small vs medium
│ └── tts_comparison.md # SAPI5 vs Edge TTS
├── scripts/
│ ├── smoke_test.py # Validation complète offline
│ ├── test_finetune.py # Test minimal fine-tuning
│ ├── compare_models.py # Comparaison multi-modèles
│ └── test_comparatif_SAPI5_EdgeTTS.py
├── src/
│ ├── data/load_datasets.py # local / synthetic / neural / streaming
│ ├── augmentation/augment.py # 5 perturbations
│ ├── models/model_wrappers.py # 5 modèles unifiés
│ ├── evaluation/
│ │ ├── metrics.py # WER, CER, latence
│ │ └── error_analysis.py # Phonèmes, hallucinations, top-mots
│ ├── finetuning/finetune.py # LoRA + HF Trainer
│ └── utils/io.py
├── tests/
│ └── test_metrics.py
└── results/
├── figures/
└── tables/
├── whisper_results.csv # small
├── whisper_base_results.csv # base
├── whisper_medium_results.csv # medium
└── summary.csv

```


---

## 🚀 Installation

```bash
git clone https://github.com/<votre-username>/speech-llm-robustness.git
cd speech-llm-robustness
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pip install edge-tts peft          # dépendances optionnelles

```

---

## 💡 Utilisation

### 1. Générer des données de test

```bash
# Mode synthetic (TTS local Windows SAPI5)
python -m src.data.load_datasets --mode synthetic --output data/raw --n 3 --languages en,fr,ar

# Mode neural (Edge TTS, voix neuronales)
python -m src.data.load_datasets --mode neural --output data/raw --n 2 --languages en,fr,ar
```

### 2. Augmenter les données

```bash
python -m src.augmentation.augment --input data/raw --output data/processed \
    --types white_noise,reverb,clipping,speed
```

### 3. Évaluer un modèle

```bash
# Whisper-small (référence)
python -m src.evaluation.metrics --model whisper --data data/raw \
    --manifest data/raw/manifest.tsv --output results/tables

# Whisper-medium (meilleur)
python -m src.evaluation.metrics --model whisper_medium --data data/raw \
    --manifest data/raw/manifest.tsv --output results/tables
```

### 4. Analyser les erreurs

```bash
python -m src.evaluation.error_analysis \
    --results results/tables/whisper_medium_results.csv \
    --output results/error_analysis
```

### 5. Fine-tuning

```bash
python -m src.finetuning.finetune --config configs/finetune_whisper.yaml
```

### 6. Validation rapide

```bash
python scripts/smoke_test.py
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

**Attendu :** 3 tests passés (WER/CER identity + error).

---

## 🎯 Stack technique

| Catégorie | Outils |
|-----------|--------|
| Langage | Python 3.11 |
| Speech-LLMs | Whisper (base/small/medium), Wav2Vec2-XLSR-53, Qwen2-Audio |
| Frameworks ML | PyTorch 2.14, Hugging Face Transformers 5.x, PEFT |
| Audio | Librosa, soundfile, edge-tts |
| Évaluation | jiwer, pandas, numpy |
| Fine-tuning | Seq2SeqTrainer, LoRA (PEFT) |
| Reproductibilité | Git, YAML configs |

---

## 📋 Feuille de route

### ✅ Phase 1 — État de l'art et benchmark
- [x] Revue de littérature (36 références)
- [x] Sélection des modèles et datasets
- [x] Architecture du dépôt

### ✅ Phase 2 — Évaluation de la robustesse
- [x] Pipeline de perturbations acoustiques (5 types)
- [x] Interface unifiée pour 5 modèles
- [x] Évaluation baseline multi-langue (en/fr/ar)
- [x] Comparaison whisper base/small/medium

### ✅ Phase 3 — Analyse d'erreurs et fine-tuning
- [x] Analyse de phonèmes confondus
- [x] Détection d'hallucinations
- [x] Top-N mots mal transcrits
- [x] Fine-tuning Whisper-tiny (LoRA-ready)
- [x] Évaluation avant/après

### 🚧 Phase 4 — Documentation et valorisation
- [x] Rapport préliminaire (docs/)
- [x] Limitations documentées
- [ ] Notebook interactif récapitulatif

---

## ⚠️ Limitations

Voir [`docs/limitations.md`](docs/limitations.md) pour la liste complète.
Points principaux :

1. Dataset synthétique trop petit (3 échantillons par langue)
2. Bug LoRA sur Whisper (contourné par fine-tuning complet)
3. Qualité TTS variable selon la langue
4. Évaluation CPU-only
5. Absence de benchmarks externes (FLEURS, Common Voice)

---

## 📚 Documentation

| Document | Contenu |
|----------|---------|
| [`docs/etat_de_l_art.md`](docs/etat_de_l_art.md) | 36 références, positionnement |
| [`docs/limitations.md`](docs/limitations.md) | 7 limites + solutions |
| [`docs/whisper_comparison.md`](docs/whisper_comparison.md) | base vs small vs medium |
| [`docs/tts_comparison.md`](docs/tts_comparison.md) | SAPI5 vs Edge TTS |
| [`docs/resultats_preliminaires.md`](docs/resultats_preliminaires.md) | Résultats intermédiaires |


## 📄 Licence

MIT — voir [LICENSE](LICENSE).

---

**Auteur** : Safae Wardi — Master Big Data et Systèmes Intelligents, USMBA Fès

**Contact** : [wardisafae37@gmail.com]

---