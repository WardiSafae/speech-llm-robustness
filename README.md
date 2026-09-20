
# 🎙️ Robustesse des Speech-LLMs

**Évaluation systématique, analyse d'erreurs et amélioration par fine-tuning ciblé**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-green.svg)](#)
[![GitHub](https://img.shields.io/badge/GitHub-WardiSafae-blue.svg)](https://github.com/WardiSafae/speech-llm-robustness)

> **Première étude systématique** de la robustesse des Speech-LLMs face aux
> perturbations acoustiques et aux variations linguistiques, avec analyse
> d'erreurs fine et détecteur d'hallucinations.

---

## 🎯 Vision du projet

### Le problème

Les Speech-LLMs (Whisper, Wav2Vec2, Qwen2-Audio) atteignent des performances
remarquables sur des benchmarks standards (LibriSpeech, FLEURS), mais leur
**robustesse en conditions réelles** — bruit urbain, réverbération, accents
sous-représentés — reste **insuffisamment caractérisée**.

**Question centrale** : Les modèles plus grands sont-ils **systématiquement**
meilleurs ? Ou existe-t-il des **compromis cachés** ?

### La réponse en chiffres

```
┌──────────────────────────────────────────────────────────────┐
│  WER (normalisé) sur FLEURS × 12 perturbations              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  whisper-base (74 M)     ████████████████       0.304       │
│  whisper-small (244 M)   ███████████████        ~0.300      │
│  whisper-medium (769 M)  ██████                  0.107 ✅   │
│  whisper-large-v3 (1.5B) █████                   0.080 ✅   │
│                                                              │
│  MAIS...                                                     │
│                                                              │
│  whisper-medium          █                       5 hallu.   │
│  whisper-large-v3        ████████████████████  237 hallu.⚠️ │
└──────────────────────────────────────────────────────────────┘
```

**Découverte majeure** : large-v3 est **30 % meilleur en WER** mais
**hallucine 47× plus** que medium. Cette découverte remet en question la
notion naïve de *"scaling win"*.

---

## 🏆 Contributions scientifiques

### 1. Benchmark exhaustif

**4 modèles Whisper** évalués sur **3600 fichiers** (100/langue × 12 perturbations) :

| Modèle | Params | WER baseline | Hallucinations |
|--------|--------|--------------|----------------|
| whisper-base | 74 M | 0.304 (norm) | — |
| whisper-small | 244 M | ~0.300 | — |
| whisper-medium | 769 M | **0.107** | **5** (0.14 %) |
| **whisper-large-v3** | **1.55 B** | **0.080** ✅ | **237** ⚠️ |

### 2. Découverte du paradoxe large-v3

> **"Meilleur WER, pire hallucinations"**

| Métrique | medium | large-v3 | Verdict |
|----------|--------|----------|---------|
| WER baseline | 0.114 | **0.080** | ✅ -30 % |
| WER perturbé max | 8.68 | **1.50** | ✅ -83 % |
| Hallucinations | **5** | **237** | ⚠️ **+47×** |

**Interprétation** : large-v3 "échoue créativement" (inventions grammaticales),
medium "échoue proprement" (troncatures).

### 3. Hiérarchie des perturbations

Classement par impact décroissant :

```
1. reverb room0.5     : ΔWER = +0.32  🔴🔴🔴
2. white_noise snr5   : ΔWER = +0.22  🔴🔴
3. speed 1.1          : ΔWER = +0.18  🔴
4. speed 0.9          : ΔWER = +0.13  🟡
5. reverb room0.3     : ΔWER = +0.12  🟡
6. white_noise snr10  : ΔWER = +0.10  🟡
7. white_noise snr15  : ΔWER = +0.07  🟡
8. reverb room0.1     : ΔWER = +0.02  🟢
9. clipping (tous)    : ΔWER ≈ 0      🟢
```

**Découverte contre-intuitive** : le **clipping est inoffensif** pour Whisper
(hypothèse : features Mel invariantes à l'amplitude).

### 4. Déblocage de l'arabe

| Modèle | Arabe WER | Amélioration |
|--------|-----------|--------------|
| whisper-base | ~0.70 | — |
| whisper-medium | 0.226 | -68 % |
| **whisper-large-v3** | **0.151** | **-78 %** ✅✅ |

**Large-v3 transforme l'arabe** : de quasi-inutilisable à exploitable.

### 5. Détecteur d'hallucinations

**Production-ready** : détecte les hallucinations avec 4/4 tests passés.

```python
from src.evaluation.hallucination_detector import detect_hallucination

result = detect_hallucination(reference, hypothesis)
if result.is_hallucination:
    print(f"⚠️ {result.reason} (confiance {result.confidence})")
```

**Signaux** :
- Longueur anormale (ratio > 1.5)
- Répétition (token, bigramme, trigramme)
- Troncature (ratio < 0.3)

### 6. Seuil critique de longueur

**5 hallucinations** localisées sur des phrases de **10-27 mots**.

**Seuil recommandé en production** : **25 mots**.

---

## 🗺️ Architecture du projet

```
speech-llm-robustness/
├── README.md                          ← Vous êtes ici
├── LICENSE
├── requirements.txt
├── conftest.py
├── configs/
│   └── finetune_whisper.yaml
│
├── docs/                              📚 Documentation scientifique
│   ├── etat_de_l_art.md               # 36 références
│   ├── limitations.md                 # 7 limites
│   ├── whisper_comparison.md          # base/small/medium/large-v3
│   ├── tts_comparison.md              # SAPI5 vs Edge TTS
│   ├── fleurs_results.md              # Résultats FLEURS
│   ├── perturbation_robustness.md     # Robustesse
│   ├── sample_analysis.md             # Analyse par échantillon
│   ├── length_and_hallucination.md    # Seuil + détecteur
│   ├── hallucination_paradox.md       # 🚨 Paradoxe large-v3
│   └── rapport_final.md               # Rapport complet
│
├── scripts/                           🔧 Utilitaires
│   ├── smoke_test.py                  # Validation
│   ├── compare_models.py              # Comparaison
│   ├── analyze_perturbation.py        # Analyse robustesse
│   ├── analyze_samples.py             # Analyse par échantillon
│   ├── analyze_length_threshold.py    # Analyse par longueur
│   ├── clean_corrupted.py             # Nettoyage
│   └── compare_models_final.py        # Comparaison finale
│
├── src/                               🧠 Code principal
│   ├── data/load_datasets.py          # 3 modes (local/synthetic/neural)
│   ├── augmentation/augment.py        # 5 perturbations × 3 sévérités
│   ├── models/model_wrappers.py       # 5 modèles unifiés
│   ├── evaluation/
│   │   ├── metrics.py                 # WER, CER, latence
│   │   ├── error_analysis.py          # Analyse d'erreurs
│   │   └── hallucination_detector.py  # 🚨 Détecteur
│   ├── finetuning/finetune.py         # LoRA + HF Trainer
│   └── utils/io.py
│
├── tests/
│   └── test_metrics.py                # Tests unitaires
│
├── data/                              📁 Données (gitignore)
│   ├── raw/
│   └── processed/
│
└── results/                           📊 Résultats
    ├── figures/
    │   ├── perturbation/              # Figures robustesse
    │   ├── perturbation_large_v3/     # Figures large-v3
    │   └── ...
    ├── tables/
    │   └── fleurs/
    │       ├── whisper_results.csv
    │       ├── whisper_medium_results.csv
    │       ├── perturbed_medium_results.csv
    │       └── perturbed_large_v3_results.csv
    └── analysis/
        ├── samples/                   # Analyse medium
        ├── samples_large_v3/          # Analyse large-v3
        └── length/                    # Analyse longueur
```

---

## 🚀 Installation

### Prérequis

- **Python 3.11** (3.13 incompatible avec librosa/numba)
- **8 Go RAM** minimum (16 Go recommandé pour large-v3)
- **Connexion internet** pour télécharger les modèles

### Setup

```bash
# Cloner le repo
git clone https://github.com/WardiSafae/speech-llm-robustness.git
cd speech-llm-robustness

# Environnement virtuel
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/Mac

# Dépendances
pip install -r requirements.txt
pip install edge-tts peft        # optionnel

# Vérification
python scripts/smoke_test.py
```

**Attendu** : `12/12 tests passés`

---

## 💻 Utilisation

### 1. Générer des données de test

```bash
# Mode synthetic (TTS local Windows)
python -m src.data.load_datasets --mode synthetic --output data/raw --n 3 --languages en,fr,ar

# Mode neural (Edge TTS - voix neuronales)
python -m src.data.load_datasets --mode neural --output data/raw --n 2 --languages en,fr,ar
```

### 2. Générer les perturbations

```bash
python -m src.augmentation.augment \
    --input data/raw/fleurs_subset \
    --output data/processed/fleurs_perturbed \
    --types white_noise,reverb,clipping,speed \
    --severity-levels
```

### 3. Évaluer un modèle

```bash
python -m src.evaluation.metrics \
    --model whisper_medium \
    --data data/processed/fleurs_perturbed \
    --manifest data/processed/fleurs_perturbed/manifest_perturbed.tsv \
    --output results/tables/fleurs
```

### 4. Analyser les résultats

```bash
# Analyse par perturbation
python scripts/analyze_perturbation.py \
    --results results/tables/fleurs/perturbed_medium_results.csv \
    --output results/figures/perturbation

# Analyse par échantillon
python scripts/analyze_samples.py \
    --results results/tables/fleurs/perturbed_medium_results.csv \
    --output results/analysis/samples

# Analyse par longueur
python scripts/analyze_length_threshold.py \
    --results results/tables/fleurs/perturbed_medium_results.csv \
    --output results/analysis/length
```

### 5. Détecter les hallucinations

```python
from src.evaluation.hallucination_detector import detect_hallucination

result = detect_hallucination(
    reference="le chocolat chaud est conforme aux normes belges",
    hypothesis="Je suis au courant et je vous ai dit que je suis un belge.",
)

if result.is_hallucination:
    print(f"⚠️ {result.reason}")
    print(f"   Confiance : {result.confidence}")
```

**Sortie** :
```
⚠️ hypothèse trop longue (ratio 1.7×); token répété (0.32)
   Confiance : 0.75
```

### 6. Fine-tuning ciblé

```bash
python -m src.finetuning.finetune --config configs/finetune_whisper.yaml
```

---

## 📊 Résultats clés

### Tableau de scaling

| Modèle | Params | WER (baseline) | WER (perturbé) | Hallucinations | Latence |
|--------|--------|----------------|----------------|----------------|---------|
| whisper-base | 74 M | 0.304 | 8.68 | ? | 4 s |
| whisper-small | 244 M | ~0.300 | — | ? | 14 s |
| whisper-medium | 769 M | **0.107** | 8.68 | **5** (0.14 %) | 45 s (CPU) / 1.5 s (GPU) |
| **whisper-large-v3** | **1.55 B** | **0.080** ✅ | 1.50 ✅ | **237** ⚠️ | 1.5 s (GPU) |

### WER par langue (whisper-large-v3)

| Langue | Baseline | reverb 0.5 | white_noise snr5 | speed 1.1 |
|--------|----------|------------|------------------|-----------|
| 🇬🇧 EN | **0.043** | 0.089 | 0.119 | 0.094 |
| 🇫🇷 FR | 0.061 | 0.374 | 0.288 | 0.202 |
| 🇸🇦 AR | 0.151 | 0.396 | 0.275 | 0.316 |

### Hiérarchie des perturbations

```
Reverb > White noise > Speed >> Clipping
```

**Note** : le **clipping est inoffensif** (ΔWER ≈ 0).

---

## 📚 Documentation

| Document | Contenu |
|----------|---------|
| [`docs/etat_de_l_art.md`](docs/etat_de_l_art.md) | 36 références scientifiques |
| [`docs/limitations.md`](docs/limitations.md) | 7 limites documentées |
| [`docs/whisper_comparison.md`](docs/whisper_comparison.md) | base vs small vs medium |
| [`docs/tts_comparison.md`](docs/tts_comparison.md) | SAPI5 vs Edge TTS |
| [`docs/fleurs_results.md`](docs/fleurs_results.md) | Résultats FLEURS |
| [`docs/perturbation_robustness.md`](docs/perturbation_robustness.md) | Robustesse par perturbation |
| [`docs/sample_analysis.md`](docs/sample_analysis.md) | Cas difficiles |
| [`docs/length_and_hallucination.md`](docs/length_and_hallucination.md) | Seuil + détecteur |
| [`docs/hallucination_paradox.md`](docs/hallucination_paradox.md) | 🚨 Paradoxe large-v3 |
| [`docs/rapport_final.md`](docs/rapport_final.md) | Rapport complet |

---

## 🎯 Feuille de route

### ✅ Phase 1-6 : Complétées (5 mois)

- [x] Pipeline complet (data → augment → models → eval)
- [x] Benchmark 4 modèles Whisper
- [x] 3600 fichiers perturbés (5 perturbations × 3 sévérités)
- [x] Analyse par langue (EN, FR, AR)
- [x] Analyse par échantillon (top-20, hallucinations)
- [x] Détecteur d'hallucinations (4/4 tests)

### 🚧 Phase 7-8 : En cours

- [x] whisper-large-v3 évalué
- [ ] Wav2Vec2-XLSR-53 (comparaison CTC)
- [ ] Qwen2-Audio (Speech-LLM instructible)
- [ ] Analyse phonétique (panphon)
- [ ] Fine-tuning ciblé FR

### 📅 Phase 9-12 : À venir

- [ ] Rédaction du rapport scientifique (arXiv)
- [ ] Soumission workshop (ArabicNLP 2026)
- [ ] Publication code (pip package)
- [ ] Blog post + LinkedIn

---

## 🛠️ Stack technique

| Catégorie | Outils |
|-----------|--------|
| **Langage** | Python 3.11 |
| **Speech-LLMs** | Whisper (base/small/medium/large-v3), Wav2Vec2, Qwen2-Audio |
| **Frameworks ML** | PyTorch 2.14, Hugging Face Transformers 5.x, PEFT |
| **Audio** | Librosa, soundfile, edge-tts |
| **Évaluation** | jiwer, pandas, numpy |
| **Fine-tuning** | Seq2SeqTrainer, LoRA |
| **Visualisation** | Matplotlib, Seaborn |
| **Reproductibilité** | Git, YAML configs |

---

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/ -v

# Test du détecteur d'hallucinations
python -m src.evaluation.hallucination_detector

# Smoke test complet
python scripts/smoke_test.py
```

**Résultats attendus** :
- `pytest` : 3/3 tests passés
- Détecteur : 4/4 tests passés
- Smoke test : 12/12 tests passés

---

## ⚠️ Limitations

Voir [`docs/limitations.md`](docs/limitations.md) pour la liste complète.

**Points principaux** :

1. **Dataset synthétique** : TTS au lieu de parole humaine (partiellement résolu avec FLEURS)
2. **Bug LoRA Whisper** : contourné par fine-tuning complet sur tiny
3. **CPU-only local** : limite les modèles testables (large-v3 sur Colab)
4. **Disque** : limite à 2.5 Go localement, donc 30-100 échantillons/langue
5. **Hallucinations** : détecteur en développement continu

---




## 🤝 Contribution

Les contributions sont bienvenues !

**Idées d'amélioration** :
1. Tester d'autres modèles (Wav2Vec2, Qwen2-Audio)
2. Ajouter des perturbations (compression MP3, écho)
3. Améliorer le détecteur d'hallucinations
4. Fine-tuning ciblé par langue

**Workflow** :
```bash
git checkout -b feature/ma-contribution
git commit -m "feat: add XYZ"
git push origin feature/ma-contribution
```

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE)

---

## 🙏 Remerciements

- **OpenAI** pour Whisper (open-source)
- **Google** pour FLEURS (dataset)
- **Hugging Face** pour Transformers
- **Microsoft** pour Edge TTS

---

## 📞 Contact

- **GitHub** : [@WardiSafae](https://github.com/WardiSafae)
- **LinkedIn** : [Safae Wardi](https://linkedin.com/in/safae-wardi-1605b0434)
- **Email** : [wardisafae37@gmail.com](mailto:wardisafae37@gmail.com)

---

<p align="center">
  <b>⭐ Si ce projet vous intéresse, n'hésitez pas à mettre une étoile !</b>
</p>

<p align="center">
  <i>Dernière mise à jour : Septembre 2026</i>
</p>


---
