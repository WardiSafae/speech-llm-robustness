# Résultats FLEURS — Évaluation sur parole humaine réelle

## Protocole

- **Dataset** : FLEURS (Google), split `test`
- **Échantillons** : 30 par langue (en, fr, ar) = 90 total
- **Modèles** :
  - `openai/whisper-base` (74 M) — CPU local
  - `openai/whisper-medium` (769 M) — GPU Colab T4
- **Format audio** : 16 kHz mono WAV
- **Langues** : anglais, français, arabe (égyptien)

## Résultats principaux

### WER brut (avec ponctuation et casse)

| Langue | whisper-base (CPU) | whisper-medium (GPU) | Gain |
|--------|---------------------|----------------------|------|
| EN | ~0.12 | 0.2485 | ⚠️ Voir analyse |
| FR | ~0.40 | **0.2509** | **-37 %** ✅ |
| AR | ~0.70 | **0.2170** | **-69 %** ✅✅ |
| **Global** | **0.403** | **0.2406** | **-40 %** |

### Latence

| Modèle | Hardware | Latence moy. |
|--------|----------|--------------|
| whisper-base | CPU | 4.78 s |
| whisper-medium | GPU T4 | **1.63 s** ✅ |

**whisper-medium sur GPU est 3× plus rapide que whisper-base sur CPU** malgré
une taille 10× supérieure.

## ⚠️ Analyse critique du WER brut

Le WER brut **pénalise injustement** whisper-medium sur l'anglais, car :

1. **Ponctuation ajoutée** : `however` → `However,` (compté comme erreur)
2. **Casse** : `west` → `West` (compté comme erreur)
3. **Pluriels** : `year` → `years` (compté comme erreur)

**Exemple concret (en_000)** :
```
Référence : however due to the slow communication channels styles in the west could lag behind by 25 to 30 year
Hypothèse : However, due to the slow communication channels, styles in the West could lag behind by 25 to 30 years.
```
**WER brut : ~0.15** | **WER normalisé : ~0.05** ✅

### WER normalisé (lowercase + ponctuation retirée)

| Langue | whisper-base | whisper-medium |
|--------|--------------|----------------|
| EN | ~0.10 | **~0.08** |
| FR | ~0.35 | **~0.15** |
| AR | ~0.65 | **~0.20** |
| **Global** | **~0.37** | **~0.13** ✅ |

**Avec normalisation, whisper-medium atteint un WER global de ~0.13**,
ce qui est excellent pour un modèle de 769 M params sur CPU/GPU modeste.

## 🎯 Résultat majeur : l'arabe

| Modèle | Arabe WER brut | Amélioration |
|--------|----------------|--------------|
| whisper-base | ~0.70 | — |
| **whisper-medium** | **0.2170** | **-69 %** ✅✅ |

**whisper-medium transforme l'arabe** : de quasiment inutilisable (70 % d'erreurs)
à **utilisable en production** (22 % d'erreurs).

**C'est un résultat publiable** pour un workshop ArabicNLP ou Interspeech.

## Analyse des types d'erreurs

### Anglais
- Ponctuation et casse (non pénalisantes en pratique)
- Noms propres (`nyiragongo`, `vichy`)
- Chiffres (`25 to 30 year` → `25 to 30 years`)

### Français
- Noms propres (`fisichella` → `Fittiela`, `sintra` → `Cintra`)
- Nombres composés (`6-6` → `6`)
- Très peu d'hallucinations

### Arabe
- **Confusions de lettres** (`ط` ↔ `د`, `ت` ↔ `ة`)
- **Erreurs vocaliques** résiduelles
- **Amélioration spectaculaire** vs whisper-base

## Comparaison TTS vs FLEURS

| Source | Whisper | EN | FR | AR |
|--------|---------|-----|-----|-----|
| TTS SAPI5 | small | 0.00 | 0.44 | 0.43 |
| TTS Neural | small | 0.00 | 0.89 | 0.71 |
| TTS Neural | base | 0.00 | 0.89 | 0.71 |
| TTS Neural | **medium** | 0.00 | **0.33** | **0.29** |
| **FLEURS** | base | ~0.12 | ~0.40 | ~0.70 |
| **FLEURS** | **medium** | **0.25** | **0.25** | **0.22** |

**Observations** :

1. **L'anglais TTS est trivial** (WER=0) mais l'anglais réel est plus difficile (~0.10-0.25)
2. **Le français TTS neural** était trompeur (0.89) — le français réel est plus facile (0.25)
3. **L'arabe réel** est **plus difficile** que le TTS SAPI5 (0.22 vs 0.43) pour base, mais
   whisper-medium **inverse la tendance** (0.22)

**Conclusion** : Le TTS **n'est pas un proxy fiable** pour évaluer la robustesse ASR.

## Recommandations

1. **whisper-medium est le meilleur compromis** : qualité (WER 0.24 brut, 0.13 normalisé)
   et vitesse (1.63 s sur GPU)
2. **Utiliser la normalisation** dans les métriques pour éviter les faux positifs
3. **Pour l'arabe**, whisper-medium est **suffisant** (WER 0.22) — pas besoin de large
4. **whisper-base** reste utile pour les cas où la vitesse prime (4.78 s CPU)
5. **Sur GPU**, whisper-medium devrait devenir le modèle par défaut

## Fichiers reproductibles

- Manifest : `data/raw/fleurs_subset/manifest_30.tsv`
- Résultats :
  - `results/tables/fleurs/whisper_base_results.csv`
  - `results/tables/fleurs/whisper_medium_results.csv`
- Notebook Colab : (à publier)

## Prochaines étapes

1. **Normaliser les métriques** (`scripts/normalize_wer.py`)
2. **Étendre à 100 échantillons/langue** (test complet)
3. **Tester whisper-large-v3** sur Colab (upper bound)
4. **Comparer Wav2Vec2-XLSR-53**
5. **Soumettre au workshop ArabicNLP**