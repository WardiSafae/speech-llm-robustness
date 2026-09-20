# Analyse par échantillon — Whisper-medium sur FLEURS perturbé

## Protocole

- **3600 échantillons** (300 × 12 perturbations)
- **Modèle** : `openai/whisper-medium` (GPU T4)
- **Métriques** : WER par échantillon, CER, ratio de longueur

## Statistiques globales

| Métrique | Valeur |
|----------|--------|
| WER moyen | 0.230 |
| WER médian | 0.154 |
| WER 95e percentile | 0.688 |
| **WER max** | **8.680** 🚨 |
| CER moyen | 0.100 |

## Distribution par langue

| Langue | WER moyen | Médiane | **Max** | Écart-type |
|--------|-----------|---------|---------|-----------|
| EN | 0.085 | 0.056 | 0.867 | 0.13 |
| FR | **0.259** | 0.177 | **8.680** 🚨 | 0.51 |
| AR | **0.346** | 0.313 | 3.800 | 0.42 |

## 🚨 Découvertes majeures

### 1. Toutes les catastrophes sont des hallucinations

Les 4 pires cas (WER > 3.5) sont **tous** des hallucinations par répétition :

| Racine | Perturbation | WER | Ratio | Pattern |
|--------|--------------|-----|-------|---------|
| fr_064 | reverb 0.5 | 8.68 | 8.84 | `ville de la ville de la ville...` |
| fr_031 | white_noise 15 | 6.64 | 6.95 | `milliers de milliers...` |
| ar_000 | reverb 0.5 | 3.80 | 3.80 | `تشتت تشتت تشتت...` |
| fr_058 | white_noise 5 | 3.52 | 3.67 | `Suji Suji Suji...` |

**Pattern commun** : répétition infinie d'un mot ou d'une phrase.

### 2. Le français domine les cas catastrophiques

Sur les **10 pires échantillons** :
- **FR** : 7 cas (70 %)
- **AR** : 3 cas (30 %)
- **EN** : **0 cas** (0 %) ✅

**L'anglais n'a AUCUN cas catastrophique.**

### 3. Reverb et white_noise déclenchent les hallucinations

| Perturbation | Cas catastrophiques |
|--------------|---------------------|
| **reverb 0.5** | 7 cas |
| **white_noise** | 3 cas |
| speed | 1 cas |
| **clipping** | **0 cas** ✅ |

**Reverb et white noise** sont les déclencheurs.

### 4. Corrélation avec la longueur

Les cas catastrophiques ont tous :
- **ref_len entre 10 et 29 mots**
- **hyp_len entre 14 et 221 mots** (hallucinations massives)

**Plus la phrase est longue, plus Whisper hallucine.**

## 🎯 Recommandations

### Pour la production

1. **Détecter les hallucinations** :
   - Si `hyp_len / ref_len > 2.0` → rejeter
   - Si répétition détectée (n-grammes) → rejeter
2. **Limiter la longueur d'audio** : phrases > 20 mots sont risquées
3. **Déréverbérer** les audios avant transcription

### Pour le fine-tuning

1. **Cibler FR** (70 % des cas catastrophiques)
2. **Augmenter avec reverb 0.3-0.5** (7 cas)
3. **Ajouter white_noise snr5-10** (3 cas)
4. **Fine-tuner sur les phrases longues** (10-29 mots)

### Pour les prochaines expériences

1. **Tester whisper-large-v3** : est-il plus robuste aux hallucinations ?
2. **Utiliser un seuil de longueur** : rejeter si ratio > 1.5
3. **Compare avec Wav2Vec2** (CTC, pas d'hallucinations possibles)

## Fichiers reproductibles

- `results/analysis/samples/top20_difficult.csv`
- `results/analysis/samples/hallucinations.csv`
- `results/analysis/samples/worst_per_root.csv`
- `results/analysis/samples/stats_by_perturbation.csv`

## Prochaines étapes

1. **Analyse par longueur** (seuil critique ?)
2. **Détecteur d'hallucinations** en production
3. **Fine-tuning ciblé FR**
4. **Test whisper-large-v3**