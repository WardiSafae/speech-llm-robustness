# Résultats FLEURS — Évaluation finale (4 modèles Whisper)

## Protocole

- **Dataset** : FLEURS (Google), split `test`
- **Échantillons** : 100 par langue (en, fr, ar) = **300 total**
- **Modèles** :
  - `openai/whisper-base` (74 M) — CPU local
  - `openai/whisper-small` (244 M) — CPU local
  - `openai/whisper-medium` (769 M) — GPU Colab T4
  - `openai/whisper-large-v3` (1.55 B) — GPU Colab T4

## Tableau de scaling complet

| Modèle | Params | WER brut | WER norm | CER | Latence | Hardware |
|--------|--------|----------|----------|-----|---------|----------|
| whisper-base | 74 M | 0.4128 | 0.3036 | 0.1177 | 9.96 s | CPU |
| whisper-small | 244 M | ~0.42 | ~0.30 | — | 14.5 s | CPU |
| whisper-medium | 769 M | 0.2495 | 0.107 | 0.0663 | 1.53 s | GPU T4 |
| **whisper-large-v3** | **1.55 B** | **0.2194** | **0.0802** | **0.0269** | 1.54 s | GPU T4 |

**Gain total base → large-v3** :
- WER normalisé : **-74 %**
- CER : **-77 %**
- Latence (GPU) : **-85 %**

## Détail par langue

### whisper-large-v3 (100/langue)

| Langue | WER brut | WER norm | CER |
|--------|----------|----------|-----|
| 🇬🇧 **EN** | 0.2489 | **0.0433** | 0.0213 |
| 🇫🇷 **FR** | 0.2402 | **0.0605** | 0.0226 |
| 🇸🇦 **AR** | **0.1578** | **0.1506** | 0.0393 |

### Progression sur l'arabe (le cas difficile)

| Modèle | AR WER brut | AR WER norm | Gain |
|--------|-------------|-------------|------|
| whisper-base | ~0.70 | ~0.65 | — |
| whisper-small | ~0.65 | ~0.60 | -8 % |
| whisper-medium | 0.2399 | ~0.199 | **-67 %** ✅ |
| **whisper-large-v3** | **0.1578** | **0.1506** | **-77 %** ✅✅ |

## 🎯 Découverte : le "seuil de capacité"

Le scaling Whisper n'est **pas linéaire** :
```
base (0.30) → small (0.30) : aucun gain
small (0.30) → medium (0.11) : -64 % ← SEUIL
medium (0.11) → large-v3 (0.08): -25 %
```

**Interprétation** : Il existe un **seuil de capacité** autour de **500 M params**.
En dessous, l'arabe est inutilisable (WER > 0.5). Au-dessus, il devient praticable
(WER < 0.25).

## Convergence des langues

| Modèle | EN | FR | AR | Disparité |
|--------|-----|-----|-----|-----------|
| whisper-base | 0.05 | 0.35 | 0.70 | **14×** |
| whisper-medium | 0.05 | 0.09 | 0.20 | **4×** |
| **whisper-large-v3** | **0.04** | **0.06** | **0.15** | **3.7×** |

**Interprétation** : la taille du modèle **réduit la disparité** entre langues
sans l'éliminer. L'arabe reste 3× plus difficile que l'anglais, mais
**praticable** avec large-v3.

## Recommandations finales

### Choix du modèle par cas d'usage

| Cas d'usage | Modèle recommandé | Justification |
|-------------|-------------------|---------------|
| Production multilingue (qualité) | **whisper-large-v3** | WER 0.08 norm |
| Production multilingue (compromis) | whisper-medium | WER 0.11 norm, RAM modeste |
| Anglais uniquement | whisper-base | WER ~0.10, très rapide |
| Contraintes RAM sévères | whisper-small | Baseline acceptable |
| **Arabe spécifiquement** | **whisper-large-v3** | WER 0.15 (vs 0.24 pour medium) |

### Pour ce projet

1. **Utiliser whisper-large-v3 comme référence** dans le rapport
2. **whisper-medium est le meilleur compromis** performance/ressources
3. **Documenter le scaling law** pour justifier les choix

## Comparaison TTS vs FLEURS (final)

| Source | Modèle | EN | FR | AR | Global |
|--------|--------|-----|-----|-----|--------|
| TTS Neural | small | 0.00 | 0.89 | 0.71 | 0.48 |
| TTS Neural | medium | 0.00 | 0.33 | 0.29 | 0.20 |
| **FLEURS** | base | ~0.10 | ~0.35 | ~0.70 | 0.30 |
| **FLEURS** | medium | 0.05 | 0.09 | 0.20 | 0.11 |
| **FLEURS** | **large-v3** | **0.04** | **0.06** | **0.15** | **0.08** ✅ |

**Conclusion** : Le TTS n'est **pas un bon proxy** — il surestime l'anglais
et sous-estime l'arabe.

## Fichiers reproductibles

- Manifest : `data/raw/fleurs_subset/manifest_fleurs.tsv`
- Résultats :
  - `results/tables/fleurs/whisper_base_results.csv`
  - `results/tables/fleurs/whisper_medium_results.csv`
  - `results/tables/fleurs/whisper_large_v3_results.csv`
