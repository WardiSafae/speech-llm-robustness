# Comparaison Whisper : base, small, medium

## Protocole

- **Modèles** :
  - `openai/whisper-base` (74 M params)
  - `openai/whisper-small` (244 M params)
  - `openai/whisper-medium` (769 M params)
- **Données** : 15 échantillons TTS (SAPI5 + Edge TTS), 3 langues (en, fr, ar)
- **Hardware** : CPU-only (pas de GPU)
- **Métriques** : WER, CER, latence moyenne

## Résultats globaux

| Modèle | Params | WER | CER | Latence | N |
|--------|--------|-----|-----|---------|---|
| whisper-base | 74 M | **0.480** | 0.102 | **4.3 s** | 6 |
| whisper-small | 244 M | 0.376 | 0.091 | 13.9 s | 15 |
| **whisper-medium** | **769 M** | **0.176** | **0.044** | 45.2 s | 15 |

## Résultats par langue (données neural, Edge TTS)

| Langue | base | small | medium |
|--------|------|-------|--------|
| EN | 0.00 | 0.00 | **0.00** |
| FR | 0.89 | 0.89 | **0.20** |
| AR | 0.71 | 0.71 | **0.15** |

## Analyse

### 1. La taille du modèle est critique sur les langues difficiles
FR : 0.89 → 0.89 → 0.20 (base → small → medium : +77 % de gain)
AR : 0.71 → 0.71 → 0.15 (base → small → medium : +79 % de gain)
EN : 0.00 → 0.00 → 0.00 (aucun impact)


**Interprétation** : Sur les langues bien dotées (anglais), la taille du modèle
n'apporte rien. Sur les langues sous-dotées (arabe) ou avec des liaisons
complexes (français), whisper-medium apporte un **gain majeur**.

### 2. Le TTS neural n'est pas le facteur limitant

Avec whisper-medium + Edge TTS, le WER français tombe à **0.20** (contre 0.89
avec whisper-small). Cela **infirme notre hypothèse initiale** selon laquelle
la qualité TTS était le problème — c'est bien **whisper-small** qui était la
limite.

### 3. Erreurs linguistiques restantes (whisper-medium, FR)

Même whisper-medium fait des erreurs sur :
- `renard` → `nard` ou `vifrenard` (élision + fusion)
- `Le vif` → `Le vifre` (ajout de `re`)

Ces erreurs sont typiques des **liaisons françaises**, difficiles même pour
des humains non-natifs.

### 4. Compromis performance/latence

| Cas d'usage | Modèle recommandé |
|-------------|-------------------|
| Temps réel (faible latence) | whisper-base (4.3 s) |
| Production standard | whisper-small (13.9 s) |
| Précision maximale (batch) | **whisper-medium** (45.2 s) |
| Anglais uniquement | whisper-base |

**Sur CPU**, whisper-medium est **10× plus lent** que whisper-base pour un
gain de 60 % de WER. Le compromis dépend du cas d'usage.

## Recommandations pour le projet

1. **Utiliser whisper-medium comme modèle de référence** pour les évaluations
2. **Tester whisper-large-v3 sur Colab** (GPU) pour mesurer le plafond
3. **Documenter le scaling** : base → small → medium → large
4. **Ne pas se focaliser sur le TTS** — la qualité TTS est suffisante

## Fichiers reproductibles

- Résultats :
  - `results/tables/whisper_results.csv` (small)
  - `results/tables/whisper_base_results.csv` (base)
  - `results/tables/whisper_medium_results.csv` (medium)
- Manifest : `data/raw/manifest.tsv`
- Script de comparaison : `scripts/compare_models.py`

## Prochaines étapes

1. Tester **whisper-large-v3** (1.5 B params) sur GPU Colab
2. Comparer sur **dataset plus large** (n ≥ 30 par condition)
3. Ajouter **Wav2Vec2-XLSR-53** et **Qwen2-Audio** (si ressources)
4. Évaluer sur **enregistrements humains** (FLEURS, Common Voice)




# Comparaison finale — 4 modèles Whisper

## Résumé

| Modèle | Params | WER baseline (ALL) | WER perturbé max | Hallucinations FR |
|--------|--------|-------------------|------------------|-------------------|
| whisper-base | 74 M | 0.304 (norm) | 8.68 | ? |
| whisper-small | 244 M | ~0.30 | — | ? |
| whisper-medium | 769 M | **0.107** | 8.68 | **5** (0.14 %) |
| **whisper-large-v3** | **1.55 B** | **0.080** ✅ | ? | **44** (3.67 %) ⚠️ |

## 🚨 Découverte majeure : le paradoxe WER vs hallucinations

Large-v3 est **30 % meilleur** en WER mais **8× pire** en hallucinations sur FR.

**Interprétation** :
- **WER** capture la performance moyenne
- **Hallucinations** capturent les cas extrêmes
- Large-v3 a une **queue lourde** de distribution

## Recommandation production

- **whisper-medium** : préférable pour la production (moins d'hallucinations)
- **whisper-large-v3 + détecteur** : alternative si latence OK
- **Fine-tuning FR** : priorité absolue pour large-v3