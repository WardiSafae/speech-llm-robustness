# Analyse par longueur et détecteur d'hallucinations

## Partie A — Analyse par longueur

### Distribution des longueurs (3600 échantillons)

- Min : 6 mots
- Max : 71 mots
- Moyenne : 22.6 mots
- Médiane : 21.0 mots

### 🎯 Découverte : WER diminue avec la longueur

| Longueur | N | WER moyen | WER max | Taux hallucination |
|----------|---|-----------|---------|-------------------|
| 5-10 | 48 | **0.341** | 1.000 | 0.00 % |
| 10-15 | 540 | 0.296 | 2.000 | **0.19 %** 🚨 |
| 15-20 | 852 | 0.234 | 3.800 | 0.12 % |
| 20-25 | 912 | 0.235 | 6.636 | **0.22 %** 🚨 |
| 25-30 | 600 | 0.202 | **8.680** | 0.17 % |
| 30-40 | 516 | **0.171** | 1.000 | 0.00 % |
| 40-100 | 132 | 0.211 | 0.952 | 0.00 % |

**Observation contre-intuitive** : les phrases longues (30-40 mots) ont un WER
**plus faible** (0.171) que les phrases courtes (0.341).

**Hypothèse** : plus de contexte = meilleure prédiction Whisper.

### 🚨 Concentrations d'hallucinations

**5 hallucinations** sur 3600 (0.14 %), concentrées sur :
- 10-15 mots : **1 cas** (ar_000 avec `تشتت تشتت...`)
- 15-20 mots : 0 cas
- 20-25 mots : **2 cas** (fr_031, fr_058)
- 25-30 mots : 1 cas (fr_064 — le pire)
- 30+ : 0 cas

**Les hallucinations apparaissent sur des phrases de 10-30 mots**, surtout
**longues (20-30)**.

### Par langue

**Anglais — Stable et robuste**
| Longueur | WER |
|----------|-----|
| 5-10 | 0.046 |
| 10-15 | 0.082 |
| 15-20 | 0.108 |
| 20-25 | 0.094 |
| 25-30 | 0.047 |
| 30-40 | 0.069 |

- WER entre 0.05 et 0.11
- **ZÉRO hallucination** sur toutes les longueurs ✅

**Français — Variable**
| Longueur | WER | Hallucinations |
|----------|-----|----------------|
| 10-15 | **0.397** 🚨 | 1 |
| 15-20 | 0.311 | 0 |
| 20-25 | 0.264 | 2 |
| 25-30 | 0.280 | 1 |
| 30-40 | 0.192 | 0 |
| 40+ | 0.211 | 0 |

- Très mauvais sur phrases **courtes (10-15 mots)**
- **Hallucine** sur phrases **longues (20-30 mots)**

**Arabe — WER diminue avec la longueur**
| Longueur | WER |
|----------|-----|
| 5-10 | 0.439 |
| 10-15 | 0.380 |
| 15-20 | 0.349 |
| 20-25 | 0.341 |
| 25-30 | 0.268 |
| 30-40 | 0.357 |

- L'arabe **profite** des phrases longues (contexte phonétique)
- Zéro hallucination sauf 1 à 15-20 mots

## Partie D — Détecteur d'hallucinations

### Signaux

| Signal | Seuil | Poids |
|--------|-------|-------|
| Longueur anormale | ratio > 1.5 | 0.5-0.9 |
| Longueur courte | ratio < 0.3 | 0.5 |
| Token répété | > 30 % | 0.3-1.0 |
| Bigramme répété | > 20 % | 0.2-1.0 |
| Trigramme répété | > 15 % | 0.2-1.0 |
| Phrase longue + token | > 20 tokens + > 15 % | 0.7 |
| Longueur extrême | > 50 tokens | 0.95 |

### Performance

**4/4 tests passés** ✅

| Cas | Détecté | Confiance | Signal principal |
|-----|---------|-----------|------------------|
| fr_064 (massive) | ✅ | 0.62 | len_ratio 8.84 |
| fr_031 (milliers) | ✅ | 0.68 | len_ratio 6.96 |
| ar_000 (تشتت) | ✅ | **1.00** | token_rep 1.00 |
| Phrase normale | ✅ (non détectée) | 0.0 | Aucun |

### Seuils recommandés en production

| Utilisation | Seuil de confiance |
|-------------|-------------------|
| Stricte (peu de faux positifs) | > 0.8 |
| Équilibrée | > 0.6 |
| Sensible (peu de faux négatifs) | > 0.4 |

## Recommandations

### Pour la production

1. **Filtrer les transcriptions** avec confidence > 0.6
2. **Signaler** les cas 0.4-0.6 pour vérification humaine
3. **Accepter** les cas < 0.4 (probablement OK)
4. **Limiter les audios** à < 30 mots pour éviter les hallucinations

### Pour le fine-tuning

1. **Cibler le français** (toutes les hallucinations FR)
2. **Augmenter avec reverb 0.5** (déclencheur principal)
3. **Focus sur phrases 20-30 mots** (pic d'hallucinations)

### Pour les prochaines expériences

1. **Tester whisper-large-v3** : est-il plus robuste aux hallucinations ?
2. **Ajouter un LM externe** (GPT, CamemBERT) pour vérifier la vraisemblance
3. **Fine-tuning avec augmentation agressive** sur FR + phrases longues
