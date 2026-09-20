# Paradoxe large-v3 : meilleur WER, plus d'hallucinations

## Le paradoxe en une phrase

> Large-v3 est **30 % meilleur** que medium en WER, mais **produit 47× plus
> d'hallucinations**.

## Données

| Métrique | medium | large-v3 | Δ |
|----------|--------|----------|---|
| WER baseline (ALL) | 0.114 | **0.080** | -30 % ✅ |
| WER perturbé max | 8.68 | **1.50** | -83 % ✅ |
| Hallucinations | **5** (0.14 %) | **237** (6.6 %) | **+47×** ⚠️ |

## Détail des hallucinations par langue

| Langue | medium | large-v3 | Δ |
|--------|--------|----------|---|
| **EN** | 0 (0 %) | **120 (10.0 %)** | +120 🚨 |
| **FR** | 5 (0.14 %) | **68 (5.7 %)** | +63 |
| **AR** | 0 (0 %) | **49 (4.1 %)** | +49 |
| **Total** | 5 | **237** | +232 |

**Renversement** : avec medium, c'était FR le pire. Avec large-v3, c'est **EN**.

## Hypothèses explicatives

### H1 — Large-v3 est plus "créatif"

Large-v3 génère des sorties **plus longues, naturelles et grammaticales**.
Sur audios dégradés, il **invente** des phrases cohérentes au lieu de tronquer.

**Medium** échoue **proprement** (hypothèses courtes ou vides).
**Large-v3** échoue **créativement** (inventions complètes).

### H2 — Distribution EN massive

Large-v3 a vu **plus d'anglais** que medium. Il est plus "confiant" en anglais
et **invente** plus facilement.

### H3 — Beam search plus exploratoire

Large-v3 utilise un décodage plus complexe → plus de dérive sur audios ambigus.

### H4 — Sensibilité aux perturbations longues

Sur `reverb 0.5`, large-v3 **abandonne** (sortie "Merci.") sur certaines phrases
longues FR.

## Cas notable : "Merci."

**4 cas** de troncature massive sur FR + reverb 0.5 :

| Racine | Ref_len | Hyp |
|--------|---------|-----|
| fr_052 | 27 | "Merci." |
| fr_064 | 25 | "Merci." |
| fr_066 | 20 | "Merci à tous." |
| fr_097 | 19 | "Merci." |

**Pattern** : 100 % FR + 100 % reverb 0.5.

**Interprétation** : Large-v3 **détecte** que l'audio est trop dégradé et
**refuse** de transcrire → sortie stéréotypée.

## Implication pour la production

### Recommandation

| Cas d'usage | Modèle recommandé |
|-------------|-------------------|
| WER minimal | large-v3 |
| **Fiabilité (peu d'hallucinations)** | **medium** ✅ |
| Production critique | medium |
| WER + détecteur | large-v3 + filtre |

**Verdict** : en production, **medium est préférable** malgré un WER plus élevé.

## Implication pour la recherche

**Large-v3 n'est pas strictement "meilleur"** que medium. Il a un **profil
différent** : plus performant en moyenne, mais plus risqué sur les cas extrêmes.

**Cette découverte remet en question la notion de "scaling win"**.
Les modèles plus grands ne sont pas **systématiquement** meilleurs.

## Recommandations de fine-tuning

### Priorités

1. **Large-v3 sur FR + reverb 0.5** (4 cas "Merci.")
2. **Large-v3 sur EN** (10 % d'hallucinations)
3. **Large-v3 sur AR** (4.1 % d'hallucinations)

### Stratégie

- **Augmentation agressive** sur reverb et white_noise
- **LoRA** pour éviter le catastrophic forgetting
- **Détecteur d'hallucinations** en post-traitement

## Prochaines expériences

1. **Tester whisper-large-v2** : hallucine-t-il moins que v3 ?
2. **Analyser les hallucinations EN** (10 % semble énorme)
3. **Fine-tuning ciblé** sur les cas extrêmes
4. **Ajouter un LM externe** pour vérifier la vraisemblance
