## Résultats préliminaires

**Modèle** : `openai/whisper-small` (244 M params, CPU-only)
**Données** : 9 échantillons TTS locaux (3 langues × 3 phrases)

| Langue | WER | CER | Latence |
|--------|-----|-----|---------|
| 🇬🇧 Anglais | 0.00 | 0.00 | 12.9 s |
| 🇸🇦 Arabe | 0.25 | 0.08 | 14.8 s |
| 🇫🇷 Français | 0.45 | 0.10 | 14.4 s |
| **Global** | **0.28** | **0.09** | **14.0 s** |

**Interprétation** : forte disparité selon la langue. Le français, malgré une
voix TTS native, produit des erreurs phonétiques (`brun` → `brin`,
`paresseux` → `par-essu`) qui motivent un fine-tuning ciblé. L'arabe,
pourtant moins doté en TTS de qualité, obtient un WER meilleur — résultat
contre-intuitif qui sera analysé dans le rapport.


## Résultats préliminaires — Fine-tuning Whisper-tiny (FR)

### Protocole
- Modèle : `openai/whisper-tiny` (39 M params)
- Dataset : 3 échantillons TTS français (même phrase répétée)
- Split : train=2, eval=1
- Steps : 10 (5 epochs)

### Résultats

| Métrique | Avant | Après | Δ |
|----------|-------|-------|---|
| WER | 0.444 | 0.000 | -100 % |
| CER | 0.127 | 0.000 | -100 % |
| train_loss | — | 2.93 | — |

### ⚠️ Limites et interprétation

Le WER=0 obtenu **ne démontre pas** une capacité de généralisation :
1. Les 3 échantillons partagent la **même transcription**, ce qui favorise
   la mémorisation.
2. Le dataset est **trop petit** (2 samples de train).
3. Aucune évaluation sur des **phrases non vues**.

**Ce résultat valide uniquement le pipeline technique de fine-tuning.**
Une évaluation rigoureuse nécessitera un dataset >100 échantillons avec
des transcriptions variées, ainsi qu'un split train/val/test disjoint.

### Prochaine étape
Constituer un dataset plus large (FLEURS, Common Voice) pour mesurer
la vraie amélioration de généralisation.


