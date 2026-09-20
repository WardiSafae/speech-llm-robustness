# Robustesse aux perturbations acoustiques — whisper-medium

## Protocole

- **Dataset** : FLEURS (Google), 100 échantillons/langue
- **Perturbations** : 4 types × 3 sévérités = 12 conditions
- **Modèle** : `openai/whisper-medium` (GPU T4)
- **Échantillons évalués** : 3600 fichiers (100/langue × 12 perturbations)
- **Normalisation** : lowercase + ponctuation retirée

## Baseline (speed=1.0, non perturbé)

| Langue | WER | vs EN |
|--------|-----|-------|
| EN | 0.045 | référence |
| FR | 0.090 | +0.045 |
| AR | **0.226** | **+0.181** |

**L'arabe est 5× plus difficile que l'anglais** en conditions propres.

## Résultats par perturbation (WER global)

| Perturbation | Sévérité | WER | ΔWER | Niveau |
|--------------|----------|-----|------|--------|
| **white_noise** | snr15 | 0.193 | +0.072 | 🟡 |
| | snr10 | 0.218 | +0.098 | 🟡 |
| | **snr5** | **0.340** | **+0.220** | 🔴 |
| **reverb** | room0.1 | 0.145 | +0.025 | 🟢 |
| | room0.3 | 0.242 | +0.122 | 🟡 |
| | **room0.5** | **0.440** | **+0.319** | 🔴🔴🔴 |
| **clipping** | clip0.3 | 0.124 | +0.003 | 🟢 |
| | clip0.5 | 0.114 | -0.006 | 🟢 |
| | clip0.7 | 0.115 | -0.006 | 🟢 |
| **speed** | spd0.9 | 0.251 | +0.131 | 🟡 |
| | spd1.0 | 0.114 | baseline | ✅ |
| | spd1.1 | **0.303** | **+0.182** | 🔴 |

## 🎯 Découvertes majeures

### 1. Clipping : AUCUN impact

Whisper-medium est **totalement robuste au clipping** (ΔWER ≈ 0).

**Hypothèse** : Whisper utilise des **features Mel** (spectrogrammes). Le clipping
modifie l'**amplitude temporelle** mais **préserve les fréquences**. Comme Mel est
invariant aux changements d'amplitude globaux, Whisper "voit" le même spectrogramme.

**Implication** : Le clipping n'est pas une menace pour whisper-medium.

### 2. Reverb : la perturbation la plus destructrice

`reverb room0.5` : **ΔWER = +0.319** (×3.9 par rapport au baseline).

**Hypothèse** : La convolution avec une réponse impulsionnelle (IR) crée un
**smearing temporel** qui altère profondément les formants et ajoute des réflexions.
Contrairement au clipping, la reverb **modifie le spectrogramme Mel**.

**Implication** : La reverb est la perturbation critique à cibler en fine-tuning.

### 3. Speed : asymétrique

- `speed 0.9` (ralentir) : ΔWER = +0.131
- `speed 1.1` (accélérer) : ΔWER = +0.182

**Accélérer est pire que ralentir.**

**Hypothèse** : Whisper est entraîné sur du débit naturel. Accélérer **compresse
les formants** et crée des artefacts. Ralentir étire le signal mais Whisper a
probablement vu plus de ralentis.

### 4. White noise : dégradation progressive

- snr15 (doux) : ΔWER = +0.072
- snr10 (modéré) : ΔWER = +0.098
- snr5 (sévère) : ΔWER = +0.220

**Relation logarithmique** entre SNR et WER, cohérente avec la littérature.

## 📊 Hiérarchie des perturbations

**Par impact décroissant :**

```

reverb room0.5 : +0.319 🔴🔴🔴

white_noise snr5 : +0.220 🔴🔴

speed 1.1 : +0.182 🔴

speed 0.9 : +0.131 🟡

reverb room0.3 : +0.122 🟡

white_noise snr10 : +0.098 🟡

white_noise snr15 : +0.072 🟡

reverb room0.1 : +0.025 🟢

clipping (tous) : ~0.000 🟢

```

**Reverb > White noise > Speed >> Clipping**

## 🎯 Recommandations

### Pour le fine-tuning ciblé

**Priorités :**
1. **Reverb** (room0.3 et room0.5) → gain maximal
2. **White noise** (SNR 5-10 dB)
3. **Speed** (0.9 et 1.1)

**Ignorer le clipping** : aucune amélioration possible.

### Pour la production

- **Filtrer les audios** avec SNR < 10 dB (sinon dégradation > 10 %)
- **Débruiter** les audios réverbérés (déréverbération DSP)
- **Normaliser le débit** autour de 1.0 (speed perturbation entre 0.95 et 1.05)

### Pour les prochaines expériences

1. **Fine-tuner whisper-medium** avec augmentation ciblée (reverb + white noise)
2. **Tester d'autres modèles** (Wav2Vec2, whisper-large-v3) sur les mêmes perturbations
3. **Analyser par langue** : l'arabe est-il plus sensible à certaines perturbations ?

## Fichiers reproductibles

- Manifest : `data/processed/fleurs_perturbed/manifest_perturbed.tsv`
- Résultats : `perturbed_medium_results.csv`
- Notebook Colab : (à publier)



## Analyse par langue (détail)

### Baseline (speed=1.0, conditions propres)

| Langue | WER | Écart vs EN |
|--------|-----|-------------|
| EN | 0.0452 | référence |
| FR | 0.0904 | +0.045 |
| AR | **0.2262** | **+0.181** |

**L'arabe est 5× plus difficile que l'anglais** en baseline.

### 🎯 Découverte : le français dégrade PLUS que l'arabe (en relatif)

**Contre-intuitif !** Analysons la dégradation **relative** (ratio WER perturbé / WER baseline) :

| Perturbation | FR (ratio) | AR (ratio) | Lequel souffre plus ? |
|--------------|-----------|-----------|----------------------|
| white_noise snr5 | **×4.9** | ×1.9 | 🔴 FR |
| speed 1.1 | **×3.6** | ×2.1 | 🔴 FR |
| reverb room0.5 | **×6.3** | ×2.6 | 🔴 FR |

**Interprétation** : le baseline FR est bas (0.090), donc chaque dégradation
absolue (+0.35) est énorme en relatif. L'arabe, déjà à 0.226, dégrade moins
**proportionnellement**.

**Implication pour le fine-tuning** : le français a **plus de marge
de progression absolue** que l'arabe. Cibler FR en priorité.

### 🎯 Découverte : l'arabe est le plus robuste au clipping

| Clipping | EN (ΔWER) | FR (ΔWER) | **AR (ΔWER)** |
|----------|-----------|-----------|---------------|
| clip0.3 | +0.007 | +0.022 | **-0.003** (nul) |
| clip0.5 | 0.000 | +0.001 | 0.000 |
| clip0.7 | 0.000 | +0.002 | 0.000 |

**AR a un ΔWER négatif** à clip0.3 → bruit statistique → **robustesse totale**.

**Hypothèse** : la structure phonétique arabe (voyelles longues, consonnes
emphatiques) survit au clipping Mel.

### 🎯 Découverte : reverb, la pire pour FR et AR

À `room0.5` :

| Langue | ΔWER | Ratio de dégradation |
|--------|------|---------------------|
| **EN** | **+0.096** | ×3.1 |
| **FR** | **+0.475** | ×6.3 |
| **AR** | **+0.371** | ×2.6 |

**L'anglais résiste 4× mieux** que le français et l'arabe à la reverb.

**Hypothèse** : Whisper a été entraîné sur massive quantité d'anglais en
environnement varié (podcasts, YouTube, réunions). FR et AR sont
sous-représentés → moins de robustesse apprise.

### Hiérarchie des perturbations par langue

| Langue | Pire perturbation | Meilleure |
|--------|-------------------|-----------|
| **EN** | speed 1.1 (+0.071) | clipping (0) |
| **FR** | **reverb 0.5 (+0.475)** | clipping (+0.002) |
| **AR** | **speed 1.1 (+0.252)** | clipping (~0) |

**Note** : FR et AR ont des sensibilités différentes — FR souffre plus de la
reverb, AR plus du speed.

## 🎯 Recommandations de fine-tuning

**Priorités (par impact et marge de progression) :**

1. **Français + reverb room0.5** → gain potentiel de -0.40 WER
2. **Français + white_noise snr5** → gain potentiel de -0.30 WER
3. **Arabe + speed 1.1** → gain potentiel de -0.25 WER
4. **Arabe + reverb room0.5** → gain potentiel de -0.30 WER

**Ignorer le clipping** — aucune amélioration possible.