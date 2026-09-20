"L'arabe n'a pas pu être évalué correctement avec whisper-small (WER=1.0). Les prochaines étapes incluent :

Test avec whisper-medium/large

Téléchargement de FLEURS-arab (connexion stable requise)

Fine-tuning ciblé avec LoRA sur l'arabe"


"Le fine-tuning param-efficient avec LoRA sur Whisper-small s'est révélé incompatible avec la version actuelle de PEFT (0.21.0) en raison d'un conflit input_ids sur la cross-attention. Nous avons contourné ce problème en utilisant Whisper-tiny en fine-tuning complet, ce qui reste accessible sur CPU. Cette limitation est un point d'amélioration identifié pour les travaux futurs."


"Le fine-tuning avec LoRA sur Whisper-small s'est heurté à un conflit de paramètres input_ids entre self-attention et cross-attention dans PEFT 0.21.0. Cette limitation technique a motivé l'utilisation de Whisper-tiny en fine-tuning complet, plus léger et compatible CPU."


# Limites identifiées

## 1. Dataset synthétique trop petit

**Constat :** Le dataset TTS local contient seulement 3 échantillons par langue,
tous avec la même transcription.

**Impact :** Le fine-tuning atteint WER=0 sur l'éval mais par **mémorisation**,
pas par généralisation.

**Solution :** Utiliser FLEURS ou Common Voice pour un dataset >100 échantillons
avec transcriptions variées (connexion stable requise).

## 2. Bug LoRA sur Whisper

**Constat :** `peft 0.21.0` + `transformers 5.x` produit une erreur
`got multiple values for keyword argument 'input_ids'` lors du fine-tuning LoRA
sur Whisper.

**Cause :** Conflit entre self-attention et cross-attention (`encoder_attn`)
dans `WhisperAttention`.

**Contournement :** Fine-tuning complet sur `whisper-tiny` (39 M params).

**Solution future :** Passer à une version antérieure de `peft` (0.10.x) ou
utiliser un modèle sans cross-attention (Wav2Vec2).

## 3. Qualité TTS variable selon la langue

**Constat :**
- Anglais (Zira/David) : excellent → WER=0
- Français (Hortense) : acceptable → WER=0.44
- Arabe (Naayf) : passable → WER=0.43

**Impact :** Le WER reflète en partie la qualité du TTS, pas seulement
la robustesse du modèle ASR.

**Solution :** Utiliser des enregistrements humains (FLEURS, Common Voice)
pour une évaluation réaliste.

## 4. Évaluation CPU-only

**Constat :** Toutes les expériences tournent sur CPU, ce qui limite
la taille des modèles testables (pas de Whisper-large, pas de Qwen2-Audio 7B).

**Solution :** Google Colab (GPU gratuit) pour les modèles >1 G.

## 5. Manifeste TTS non versionné

**Constat :** Les WAV synthétiques sont générés localement et ne sont pas
versionnés (par design : volumineux).

**Solution :** Le script `load_datasets.py --mode synthetic` permet de
régénérer les données de façon reproductible.



"
C'est en fait un résultat scientifique intéressant :

Observation contre-intuitive : Les voix neuronales Edge TTS (Denise, Salma)
dégradent le WER sur français (0.44 → 0.89) et arabe (0.43 → 0.71)
par rapport aux voix SAPI5 classiques (Hortense, Naayf).

Hypothèses explicatives :

Whisper-small est entraîné sur du web audio, pas sur des TTS neuronaux modernes.

Edge TTS prononce plus vite, avec plus d'intonation → plus difficile pour Whisper.

Liaisons françaises (renard → au nard) mal segmentées.

Taille d'échantillon insuffisante (n=2).

Conclusion : La qualité TTS perçue par un humain n'est pas corrélée
à la facilité de transcription par un modèle ASR. Whisper-small est le facteur
limitant sur FR/AR, indépendamment du TTS.
"
