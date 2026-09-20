# État de l'art

> Document de référence pour le projet **Robustesse des Speech-LLMs**.
> Dernière mise à jour : septembre 2026.

## Table des matières

1. [Introduction](#1-introduction)
2. [Speech-LLMs : des ASR classiques aux modèles génératifs](#2-speech-llms--des-asr-classiques-aux-modèles-génératifs)
3. [Robustesse acoustique en ASR et Speech-LLM](#3-robustesse-acoustique-en-asr-et-speech-llm)
4. [Robustesse aux accents et variétés linguistiques](#4-robustesse-aux-accents-et-variétés-linguistiques)
5. [Augmentation de données et fine-tuning ciblé](#5-augmentation-de-données-et-fine-tuning-ciblé)
6. [Évaluation des Speech-LLMs : métriques et benchmarks](#6-évaluation-des-speech-llms--métriques-et-benchmarks)
7. [Synthèse et positionnement du projet](#7-synthèse-et-positionnement-du-projet)
8. [Références](#8-références)

---

## 1. Introduction

La reconnaissance automatique de la parole (ASR) a connu une transformation majeure
avec l'avènement des modèles auto-supervisés (Wav2Vec2, HuBERT, WavLM) puis des
modèles encodeur-décodeur à grande échelle (Whisper, Qwen2-Audio). Ces derniers,
souvent qualifiés de **Speech-LLMs**, combinent un encodeur acoustique avec un
décodeur de langage, héritant des capacités des grands modèles de langage (LLM)
tout en traitant directement le signal audio.

Si ces modèles atteignent des performances remarquables sur des benchmarks
standards (LibriSpeech, Common Voice, FLEURS), leur **robustesse** face à des
conditions réelles dégradées — bruit urbain, réverbération, micro de mauvaise
qualité, accents sous-représentés — reste **insuffisamment caractérisée de façon
systématique**. Ce document dresse l'état de l'art sur ces questions et positionne
la contribution du projet.

---

## 2. Speech-LLMs : des ASR classiques aux modèles génératifs

### 2.1 Modèles ASR classiques (HMM-GMM, hybrides DNN-HMM)

Avant 2012, les systèmes ASR reposaient sur des modèles **HMM-GMM** (Hidden
Markov Models — Gaussian Mixture Models). Ces systèmes nécessitaient :
- une ingénierie manuelle de features (MFCC, PLP)
- un alignement forcé avec un lexique et un modèle de langage
- des pipelines complexes (GMM → DNN → LM)

Ils étaient **fragiles** face au bruit, aux accents et aux variations de
conditions d'enregistrement, avec des taux d'erreur de 20-40 % en conditions
réelles.

### 2.2 Apprentissage auto-supervisé (Wav2Vec2, HuBERT, WavLM)

**Wav2Vec2** (Baevski et al., 2020) a introduit l'apprentissage auto-supervisé
sur des représentations audio, permettant un **pré-entraînement sur de grandes
quantités d'audio non annoté**. La variante **XLSR-53** (Conneau et al., 2021)
étend cette approche à **53 langues**, ce qui en fait un modèle **multilingue**
efficace. Ses performances sur LibriSpeech (WER < 2 %) ont marqué une rupture.

**HuBERT** (Hsu et al., 2021) et **WavLM** (Chen et al., 2022) ont raffiné cette
approche, avec des gains supplémentaires en robustesse au bruit et aux locuteurs.

Cependant, ces modèles restent **sensibles aux conditions acoustiques** non
vues à l'entraînement et ne disposent **pas de capacités génératives**
multilingues aussi étendues que les modèles encodeur-décodeur.

### 2.3 Modèles encodeur-décodeur à grande échelle (Whisper)

**Whisper** (Radford et al., 2022) est un modèle **encodeur-décodeur** entraîné
sur **680 000 heures** d'audio multilingue et multitâche (transcription,
traduction, détection de langue). Ses principales innovations :

- **Robustesse native** grâce à un entraînement sur données massives et variées
- **Multilingue** (99 langues, dont l'arabe, le français, l'anglais)
- **Zéro-shot** : pas de fine-tuning nécessaire pour la plupart des langues
- **Sortie texte libre**, intégrant ponctuation et casse

Whisper a rapidement été adopté comme **baseline** dans la littérature et comme
**composant ASR** dans des pipelines plus larges (RAG, agents vocaux).

### 2.4 Speech-LLMs instructibles (Qwen2-Audio, SpeechT5)

**SpeechT5** (Ao et al., 2021) propose une architecture unifiée pour ASR, TTS
et conversion vocale, avec un encodeur-décodeur partagé et des pré-nets
spécifiques à chaque modalité.

**Qwen2-Audio** (Chu et al., 2024) est un **LLM multimodal** capable de traiter
audio et texte de manière unifiée. Il peut répondre à des instructions complexes
("transcris cet audio", "résume cette conversation", "traduis en anglais") et
hérite des capacités de raisonnement de Qwen2.

Ces modèles ouvrent la voie à des applications interactives, mais leur
**robustesse aux perturbations** est encore peu étudiée.

### 2.5 Positionnement

| Modèle | Type | Params | Multilingue | Instructible |
|--------|------|--------|-------------|--------------|
| Wav2Vec2-XLSR-53 | CTC | 317 M | ✅ (53 langues) | ❌ |
| Whisper-small | Enc-Déc | 244 M | ✅ (99 langues) | ⚠️ via prompts |
| Whisper-large-v3 | Enc-Déc | 1.55 B | ✅ (99 langues) | ⚠️ |
| SpeechT5 | Enc-Déc unifié | 250 M | ❌ | ❌ |
| Qwen2-Audio-7B | LLM multimodal | 7 B | ✅ | ✅ |

---

## 3. Robustesse acoustique en ASR et Speech-LLM

### 3.1 Types de perturbations acoustiques

La littérature distingue plusieurs familles de perturbations :

1. **Bruit additif** (bruit blanc, bruit rose, bruit de foule, bruit de rue)
   — modélisé par un SNR (Signal-to-Noise Ratio) en dB
2. **Réverbération** (convolution avec une réponse impulsionnelle)
3. **Clipping / saturation** (écrêtage des pics)
4. **Changements de vitesse / pitch** (speed perturbation, SpecAugment)
5. **Compression audio** (codecs bas débit, MP3, Opus)
6. **Perturbations adversariales** (attaques ciblées, imperceptibles)

### 3.2 Robustesse de Whisper

**Radford et al. (2022)** rapportent une dégradation mesurée de Whisper sur
bruit additif, mais **limitée** grâce à l'entraînement sur données variées.
Cependant :

- Les perturbations **non vues** (bruit industriel, réverbération forte) restent
  problématiques.
- **Koenecke et al. (2024)** ont montré que Whisper **hallucine** sur des
  silences, de la musique, ou des enregistrements de mauvaise qualité —
  parfois en inventant des phrases complètes.
- **Frieske & Swietlicka (2023)** ont quantifié une **dégradation de 10-30 %
  du WER** sur Common Voice avec bruit réaliste.

### 3.3 Robustesse de Wav2Vec2

**Hsu et al. (2021)**, **Chen et al. (2022)** et **Shor et al. (2022)** ont
montré que Wav2Vec2 est **plus sensible au bruit** que Whisper, mais :

- Il se **fine-tune plus facilement** sur données bruitées
- Il est **plus léger** (317 M vs 1.55 B)
- Il ne souffre **pas d'hallucinations** (sortie CTC contrainte)

### 3.4 Benchmark de robustesse

Plusieurs benchmarks ont été proposés :

- **CHiME-6/7/8** : robustesse en environnement domestique multi-micro
- **Voicebank-DEMAND** : débruitage
- **MUSAN** : bruit de musique, parole, bruit blanc
- **RIR-Noise** : réverbération + bruit

**Mais** ces benchmarks sont **mono-tâche** (ASR) et ne couvrent pas les
**Speech-LLMs instructibles** comme Qwen2-Audio.

---

## 4. Robustesse aux accents et variétés linguistiques

### 4.1 Biais d'accents

Plusieurs études ont documenté un **biais systématique** des ASR envers
certains accents :

- **Koenecke et al. (2020)** : disparité de WER entre locuteurs noirs et blancs
  aux États-Unis (jusqu'à **2× plus élevé** pour les accents afro-américains).
- **Chung et al. (2021)** : WER plus élevé pour les accents non-natifs en anglais.
- **Yuan et al. (2023)** : biais similaires pour les accents régionaux chinois.

### 4.2 Arabe et dialectes

L'arabe présente une **diglossie** (arabe standard vs dialectes) qui complique
l'ASR :

- **Ali et al. (2017)** : état de l'art ASR arabe avant Wav2Vec2.
- **Mubarak et al. (2020)** : benchmarks pour dialectes arabes (égyptien, levantin, golfe).
- **Talafha et al. (2023)** : évaluation de Whisper sur 15 dialectes arabes —
  **WER de 15 à 60 %** selon le dialecte.

### 4.3 Français et variétés

- **Variétés africaines** (Sénégal, Côte d'Ivoire, Cameroun) : sous-représentées.
- **Accents canadiens** : relativement bien couverts (Common Voice).
- **Accents maghrébins** : peu couverts, malgré une diaspora importante.

### 4.4 Ce que ce projet apporte

Notre projet évalue **systématiquement** la robustesse de plusieurs Speech-LLMs
sur :
- **3 langues** : arabe, français, anglais
- **Accents variés** : standard vs dialectal (arabe égyptien vs standard)
- **Perturbations acoustiques contrôlées**

C'est, à notre connaissance, **une des premières études** combinant ces deux
dimensions (acoustique + accent) sur des **Speech-LLMs instructibles**.

---

## 5. Augmentation de données et fine-tuning ciblé

### 5.1 Techniques d'augmentation classiques

- **Speed perturbation** (Ko et al., 2015) : variation de vitesse (0.9, 1.0, 1.1)
  → gain de 5-10 % relatif sur WER
- **SpecAugment** (Park et al., 2019) : masquage temporel et fréquentiel sur
  les spectrogrammes → gain notable en conditions bruitées
- **Bruit additif** (MUSAN, AudioSet) à SNR variés → robustesse accrue
- **Réverbération** (RIR synthétiques) → robustesse en environnement réel

### 5.2 Augmentation pour Speech-LLMs

Contrairement aux modèles classiques, les Speech-LLMs comme Whisper sont :

- **Pré-entraînés sur des données massives** → l'augmentation simple apporte
  peu
- **Coûteux à fine-tuner** (milliards de paramètres)
- **Sensibles à la distribution** : fine-tuning trop agressif → oubli
  catastrophique

**Stratégies recommandées** :

- **LoRA / QLoRA** : fine-tuning param-efficient (Hu et al., 2021)
- **Adapter layers** (Houlsby et al., 2019)
- **Data augmentation ciblée** sur les conditions dégradées identifiées

### 5.3 Ce que ce projet propose

- **Augmentation ciblée** : appliquer uniquement les perturbations qui
  dégradent significativement les performances (identifiées en phase 2)
- **Fine-tuning param-efficient** : LoRA sur les couches d'attention
- **Évaluation avant/après** : mesurer le gain, le coût computationnel, et la
  stabilité

---

## 6. Évaluation des Speech-LLMs : métriques et benchmarks

### 6.1 Métriques standard

- **WER (Word Error Rate)** : métrique principale en ASR
- **CER (Character Error Rate)** : utile pour les langues à morphologie riche
  (arabe, turc)
- **MER (Match Error Rate)**, **WIL (Word Information Lost)** : variantes
- **BLEU** : pour la traduction (Whisper peut traduire)

### 6.2 Métriques émergentes

- **Latence** : critique pour les applications temps réel
- **Robustesse relative** : ΔWER(bruit) − WER(propre)
- **Hallucination rate** : proportion d'échantillons générés sans correspondance
  avec l'audio (Koenecke et al., 2024)
- **Fairness metrics** : disparité de WER entre groupes démographiques

### 6.3 Benchmarks

- **SUPERB** (Yang et al., 2021) : benchmark multi-tâches auto-supervisé
- **FLEURS** (Conneau et al., 2022) : 102 langues, ASR et traduction
- **ML-SUPERB** (Shi et al., 2023) : version multilingue de SUPERB
- **HEAR** (Turian et al., 2022) : benchmark audio généraliste

### 6.4 Ce que ce projet utilise

- **FLEURS** : pour les langues et accents
- **LibriSpeech** : baseline anglaise propre
- **Common Voice** : variabilité de locuteurs
- **Perturbations contrôlées** : bruit, réverbération, clipping, speed
- **Métriques** : WER, CER, latence, hallucination rate

---

## 7. Synthèse et positionnement du projet

### 7.1 Lacunes identifiées dans la littérature

| Lacune | Description |
|--------|-------------|
| **L1** | Peu d'études systématiques de robustesse sur **Speech-LLMs instructibles** (Qwen2-Audio) |
| **L2** | Les benchmarks existants couvrent **ASR classique** ou **LLM**, rarement les deux |
| **L3** | Combinaison **perturbations acoustiques + accents** rarement étudiée conjointement |
| **L4** | Manque de **pipelines reproductibles** open-source pour ces évaluations |
| **L5** | Peu de travaux sur **arabe dialectal** avec Speech-LLMs |
| **L6** | Analyse fine des **erreurs** (phonèmes, mots, accents) souvent absente |

### 7.2 Contribution du projet

Ce projet vise à **combler ces lacunes** par :

1. **Un benchmark systématique** de 3-4 Speech-LLMs sur :
   - 5 types de perturbations acoustiques (white noise, street noise, reverb, clipping, speed)
   - 3 langues (arabe, français, anglais) et variantes dialectales
   - Multiples niveaux de sévérité (SNR, room scale)

2. **Une analyse d'erreurs fine** :
   - Phonèmes confondus
   - Accents problématiques par langue
   - Mots fréquents mal transcrits
   - Détection d'hallucinations

3. **Une amélioration mesurable** :
   - Augmentation ciblée sur conditions difficiles identifiées
   - Fine-tuning param-efficient (LoRA)
   - Comparaison avant/après

4. **Un pipeline reproductible** :
   - Code open-source (MIT)
   - Configuration versionnée
   - Documentation claire
   - Tests unitaires

5. **Une valorisation scientifique** :
   - Rapport format arXiv (10-15 pages)
   - Soumission à Interspeech / ICASSP / ACL
   - Communication grand public (blog, LinkedIn)

### 7.3 Alignement avec la littérature

| Aspect | Référence principale | Contribution projet |
|--------|---------------------|---------------------|
| Robustesse Whisper | Radford et al., 2022 | Évaluation systématique + analyse erreurs |
| Robustesse Wav2Vec2 | Hsu et al., 2021 | Comparaison avec Speech-LLMs |
| Augmentation | Park et al., 2019 (SpecAugment) | Application ciblée + mesure gain |
| Accents arabe | Talafha et al., 2023 | Extension aux dialectes + comparaison modèles |
| Fine-tuning efficient | Hu et al., 2021 (LoRA) | Application à Speech-LLMs + coût/bénéfice |
| Hallucinations | Koenecke et al., 2024 | Quantification systématique |

---

## 8. Références

### Speech-LLMs et ASR

1. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020.

2. **Conneau, A., Baevski, A., Collobert, R., Mohamed, A., & Auli, M. (2021).** *Unsupervised Cross-lingual Representation Learning for Speech Recognition.* Interspeech 2021. [XLSR-53]

3. **Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., et al. (2021).** *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* IEEE/ACM TASLP.

4. **Chen, S., Wang, C., Chen, Z., et al. (2022).** *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* IEEE JSTSP.

5. **Radford, A., Kim, J. W., Xu, T., et al. (2022).** *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML 2023. [Whisper]

6. **Ao, J., Wang, R., Zhou, L., et al. (2021).** *SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing.* ACL 2022.

7. **Chu, Y., Xu, J., Zhou, X., et al. (2024).** *Qwen2-Audio Technical Report.* arXiv:2407.10759.

### Robustesse acoustique des Speech-LLMs

8. **Aboietta, A., Sharshar, A., Nafea, Y., & Shehata, S. (2025).** *Bridging ASR and LLMs for Dysarthric Speech Recognition: Benchmarking Self-Supervised and Generative Approaches.* Interspeech 2025. [Démontre l'apport du décodage LLM pour la restauration phonétique en conditions dégradées] 

9. **Ankita, Bharadwaj, K. M., et al. (2026).** *Exploring LoRA variants to adapt whisper models for robust recognition of children's speech.* Speech Communication, 181. [Comparaison systématique de LoRA, QLoRA, AdaLoRA, DoRA sur Whisper] 

10. **Althubaiti, S., Lodagala, V. S., Clark, T., et al. (2025).** *Octopus: Towards Building the Arabic Speech LLM Suite.* ArabicNLP 2025, ACL. [Modèles Speech-LLM modulaires pour l'arabe et l'anglais, avec robustesse au code-switching] 

11. **Anonymous. (2026).** *Robustness Analysis of Speech LLMs under Diverse Acoustic Conditions.* arXiv preprint. [Résultats : dégradation sévère des tâches de raisonnement sous bruit ; collapse fonctionnel de l'ASR à K=4] 

12. **Awobade, B., et al. (2025).** *AfriSpeech-MultiBench: A Domain-Specific Evaluation Suite for African English Accents.* ACL Anthology. [Benchmark multi-domaines sur 100+ accents africains, incluant robustesse aux hallucinations] 

13. **Carrick, J. E., et al. (2025).** *Speech-Controlled Smart Speaker for Accurate, Real-Time Health and Care Record Management.* IWSDS 2025, ACL. [Fine-tuning Whisper sur dialectes britanniques : WER réduit de 16.8 à 1.0 ; gestion des hallucinations via seuil SNR] 

14. **Frieske, R., & Swietlicka, A. (2023).** *Benchmarking Whisper's Robustness to Realistic Noise.* arXiv preprint. [Note : à vérifier pour identifiant précis]

15. **Koenecke, A., Choi, A., Mei, K., et al. (2024).** *Careless Whisper: Speech-to-Text Hallucination Harms.* FAccT 2024. [Analyse des hallucinations de Whisper sur silences et bruit]

### Accents et variétés linguistiques

16. **Koenecke, A., Nam, A., Lake, E., et al. (2020).** *Racial Disparities in Automated Speech Recognition.* PNAS. [Disparité de WER selon l'accent]

17. **Talafha, B., Chen, C., Sawaf, H., et al. (2023).** *N-Shot Benchmarking of Whisper on Diverse Arabic Speech Recognition.* Interspeech 2023. [Évaluation de Whisper sur 15 dialectes arabes]

18. **Talafha, B., Abu Alhassan, A., & Abdul-Mageed, M. (2026).** *Zero-Shot Context-Aware ASR for Diverse Arabic Varieties.* Findings of ACL 2026. [Réduction relative de WER : 22.29 % MSA, 20.54 % MSA accentué, 9.15 % arabe dialectal]

19. **Mubarak, H., Darwish, K., & Magdy, W. (2020).** *Arabic Dialect Identification in the Wild.* LREC.

20. **Ali, A., Zhang, Y., Card, P., et al. (2017).** *A Complete Kaldi Recipe for Building Arabic Speech Recognition Systems.* SLTU.

### Augmentation de données et fine-tuning ciblé

21. **Ko, T., Peddinti, V., Povey, D., & Khudanpur, S. (2015).** *Audio Augmentation for Speech Recognition.* Interspeech. [Speed perturbation]

22. **Park, D. S., Chan, W., Zhang, Y., et al. (2019).** *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech. [Masquage temporel et fréquentiel]

23. **Hu, E. J., Shen, Y., Wallis, P., et al. (2021).** *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [Fine-tuning param-efficient]

24. **Houlsby, N., Giurgiu, A., Jastrzebski, S., et al. (2019).** *Parameter-Efficient Transfer Learning for NLP.* ICML. [Adapter layers]

25. **Lee, J., et al. (2025).** *Speak & Spell: LLM-Driven Controllable Phonetic Error Augmentation for Robust Dialogue State Tracking.* AACL-IJCNLP 2025. [Augmentation contrôlée d'erreurs phonétiques pour améliorer la robustesse] 

26. **Anonymous. (2025).** *Noise-Augmented QLoRA Fine-Tuning Improves ASR Robustness.* Zenodo preprint. [Résultats : 47 % de réduction relative de la dégradation sur Qwen2.5-3B avec 60 % de données bruitées ; accessible en ~10h sur RTX 4070 Super] 

### Benchmarks et évaluation

27. **Yang, S.-w., Chi, P.-H., Chuang, Y.-S., et al. (2021).** *SUPERB: Speech Processing Universal PERformance Benchmark.* Interspeech.

28. **Conneau, A., Ma, M., Khanuja, S., et al. (2022).** *FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech.* SLT.

29. **Shi, J., Berrebbi, D., Chen, W., et al. (2023).** *ML-SUPERB: Multilingual Speech Universal PERformance Benchmark.* Interspeech.

30. **Turian, J., Shrivastava, A., & Wu, K. (2022).** *HEAR: Holistic Evaluation of Audio Representations.* PMLR.

31. **Snyder, D., Chen, G., & Povey, D. (2015).** *MUSAN: A Music, Speech, and Noise Corpus.* arXiv:1510.08484.

32. **FreedomIntelligence. (2025).** *DitingBench: Evaluating Speech Comprehension of Speech LLMs.* arXiv:2410.13268. [Benchmark évaluant les Speech-LLMs sur ASR, termes juridiques/médicaux, transcription de paroles ; montre un écart important entre humains et modèles sur les termes spécialisés] 

### Applications critiques et domaines spécifiques

33. **Anonymous. (2025).** *A Critical Analysis of End-to-End ASR Models for Air Traffic Control Speech.* IEEE Xplore (conférence Nov. 2025, ajouté Mars 2026). [WhisperAI-small hallucine parfois en générant du texte non-anglais ; fine-tuning sur un accent ne garantit pas la robustesse sur d'autres accents, même domaine] 

34. **Anonymous. (2025).** *Robust Dysarthric Speech Recognition with GAN Enhancement and LLM Correction.* Advanced Intelligent Systems, Wiley. [Architecture Llama*-DSR : Whisper-medium gelé + Llama-3.1-8B avec ~27M paramètres LoRA pour la reconnaissance de parole dysarthrique] 

35. **Anonymous. (2025).** *Closing the Gap Between Text and Speech Understanding in LLMs.* arXiv:2510.13632. [Analyse du « text-speech gap » et du phénomène de « forgetting » des capacités textuelles après adaptation à la parole] 

36. **Anonymous. (2025).** *Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs.* arXiv:2604.16659. [Fine-tuning Kimi-Audio avec LoRA sur SD-QA + bruit ; découverte contre-intuitive : le bruit de café réduit le JSR de 3.66 %, tandis que le bruit de trafic l'augmente de 13.84 % — la direction du shift acoustique compte plus que la magnitude] 

---

## Notes pour la rédaction du rapport final

Ce document servira de base pour :

1. **Section 2 (Related Work)** du rapport — 3-4 pages
2. **Justification des choix** : Whisper (état de l'art), Wav2Vec2 (léger, contrôlable), Qwen2-Audio (instructible, récent)
3. **Positionnement** : combinaison perturbations + accents + Speech-LLMs
4. **Bibliographie** : déjà structurée (25 références solides)

**Prochaines étapes de documentation** :
- [ ] Compléter chaque section avec des exemples chiffrés
- [ ] Ajouter des tableaux comparatifs de performances rapportées
- [ ] Intégrer les résultats du projet au fur et à mesure
- [ ] Préparer le rapport final (format arXiv)

---

**Auteur** : Safae Wardi — Master Big Data et Systèmes Intelligents, USMBA Fès