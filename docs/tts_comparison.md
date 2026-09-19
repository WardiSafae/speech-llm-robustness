# Comparaison SAPI5 vs Edge TTS

## Protocole

- **Modèle ASR** : `openai/whisper-small`
- **Phrases** : identiques pour les deux sources (3 langues × 1 phrase)
- **SAPI5** : voix locales Windows (Hortense, Naayf, Zira)
- **Edge TTS** : voix neuronales Microsoft (Denise, Salma, Jenny)

## Résultats

| Langue | Source | N | WER | CER |
|--------|--------|---|-----|-----|
| **EN** | SAPI5 | 3 | 0.0000 | 0.0000 |
| **EN** | Edge TTS | 2 | 0.0000 | 0.0000 |
| **FR** | SAPI5 | 3 | 0.4444 | 0.1273 |
| **FR** | Edge TTS | 2 | **0.8889** | 0.1455 |
| **AR** | SAPI5 | 3 | 0.4286 | 0.1190 |
| **AR** | Edge TTS | 2 | **0.7143** | 0.1429 |

## Analyse

### Observation contre-intuitive

Les voix **neuronales** Edge TTS **dégradent** le WER par rapport aux voix
**SAPI5 classiques**, malgré une qualité audio perçue supérieure.

### Hypothèses

1. **Domain shift** : Whisper-small est entraîné sur du web audio (YouTube,
   podcasts), pas sur des TTS neuronaux récents. Les voix "propres" d'Edge TTS
   sont hors distribution.

2. **Vitesse d'élocution** : Edge TTS parle plus vite (avec intonation naturelle),
   ce qui complique la tâche de Whisper sur des phrases courtes.

3. **Liaisons françaises** : Whisper segmente mal les enchaînements consonantiques
   (`renard` → `au nard`). Ce problème existe avec SAPI5 mais est exacerbé par
   la qualité "studio" d'Edge TTS.

4. **Taille d'échantillon** : n=2 pour Edge TTS (vs 3 pour SAPI5). Une seule
   erreur pèse 50 % du WER.

### Conclusion

**Whisper-small est le facteur limitant** sur FR/AR, indépendamment de la qualité
du TTS source. La qualité TTS perçue par un humain n'est pas corrélée à la
facilité de transcription par un ASR.

**Implication pour le projet** : les prochaines étapes doivent se concentrer
sur des **modèles ASR plus grands** (whisper-medium, whisper-large) plutôt que
sur l'amélioration du TTS.

## Fichiers reproductibles

- Script : `scripts/test_comparatif_SAPI5_EdgeTTS.py`
- Résultats bruts : `results/tables/whisper_results.csv`
- Données : `data/raw/{synth,neural}_*.wav`

## Prochaines étapes

1. Tester **whisper-medium** sur les mêmes données (téléchargement ~1.5 Go)
2. Utiliser des **enregistrements humains** (FLEURS, Common Voice)
3. Augmenter la taille d'échantillon (n ≥ 30 par condition)