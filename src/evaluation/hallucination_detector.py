"""
Détecteur d'hallucinations pour Whisper.

Utilisable en production pour filtrer les transcriptions suspectes.

Trois signaux :
    1. Longueur anormale (hypothèse >> référence)
    2. Répétition excessive (n-grammes répétés)
    3. Vraisemblance faible (optionnel, nécessite un LM externe)

Usage :
    from src.evaluation.hallucination_detector import detect_hallucination
    
    result = detect_hallucination(reference, hypothesis)
    if result["is_hallucination"]:
        print(f"⚠️ {result['reason']}")
"""

import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _n_gram_repetition_ratio(tokens: list[str], n: int = 2) -> float:
    """Ratio de n-grammes répétés (1.0 = tout répété)."""
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counter = Counter(ngrams)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(ngrams)


def _max_token_repetition_ratio(tokens: list[str]) -> float:
    """Ratio du token le plus répété (1.0 = tout le même mot)."""
    if not tokens:
        return 0.0
    counter = Counter(tokens)
    return counter.most_common(1)[0][1] / len(tokens)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HallucinationConfig:
    """Seuils pour la détection."""
    # Longueur
    max_len_ratio: float = 1.5            # hyp / ref > 1.5 → suspect
    min_len_ratio: float = 0.3          # hyp / ref < 0.3 → suspect
    min_hyp_tokens: int = 5             # Ignorer les très courtes hypothèses
    
    # Répétition
    max_token_rep: float = 0.30         # Un seul token répété > 30%
    max_bigram_rep: float = 0.20        # Un bigramme répété > 30%
    max_trigram_rep: float = 0.15       # Un trigramme répété > 20%
    
    # Longueur minimale de référence pour appliquer les règles
    min_ref_len: int = 3                # Ne pas juger les refs < 3 mots


@dataclass
class HallucinationResult:
    """Résultat de la détection."""
    is_hallucination: bool
    reason: Optional[str]
    confidence: float                    # 0.0 - 1.0
    signals: dict                        # Détails des signaux
    ref_len: int
    hyp_len: int
    len_ratio: float
    
    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Détection
# ---------------------------------------------------------------------------

def detect_hallucination(
    reference: str,
    hypothesis: str,
    config: Optional[HallucinationConfig] = None,
) -> HallucinationResult:
    """
    Détecte si la transcription est une hallucination.

    Args:
        reference: transcription de référence (peut être "")
        hypothesis: transcription générée
        config: seuils personnalisés

    Returns:
        HallucinationResult avec is_hallucination, reason, confidence, signals.
    """
    if config is None:
        config = HallucinationConfig()

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    # Ratio de longueur (éviter division par zéro)
    len_ratio = hyp_len / max(ref_len, 1)

    signals = {
        "ref_len": ref_len,
        "hyp_len": hyp_len,
        "len_ratio": round(len_ratio, 3),
        "token_rep": round(_max_token_repetition_ratio(hyp_tokens), 3),
        "bigram_rep": round(_n_gram_repetition_ratio(hyp_tokens, 2), 3),
        "trigram_rep": round(_n_gram_repetition_ratio(hyp_tokens, 3), 3),
    }

    # Si la référence est trop courte, on ne juge pas
    if ref_len < config.min_ref_len:
        return HallucinationResult(
            is_hallucination=False,
            reason=None,
            confidence=0.0,
            signals=signals,
            ref_len=ref_len,
            hyp_len=hyp_len,
            len_ratio=len_ratio,
        )

    # Ignorer les hypothèses très courtes
    if hyp_len < config.min_hyp_tokens:
        return HallucinationResult(
            is_hallucination=False,
            reason=None,
            confidence=0.0,
            signals=signals,
            ref_len=ref_len,
            hyp_len=hyp_len,
            len_ratio=len_ratio,
        )

    # --- Signaux de détection ---
    reasons = []
    confidence_scores = []

    # 1. Longueur anormale
    if len_ratio > config.max_len_ratio:
        reasons.append(f"hypothèse trop longue (ratio {len_ratio:.1f}×)")
        confidence_scores.append(min(1.0, (len_ratio - config.max_len_ratio) / 2.0 + 0.5))
    elif len_ratio < config.min_len_ratio:
        reasons.append(f"hypothèse trop courte (ratio {len_ratio:.2f}×)")
        confidence_scores.append(0.5)

    # 2. Répétition de token unique
    if signals["token_rep"] > config.max_token_rep:
        reasons.append(f"token répété ({signals['token_rep']:.2f})")
        confidence_scores.append(signals["token_rep"])

    # 3. Répétition de bigramme
    if signals["bigram_rep"] > config.max_bigram_rep:
        reasons.append(f"bigramme répété ({signals['bigram_rep']:.2f})")
        confidence_scores.append(signals["bigram_rep"])

    # 4. Répétition de trigramme
    if signals["trigram_rep"] > config.max_trigram_rep:
        reasons.append(f"trigramme répété ({signals['trigram_rep']:.2f})")
        confidence_scores.append(signals["trigram_rep"])

    # 5. Cas combiné : longueur + répétition
    if len_ratio > 1.5 and signals["bigram_rep"] > 0.2:
        reasons.append("longueur + répétition combinés")
        confidence_scores.append(0.9)

    # 6. Phrase anormalement longue avec token répété
    if hyp_len > 20 and signals["token_rep"] > 0.15:
        reasons.append(f"phrase longue + token répété ({signals['token_rep']:.2f})")
        confidence_scores.append(0.7)

    # 7. Longueur absolue anormale
    if hyp_len > 50:
        reasons.append(f"hypothèse très longue ({hyp_len} tokens)")
        confidence_scores.append(0.8)

    # Signal 9 : Troncature
    if hyp_len < 5 and ref_len > 10:
        reasons.append(f"troncature massive (hyp={hyp_len}, ref={ref_len})")
        confidence_scores.append(0.9)

    # Signal 10 : Troncature modérée
    if len_ratio < 0.3 and ref_len > 15:
        reasons.append(f"hypothèse trop courte (ratio {len_ratio:.2f})")
        confidence_scores.append(0.7)

    is_hallucination = len(reasons) > 0
    confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

    return HallucinationResult(
        is_hallucination=is_hallucination,
        reason="; ".join(reasons) if reasons else None,
        confidence=round(confidence, 3),
        signals=signals,
        ref_len=ref_len,
        hyp_len=hyp_len,
        len_ratio=len_ratio,
    )


def detect_hallucination_batch(
    references: list[str],
    hypotheses: list[str],
    config: Optional[HallucinationConfig] = None,
) -> list[HallucinationResult]:
    """Version batch (utile pour l'évaluation)."""
    return [
        detect_hallucination(ref, hyp, config)
        for ref, hyp in zip(references, hypotheses)
    ]


# ---------------------------------------------------------------------------
# CLI de test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tests unitaires simples
    test_cases = [

        {
            "ref": "en 1990 il a été ajouté à la liste des sites du patrimoine mondial en danger en raison de la menace des sables du désert",
            "hyp": "Au lieu de se faire tomber, on a eu quitter l'intérieur de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la",
            "expected": True,
            "name": "fr_064 (hallucination massive)",
        },
        {
            "ref": "aux femmes il est recommandé à toute femme qui voyage de dire qu'elle est mariée quel que soit son état civil",
            "hyp": "Il est recommandé à toute femme qui voyage de 10 à 20 milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de milliers de mill",
            "expected": True,
            "name": "fr_031 (milliers répétés)",
        },

        
        {
            "ref": "the quick brown fox jumps over the lazy dog",
            "hyp": "The quick brown fox jumps over the lazy dog.",
            "expected": False,
            "name": "Phrase parfaite",
        },
        {
            "ref": "تشتت في المحيط الأطلسي اليوم عاشر عاصفة",
            "hyp": "تشتت تشتت تشتت تشتت تشتت تشتت تشتت",
            "expected": True,
            "name": "ar_000 (répétition)",
        },
       
    ]

    print("=" * 70)
    print("TESTS DU DÉTECTEUR D'HALLUCINATIONS")
    print("=" * 70)

    passed = 0
    for tc in test_cases:
        result = detect_hallucination(tc["ref"], tc["hyp"])
        status = "✅" if result.is_hallucination == tc["expected"] else "❌"
        if result.is_hallucination == tc["expected"]:
            passed += 1
        print(f"\n{status} {tc['name']}")
        print(f"   is_hallucination : {result.is_hallucination} (attendu {tc['expected']})")
        print(f"   reason           : {result.reason}")
        print(f"   confidence       : {result.confidence}")
        print(f"   len_ratio        : {result.len_ratio:.2f}")
        print(f"   signals          : {result.signals}")

    print(f"\n{'='*70}")
    print(f"RÉSULTAT : {passed}/{len(test_cases)} tests passés")
    print(f"{'='*70}")
    