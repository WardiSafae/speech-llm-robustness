"""
Analyse d'erreurs fine pour les Speech-LLMs :
    - WER/CER par langue
    - Top-N mots mal transcrits
    - Confusions phonétiques (basique)
    - Détection d'hallucinations (répétitions, ratio tokens/sec)
    - Rapport JSON + Markdown

Usage :
    python -m src.evaluation.error_analysis \\
        --results results/tables/whisper_results.csv \\
        --output results/error_analysis
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

from src.utils.io import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Minuscules, strip, espaces multiples."""
    return " ".join(str(text).strip().lower().split())


def _tokenize(text: str) -> list[str]:
    """Tokenization simple : split sur espaces + ponctuation."""
    text = _normalize(text)
    # Garde les lettres (latin + arabe) et chiffres
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text)
    return tokens


def _char_distance(a: str, b: str) -> int:
    """Distance de Levenshtein entre deux chaînes."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current = [i + 1]
        for j, cb in enumerate(b):
            insert = previous[j + 1] + 1
            delete = current[j] + 1
            substitute = previous[j] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _cer_word(ref: str, hyp: str) -> float:
    """CER au niveau mot (0 = identique)."""
    if not ref:
        return 1.0 if hyp else 0.0
    return _char_distance(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def per_language_report(df: pd.DataFrame) -> dict:
    """WER/CER par langue."""
    report = {}
    for lang, group in df.groupby("language"):
        refs = group["reference"].tolist()
        hyps = group["hypothesis"].tolist()
        report[lang] = {
            "n_samples": len(group),
            "wer": round(jiwer.wer(refs, hyps), 4),
            "cer": round(jiwer.cer(refs, hyps), 4),
        }
    return report


def per_sample_report(df: pd.DataFrame) -> list[dict]:
    """WER/CER par échantillon, avec diff mot-à-mot."""
    rows = []
    for _, row in df.iterrows():
        ref = _normalize(row["reference"])
        hyp = _normalize(row["hypothesis"])
        rows.append({
            "audio_path": row["audio_path"],
            "language": row.get("language", "?"),
            "reference": ref,
            "hypothesis": hyp,
            "wer": round(jiwer.wer(ref, hyp), 4) if ref else 1.0,
            "cer": round(jiwer.cer(ref, hyp), 4) if ref else 1.0,
        })
    return rows


def top_misrecognized_words(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Top-N mots de référence les plus mal transcrits (par CER moyen)."""
    stats: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        ref_tokens = _tokenize(row["reference"])
        hyp_tokens = _tokenize(row["hypothesis"])

        # Alignement simple : on apparie token i avec token i
        for i, ref_tok in enumerate(ref_tokens):
            if i < len(hyp_tokens):
                hyp_tok = hyp_tokens[i]
            else:
                hyp_tok = ""
            cer = _cer_word(ref_tok, hyp_tok)
            stats.setdefault(ref_tok, []).append(cer)

    # Moyenne et tri
    results = []
    for word, cers in stats.items():
        results.append({
            "word": word,
            "n_occurrences": len(cers),
            "mean_cer": round(sum(cers) / len(cers), 4),
            "max_cer": round(max(cers), 4),
        })

    results.sort(key=lambda x: (-x["mean_cer"], -x["n_occurrences"]))
    return results[:n]


def detect_hallucinations(df: pd.DataFrame,
                          max_token_repetition: float = 0.5,
                          max_len_ratio: float = 3.0) -> list[dict]:
    """
    Détecte les hallucinations :
        - répétition excessive de tokens (ratio > max_token_repetition)
        - longueur d'hypothèse très supérieure à la référence
    """
    hallucinations = []

    for _, row in df.iterrows():
        ref_tokens = _tokenize(row["reference"])
        hyp_tokens = _tokenize(row["hypothesis"])

        if not hyp_tokens:
            continue

        # Ratio de répétition
        counter = Counter(hyp_tokens)
        most_common_count = counter.most_common(1)[0][1]
        repetition_ratio = most_common_count / len(hyp_tokens)

        # Ratio de longueur
        len_ratio = len(hyp_tokens) / max(len(ref_tokens), 1)

        reason = None
        if repetition_ratio > max_token_repetition and len(hyp_tokens) > 4:
            reason = f"répétition excessive ({repetition_ratio:.2f})"
        elif len_ratio > max_len_ratio:
            reason = f"hypothèse trop longue (ratio {len_ratio:.1f}×)"

        if reason:
            hallucinations.append({
                "audio_path": row["audio_path"],
                "language": row.get("language", "?"),
                "hypothesis": row["hypothesis"],
                "reason": reason,
            })

    return hallucinations


def phoneme_confusion_analysis(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """
    Analyse basique des confusions de caractères (proxy pour les phonèmes).
    Compte les substitutions caractère → caractère via Levenshtein alignment.
    """
    confusion: Counter = Counter()

    for _, row in df.iterrows():
        ref = _normalize(row["reference"])
        hyp = _normalize(row["hypothesis"])

        # Alignement simple via jiwer
        try:
            alignment = jiwer.process_characters(ref, hyp)
            for chunk in alignment.alignments[0]:
                if chunk.type == "substitute":
                    ref_chars = ref[chunk.ref_start_idx:chunk.ref_end_idx]
                    hyp_chars = hyp[chunk.hyp_start_idx:chunk.hyp_end_idx]
                    if len(ref_chars) == 1 and len(hyp_chars) == 1:
                        confusion[(ref_chars, hyp_chars)] += 1
        except Exception:
            continue

    results = [
        {"ref_char": ref, "hyp_char": hyp, "count": count}
        for (ref, hyp), count in confusion.most_common(n)
    ]
    return results


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame) -> dict:
    """Construit le rapport complet d'analyse d'erreurs."""
    return {
        "summary": {
            "n_samples": len(df),
            "wer_global": round(jiwer.wer(df["reference"].tolist(), df["hypothesis"].tolist()), 4),
            "cer_global": round(jiwer.cer(df["reference"].tolist(), df["hypothesis"].tolist()), 4),
        },
        "per_language": per_language_report(df),
        "per_sample": per_sample_report(df),
        "top_misrecognized_words": top_misrecognized_words(df, n=10),
        "hallucinations": detect_hallucinations(df),
        "phoneme_confusions": phoneme_confusion_analysis(df, n=10),
    }


def save_report(report: dict, output_dir: Path) -> None:
    """Sauvegarde le rapport en JSON + Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "error_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Rapport JSON : {json_path}")

    # Markdown
    md_path = output_dir / "error_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Analyse d'erreurs\n\n")

        # Summary
        s = report["summary"]
        f.write("## Résumé global\n\n")
        f.write(f"- Échantillons : **{s['n_samples']}**\n")
        f.write(f"- WER global : **{s['wer_global']}**\n")
        f.write(f"- CER global : **{s['cer_global']}**\n\n")

        # Par langue
        f.write("## Résultats par langue\n\n")
        f.write("| Langue | N | WER | CER |\n")
        f.write("|--------|---|-----|-----|\n")
        for lang, vals in report["per_language"].items():
            f.write(f"| {lang} | {vals['n_samples']} | {vals['wer']} | {vals['cer']} |\n")
        f.write("\n")

        # Top mots mal transcrits
        f.write("## Top 10 mots mal transcrits\n\n")
        f.write("| Mot | Occurrences | CER moyen | CER max |\n")
        f.write("|-----|-------------|-----------|---------|\n")
        for w in report["top_misrecognized_words"]:
            f.write(f"| `{w['word']}` | {w['n_occurrences']} | {w['mean_cer']} | {w['max_cer']} |\n")
        f.write("\n")

        # Hallucinations
        f.write("## Hallucinations détectées\n\n")
        if report["hallucinations"]:
            for h in report["hallucinations"]:
                f.write(f"- `{Path(h['audio_path']).name}` ({h['language']}) : {h['reason']}\n")
        else:
            f.write("_Aucune hallucination détectée._\n")
        f.write("\n")

        # Confusions de caractères
        f.write("## Top 10 confusions de caractères\n\n")
        f.write("| Réf | Hyp | Occurrences |\n")
        f.write("|-----|-----|-------------|\n")
        for c in report["phoneme_confusions"]:
            f.write(f"| `{c['ref_char']}` | `{c['hyp_char']}` | {c['count']} |\n")
        f.write("\n")

    logger.info(f"Rapport Markdown : {md_path}")


def print_report(report: dict) -> None:
    """Affichage console synthétique."""
    print("\n" + "=" * 60)
    print("📊 ANALYSE D'ERREURS")
    print("=" * 60)

    s = report["summary"]
    print(f"\nRésumé : WER={s['wer_global']} | CER={s['cer_global']} | N={s['n_samples']}")

    print("\n=== Par langue ===")
    for lang, vals in sorted(report["per_language"].items(),
                             key=lambda x: x[1]["wer"], reverse=True):
        bar = "█" * int(vals["wer"] * 40)
        print(f"  {lang} : WER={vals['wer']:.3f} | CER={vals['cer']:.3f} | N={vals['n_samples']:2d}  {bar}")

    print("\n=== Top 5 mots mal transcrits ===")
    for w in report["top_misrecognized_words"][:5]:
        print(f"  `{w['word']}` : CER moyen={w['mean_cer']} (n={w['n_occurrences']})")

    print("\n=== Hallucinations ===")
    if report["hallucinations"]:
        for h in report["hallucinations"]:
            print(f"  ⚠️  {Path(h['audio_path']).name} ({h['language']}) : {h['reason']}")
    else:
        print("  ✅ Aucune")

    print("\n=== Top 5 confusions de caractères ===")
    for c in report["phoneme_confusions"][:5]:
        print(f"  `{c['ref_char']}` → `{c['hyp_char']}` ({c['count']}×)")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse d'erreurs des Speech-LLMs.")
    parser.add_argument("--results", required=True,
                        help="CSV généré par metrics.py (ex: results/tables/whisper_results.csv)")
    parser.add_argument("--output", required=True,
                        help="Dossier de sortie pour le rapport")
    parser.add_argument("--language-column", default="language",
                        help="Nom de la colonne langue (défaut: language)")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)

    if not results_path.exists():
        logger.error(f"Fichier introuvable : {results_path}")
        return

    df = pd.read_csv(results_path)
    logger.info(f"Chargement de {len(df)} résultats depuis {results_path}")

    # Si pas de colonne langue, l'extraire du path (synth_en_00.wav → en)
    if args.language_column not in df.columns:
        logger.warning(f"Colonne '{args.language_column}' absente, extraction depuis le nom de fichier.")
        df["language"] = df["audio_path"].apply(
            lambda p: Path(p).stem.split("_")[1] if "_" in Path(p).stem else "?"
        )

    report = build_report(df)
    print_report(report)
    save_report(report, output_dir)


if __name__ == "__main__":
    main()