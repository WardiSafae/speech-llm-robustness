"""
Smoke test : valide l'ensemble du pipeline sans réseau ni modèles lourds.

Vérifie :
    1. Tous les modules s'importent
    2. WER/CER fonctionnent
    3. Augmentation fonctionne sur un WAV factice
    4. Manifest TSV est lisible
    5. Le registry de modèles contient les 3 entrées

Usage :
    python scripts/smoke_test.py
"""
import io
import sys
import tempfile
from pathlib import Path

# Ajouter la racine au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf


RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, fn):
    """Exécute fn() et enregistre le résultat."""
    try:
        fn()
        RESULTS.append((label, True, ""))
        print(f"✅ {label}")
    except Exception as e:
        RESULTS.append((label, False, f"{type(e).__name__}: {e}"))
        print(f"❌ {label} → {type(e).__name__}: {e}")


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST — speech-llm-robustness")
    print("=" * 60)

    # --- 1. Imports ---
    print("\n[1] Imports des modules")
    check("import src.utils.io", lambda: __import__("src.utils.io"))
    check("import src.models.model_wrappers", lambda: __import__("src.models.model_wrappers"))
    check("import src.evaluation.metrics", lambda: __import__("src.evaluation.metrics"))
    check("import src.augmentation.augment", lambda: __import__("src.augmentation.augment"))
    check("import src.data.load_datasets", lambda: __import__("src.data.load_datasets"))
    check("import src.finetuning.finetune", lambda: __import__("src.finetuning.finetune"))

    # --- 2. Métriques ---
    print("\n[2] Métriques WER/CER")
    from src.evaluation.metrics import compute_wer, compute_cer

    def _check_wer_identity():
        assert compute_wer(["hello"], ["hello"]) == 0.0

    def _check_cer_identity():
        assert compute_cer(["hello"], ["hello"]) == 0.0

    def _check_wer_error():
        assert compute_wer(["hello world"], ["hello"]) > 0.0

    check("WER(identité) == 0", _check_wer_identity)
    check("CER(identité) == 0", _check_cer_identity)
    check("WER(erreur) > 0", _check_wer_error)

    # --- 3. Augmentation ---
    print("\n[3] Augmentation audio")
    from src.augmentation.augment import apply_perturbation, SUPPORTED_PERTURBATIONS

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wav_in = tmp_path / "test.wav"
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        sf.write(str(wav_in), 0.3 * np.sin(2 * np.pi * 440 * t), sr)

        def _check_perturbations():
            for p in SUPPORTED_PERTURBATIONS:
                out = apply_perturbation(wav_in, p, tmp_path)
                assert out.exists(), f"{p} non généré"
                assert out.stat().st_size > 0, f"{p} vide"

        check(f"5 perturbations ({SUPPORTED_PERTURBATIONS})", _check_perturbations)

    # --- 4. Manifest ---
    print("\n[4] Lecture du manifest")
    from src.evaluation.metrics import load_references

    manifest = ROOT / "data" / "processed" / "manifest.tsv"

    def _check_manifest():
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} absent (normal si vide)")
        refs = load_references(manifest)
        assert len(refs) > 0, "Manifest vide"

    check(f"load_references({manifest.name})", _check_manifest)

    # --- 5. Registry ---
    print("\n[5] Registry de modèles")
    from src.models.model_wrappers import MODEL_REGISTRY

    def _check_registry():
        expected = {"whisper", "wav2vec2", "qwen2_audio"}
        assert set(MODEL_REGISTRY.keys()) == expected, f"Registry: {list(MODEL_REGISTRY)}"

    check(f"registry = {list(MODEL_REGISTRY)}", _check_registry)

    # --- Bilan ---
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"RÉSULTAT : {passed}/{total} tests passés")
    if passed < total:
        print("\nÉchecs :")
        for label, ok, err in RESULTS:
            if not ok:
                print(f"  ❌ {label} → {err}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())