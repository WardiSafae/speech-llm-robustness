"""Supprime les fichiers WAV corrompus du dossier de perturbations."""
import soundfile as sf
from pathlib import Path
import sys

def main():
    target = Path("data/processed/fleurs_perturbed")
    if not target.exists():
        print(f"❌ Dossier introuvable : {target}")
        sys.exit(1)

    all_wavs = list(target.glob("*.wav"))
    print(f"📁 {len(all_wavs)} fichiers à vérifier\n")

    corrupted = []
    for i, wav in enumerate(all_wavs):
        try:
            info = sf.info(wav)
            # Corrompu si 0 frames ou durée nulle
            if info.frames == 0 or info.duration < 0.1:
                corrupted.append(wav)
        except Exception:
            corrupted.append(wav)

        if (i + 1) % 500 == 0:
            print(f"  Vérifiés : {i+1}/{len(all_wavs)} | corrompus : {len(corrupted)}")

    print(f"\n🗑️  {len(corrupted)} fichiers corrompus identifiés")

    # Supprimer
    for wav in corrupted:
        try:
            wav.unlink()
        except Exception as e:
            print(f"  ✗ Impossible de supprimer {wav.name} : {e}")

    print(f"✅ {len(corrupted)} fichiers supprimés")

    # Bilan
    remaining = list(target.glob("*.wav"))
    print(f"\n📊 Bilan :")
    print(f"  Avant : {len(all_wavs)}")
    print(f"  Supprimés : {len(corrupted)}")
    print(f"  Restants : {len(remaining)}")


if __name__ == "__main__":
    main()