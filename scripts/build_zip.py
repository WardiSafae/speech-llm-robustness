"""
Crée le zip de fine-tuning contenant manifests + WAV.

Usage :
    python scripts/build_zip.py
"""

import pandas as pd
import zipfile
from pathlib import Path


def main():
    # 1. Charger les manifests
    print("📂 Chargement des manifests...")
    train_df = pd.read_csv("data/finetuning/train_manifest.tsv", sep="\t")
    val_df = pd.read_csv("data/finetuning/val_manifest.tsv", sep="\t")
    all_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"   Train : {len(train_df)}")
    print(f"   Val   : {len(val_df)}")
    print(f"   Total : {len(all_df)}\n")

    # 2. Créer le zip
    zip_path = Path("finetuning_data.zip")
    print(f"📦 Création de {zip_path}...")

    added = set()
    missing = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Ajouter les manifests
        zf.write("data/finetuning/train_manifest.tsv", "finetuning/train_manifest.tsv")
        zf.write("data/finetuning/val_manifest.tsv", "finetuning/val_manifest.tsv")
        print("   ✅ 2 manifests ajoutés")

        # Ajouter les WAV
        for p in all_df["audio_path"]:
            # Le manifest contient "finetuning/audio/NOM.wav"
            name = p.split("/")[-1]

            if name in added:
                continue

            # Chercher le fichier dans plusieurs endroits
            candidates = [
                Path("data/processed/fleurs_perturbed") / name,
                Path("data/finetuning/augmented") / name,
            ]

            found = False
            for c in candidates:
                if c.exists():
                    zf.write(c, f"finetuning/audio/{name}")
                    added.add(name)
                    found = True
                    break

            if not found:
                # Chercher dans tout data/
                matches = list(Path("data").rglob(name))
                if matches:
                    zf.write(matches[0], f"finetuning/audio/{name}")
                    added.add(name)
                    found = True

            if not found:
                missing.append(name)

        print(f"   ✅ {len(added)} WAV ajoutés")

    # 3. Bilan
    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n📊 Bilan :")
    print(f"   Fichiers dans le zip : {len(added) + 2}")
    print(f"   Taille : {size_mb:.1f} Mo")
    print(f"   Chemin : {zip_path.resolve()}")

    if missing:
        print(f"\n⚠️  {len(missing)} WAV manquants :")
        for m in missing[:10]:
            print(f"   • {m}")
        if len(missing) > 10:
            print(f"   ... et {len(missing) - 10} autres")

    # 4. Vérification
    print(f"\n🔍 Vérification...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        n_wav = sum(1 for n in names if n.endswith(".wav"))
        print(f"   {len(names)} entrées")
        print(f"   {n_wav} WAV")
        print(f"   Exemples : {names[2:5]}")


if __name__ == "__main__":
    main()