import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from cloudproject import CloudsClassifier, search_image_urls, url_to_image

# --- Constantes ---
# Liste des requêtes de recherche pour trouver des images de nuages.
SEARCH_QUERIES = [
    "altocumulus clouds",
    # "cirrus clouds",
    # "stratocumulus clouds",
    # "cumulus clouds sunny day",
]
MAX_IMAGES_PER_QUERY = 5  # Nombre max d'images à traiter par requête

# Définir un chemin de base pour rendre le script plus portable
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "dataset/train"
VALID_DIR = DATA_DIR / "dataset/valid"
SCRAPED_DIR = DATA_DIR / "scraped"
PREDICTIONS_CSV = SCRAPED_DIR / "predictions.csv"

# Configuration du modèle
IMG_HEIGHT = 200
IMG_WIDTH = 200
BATCH_SIZE = 32
MODEL_NAME = "ResNet50V2"
MODEL_PATH = BASE_DIR / "models" / f"{MODEL_NAME}.keras"


def initialize_classifier() -> CloudsClassifier:
    """Initialise et charge le modèle du classifieur de nuages."""
    print("Initialisation du classifieur...")
    classifier = CloudsClassifier(
        train_dir=str(TRAIN_DIR),
        valid_dir=str(VALID_DIR),
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE,
    )
    print(f"Chargement du modèle : {MODEL_PATH}")
    classifier.load_model(MODEL_PATH)
    return classifier


# def save_scraped_image(image: Image.Image) -> Path:
#     """Sauvegarde l'image et retourne son chemin."""
#     # S'assurer que le dossier de destination existe
#     SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

#     # Sauvegarder l'image avec un horodatage unique
#     time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     image_path = SCRAPED_DIR / f"nuage_{time_stamp}.png"
#     # On sauvegarde l'image telle quelle, sans recadrage.
#     # Le modèle s'occupera de la redimensionner correctement.
#     image.save(image_path)
#     print(f"Image sauvegardée avec succès : {image_path}")

#     return image_path


def log_prediction(filename: str, prediction: str):
    """Enregistre le résultat de la prédiction dans un fichier CSV."""
    new_row = {"image_name": filename, "prediction": prediction}

    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([new_row])

    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Prédiction '{prediction}' enregistrée dans {PREDICTIONS_CSV}")


def main():
    """
    Logique principale pour rechercher, télécharger, classifier et enregistrer
    des images de nuages depuis un moteur de recherche.
    """
    classifier = initialize_classifier()

    for query in SEARCH_QUERIES:
        print("-" * 50)
        print(f"Lancement de la recherche d'images pour la requête : '{query}'")

        # 1. Fonction de récupération qui cherche et télécharge les images
        urls = search_image_urls(query, max_images=MAX_IMAGES_PER_QUERY)

        images, filenames = url_to_image(urls, save=True, save_path=str(SCRAPED_DIR))

        if not images:
            print(f"Aucune image n'a pu être téléchargée pour '{query}'.")
            continue

        print(f"\n--- Traitement des {len(images)} images téléchargées pour '{query}' ---")
        # 2. Boucle qui traite chaque image, la classe et enregistre le résultat
        for i, image in enumerate(images):
            try:
                print(f"\nTraitement de l'image {i+1}/{len(images)}...")

                print("Prédiction du type de nuage...")
                predicted_class, confidence = classifier.predict(SCRAPED_DIR / filenames[i], show_image=False)
                print(f"-> Prédiction : {predicted_class} (Confiance : {confidence:.2f}%)")
                log_prediction(filenames[i], predicted_class)
            except Exception as e:
                print(f"Erreur lors du traitement d'une image pour la requête '{query}': {e}")
        time.sleep(5)  # Pause plus longue entre les requêtes pour éviter de surcharger

    print("-" * 50)
    print("Script terminé.")


if __name__ == "__main__":
    main()