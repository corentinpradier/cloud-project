import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from cloudproject import CloudsClassifier, scraping

# --- Constantes ---
URL = "https://g0.ipcamlive.com/player/player.php?alias=613202904e8bf&autoplay=1"
URL = "http://meteosandillon.fr/photo.jpg"

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
MODEL_PATH = f"{MODEL_NAME}.keras"


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


def process_and_save_image(image: Image.Image) -> Path:
    """Recadre et sauvegarde l'image, puis retourne le chemin du fichier sauvegardé."""
    # S'assurer que le dossier de destination existe
    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

    # Définir la boîte de recadrage (un carré sur le côté droit)
    width, height = image.size
    crop_box = (width - height, 0, width, height)

    # Utiliser la fonction de recadrage générique
    cropped_image = image.crop(crop_box)

    # Sauvegarder l'image recadrée avec un horodatage
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = SCRAPED_DIR / f"nuage_{time_stamp}.png"
    cropped_image.save(image_path)
    print(f"Image recadrée sauvegardée avec succès : {image_path}")

    return image_path


def log_prediction(image_path: Path, prediction: str):
    """Enregistre le résultat de la prédiction dans un fichier CSV."""
    new_row = {"image_name": image_path.name, "prediction": prediction}

    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([new_row])

    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Prédiction '{prediction}' enregistrée dans {PREDICTIONS_CSV}")


def main():
    """Logique principale du script pour scraper, classifier et enregistrer une image de nuage."""
    classifier = initialize_classifier()

    print(f"Scraping de l'image depuis : {URL}")
    image = scraping(URL)

    if image:
        image_path = process_and_save_image(image)

        print("Prédiction du type de nuage...")
        predicted_class, confidence = classifier.predict(str(image_path), show_image=False)
        print(f"-> Prédiction : {predicted_class} (Confiance : {confidence:.2f}%)")

        log_prediction(image_path, predicted_class)
    else:
        print("Échec de la récupération de l'image.")

    print("Script terminé.")


if __name__ == "__main__":
    main()