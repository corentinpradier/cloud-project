import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from cloudproject import CloudsClassifier

train_dir = "data/dataset/train"
valid_dir = "data/dataset/valid"
img_height = 200
img_width = 200
batch_size = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model_name = "ResNet50V2"
model_path = PROJECT_ROOT / (model_name + ".keras")

classifier = CloudsClassifier(
    # Il est plus sûr de convertir les objets Path en chaînes de caractères pour les bibliothèques qui ne les supportent pas
    train_dir=str(train_dir),
    valid_dir=str(valid_dir),
    img_height=img_height,
    img_width=img_width,
    batch_size=batch_size,
)

classifier.load_model(model_path)


options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Accéder à l'URL de la webcam
url = "https://g0.ipcamlive.com/player/player.php?alias=613202904e8bf&autoplay=1"
driver.get(url)

scraped_dir = PROJECT_ROOT / "data/scraped"
scraped_dir.mkdir(parents=True, exist_ok=True)  # Crée le dossier s'il n'existe pas
csv_file = scraped_dir / "predictions.csv"

try:
    # Attendre que l'élément vidéo soit disponible et cliquer dessus
    video_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "video"))
    )
    # video_element.click()
    driver.execute_script("arguments[0].click();", video_element)

    # Attendre un peu pour que la vidéo commence à jouer
    time.sleep(5)

    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))

    # Recadrer l'image (par exemple, la moitié gauche de l'image)
    width, height = image.size
    top = 0
    left = width - height
    right = width
    bottom = height
    cropped_image = image.crop((left, top, right, bottom))

    time_stamp = datetime.now().strftime("%Y%m%d%H%M")
    image_filename = scraped_dir / f"nuage_{time_stamp}.png"

    cropped_image.save(image_filename)
    print(f"Image recadrée sauvegardée avec succès : {image_filename}")

    # Prédiction
    predicted_class_name, confidence = classifier.predict(
        str(image_filename), show_image=False
    )

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["image_name", "prediction"])

    new_row = pd.DataFrame(
        [[str(image_filename), predicted_class_name]],
        columns=["image_name", "prediction"],
    )
    df = pd.concat([df, new_row], ignore_index=True)

    # Sauvegarder le DataFrame dans le fichier CSV
    df.to_csv(csv_file, index=False)

    time.sleep(10)

except Exception as e:
    print(f"Erreur lors de la capture ou du recadrage du screenshot: {e}")

finally:
    # Fermer le navigateur
    driver.quit()
