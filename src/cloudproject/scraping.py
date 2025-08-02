import io
import time

import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def scraping(url: str):
    """
    Récupère une image depuis une URL.

    Tente d'abord de télécharger l'URL comme une image directe pour plus d'efficacité.
    Si cela échoue ou si l'URL pointe vers une page web, utilise Selenium
    pour prendre une capture d'écran de la page.

    Args:
        url (str): L'URL de l'image ou de la page web.

    Returns:
        Image.Image | None: Un objet Image de PIL si la capture réussit, sinon None.
    """
    # 1. Tentative de téléchargement direct si l'URL est une image
    try:
        headers = requests.head(url, timeout=5, allow_redirects=True).headers
        content_type = headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            print(f"URL détectée comme une image directe ({content_type}). Téléchargement...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Lève une exception si le statut est une erreur
            return Image.open(io.BytesIO(response.content))
        else:
            print("L'URL ne pointe pas vers une image directe. Passage à Selenium.")
    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête directe : {e}. Passage à Selenium.")

    # 2. Si ce n'est pas une image directe, utiliser Selenium
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        print(f"Chargement de l'URL avec Selenium : {url}")
        driver.get(url)
        # Tentative optionnelle d'interagir avec une vidéo si elle existe
        try:
            video_element = WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.TAG_NAME, "video")))
            print("Élément <video> trouvé. Tentative de lecture...")
            driver.execute_script("arguments[0].play();", video_element)
            time.sleep(5)  # Laisser le temps à la vidéo de démarrer
        except Exception:
            print("Aucun élément <video> trouvé. Prise d'une capture d'écran de la page.")
        screenshot = driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(screenshot))
    except Exception as e:
        print(f"Erreur lors de la capture avec Selenium : {e}")
        return None
    finally:
        if driver:
            driver.quit()