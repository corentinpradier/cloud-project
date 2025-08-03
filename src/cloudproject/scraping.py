import io
import json
import os
import re
import time
from typing import Union
from urllib.parse import urlparse

import requests
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def search_image_urls(
    query: str, max_images: int = 10, verbose: bool = False
) -> list[str]:
    """
    Recherche des images sur Bing et retourne une liste d'URLs.
    S'arrête si le nombre d'images récupérées est le même 3 fois d'affilée.

    Args:
        query (str): Le terme de recherche (ex: "cumulus clouds").
        max_images (int): Le nombre maximum d'URLs d'images à retourner.

    Returns:
        list[str]: Une liste d'URLs d'images.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")

    driver = None
    image_urls = set()
    try:
        driver = webdriver.Chrome(options=options)
        search_url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}"
        if verbose:
            print(f"Navigation vers : {search_url}")
        driver.get(search_url)

        last_height = driver.execute_script("return document.body.scrollHeight")

        scroll_attempts_without_change = 0
        MAX_ATTEMPTS = 3

        same_count_attempts = 0
        previous_count = 0

        while len(image_urls) < max_images:
            thumbnail_elements = driver.find_elements(By.CLASS_NAME, "iusc")
            for thumb in thumbnail_elements:
                m_json = thumb.get_attribute("m")
                if m_json:
                    try:
                        m_data = json.loads(m_json)
                        image_urls.add(m_data["murl"])
                    except (json.JSONDecodeError, KeyError):
                        continue

            current_count = len(image_urls)
            if verbose:
                print(f"URLs uniques trouvées : {current_count}/{max_images}")

            # Vérifier si le nombre d'images a augmenté
            if current_count == previous_count:
                same_count_attempts += 1
                if verbose:
                    print(
                        f"Nombre d'images inchangé ({current_count}) - tentative {same_count_attempts}/3"
                    )
                if same_count_attempts >= 3:
                    if verbose:
                        print("Nombre d'images inchangé trois fois d'affilée. Arrêt.")
                    break
            else:
                same_count_attempts = 0  # reset si ça évolue
                previous_count = current_count

            if current_count >= max_images:
                if verbose:
                    print(f"Objectif de {max_images} URLs atteint.")
                break

            # Scroll vers le bas
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                scroll_attempts_without_change += 1
                if verbose:
                    print(
                        f"La hauteur de la page n'a pas changé (tentative {scroll_attempts_without_change}/{MAX_ATTEMPTS})."
                    )
                try:
                    see_more_button = driver.find_element(By.CLASS_NAME, "btn_seemore")
                    if verbose:
                        print("Bouton 'Voir plus' trouvé, clic...")
                    driver.execute_script("arguments[0].click();", see_more_button)
                    time.sleep(2)
                    scroll_attempts_without_change = 0
                except Exception:
                    if scroll_attempts_without_change >= MAX_ATTEMPTS:
                        if verbose:
                            print("Fin de page ou plus de contenu détecté.")
                        break
            else:
                scroll_attempts_without_change = 0
            last_height = new_height

    except Exception as e:
        if verbose:
            print(f"Une erreur est survenue lors du scraping : {e}")
    finally:
        if driver:
            driver.quit()

    print(f"Scraping terminé. {len(image_urls)} URLs uniques trouvées.")
    return list(image_urls)[:max_images]


def url_to_image(url: Union[str, list[str]], save: bool = False, save_path: str = "../../data/scraped/") -> list[Image.Image]:
    if isinstance(url, str):
        url = [url]

    images = []
    filenames = []
    for u in url:
        response = requests.get(u, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        image = Image.open(io.BytesIO(response.content))
        images.append(image)

        if save:
            filename = clean_image_filename(u)
            filenames.append(filename)
            image.save(f"{save_path}/{filename}")
            print(f"Image sauvegardée sous : {filename}")

    if save:
        return images, filenames
    else:
        return images

def clean_image_filename(url: str) -> str:
    """
    Extrait le nom de fichier propre à partir d'une URL d'image.
    Supprime les paramètres après l'extension (.jpg, .png, etc.).

    Args:
        url (str): L'URL de l'image.

    Returns:
        str: Le nom de fichier propre, ou None si aucun nom valide trouvé.
    """
    # Extraire le chemin brut de l'URL
    parsed = urlparse(url)
    path = parsed.path  # exemple : "/images/photo.jpg"

    # Extraire le dernier segment (le nom du fichier brut)
    filename = os.path.basename(path)

    # Nettoyer : supprimer tout après l'extension d'image
    match = re.match(r"^(.*?\.(jpg|jpeg|png|gif|bmp|webp))", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None