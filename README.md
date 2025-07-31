# ☁️ Cloud Classifier ☁️

## About This Project

This project implements an end-to-end image classifier capable of recognizing and distinguishing different types of clouds. It leverages pre-trained deep learning models (via transfer learning) with TensorFlow and offers an interactive web interface built with Gradio to test predictions on new images.

A key feature is a Selenium-based script that automatically captures images from a live webcam feed, classifies them, and saves the results, allowing the dataset to be continuously expanded.

---

## Table des matières
* [Fonctionnalités](#fonctionnalités)
* [Structure du Projet](#structure-du-projet)
* [Installation](#installation)
* [Utilisation](#utilisation)
* [Dataset](#dataset)
* [Technologies Utilisées](#technologies-utilisées)

*(La suite du README est en français)*

## Fonctionnalités

*   **Entraînement de Modèles :** Utilisation de la classe `CloudsClassifier` pour construire, entraîner et affiner (fine-tune) des modèles basés sur des architectures reconnues comme `ResNet50V2`, `MobileNetV2`, et `VGG16`.
*   **Interface Web :** Une application Gradio (`scripts/app.py`) permet de téléverser une image et d'obtenir une prédiction instantanée du type de nuage.
*   **Extension du Dataset :** Un script (`src/cloudproject/expand_dataset.py`) utilise Selenium pour capturer des images depuis une webcam en direct, les classifie, et enregistre les résultats, permettant d'enrichir le dataset au fil du temps.
*   **Gestion de Modèles :** Sauvegarde et chargement faciles des modèles entraînés au format `.keras`.
*   **Visualisation :** Traçage des courbes d'apprentissage (perte et précision) avec Matplotlib.

## Structure du Projet

```
.
├── data/
│   ├── dataset/          # Datasets d'entraînement et de validation
│   │   ├── train/
│   │   └── valid/
│   └── scraped/          # Images et prédictions issues du scraping
├── models/               # Modèles entraînés sauvegardés (ex: ResNet50V2.keras)
├── scripts/
│   ├── app.py            # Script de lancement de l'application Gradio
│   └── setup_env.sh      # Script d'installation de l'environnement
├── src/
│   └── cloudproject/
│       ├── __init__.py
│       ├── classifier.py     # Classe principale du classifieur
│       └── expand_dataset.py # Script de scraping pour étendre le dataset
├── .gitignore
├── README.md             # Ce fichier
├── requirements.txt      # Dépendances Python
└── setup.py              # Fichier de configuration du package local (supposé)
```

## Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/corentinpradier/cloud-project.git
    cd Clouds_classification
    ```

2.  **Exécuter le script d'installation :**
    Ce script va créer un environnement virtuel, installer les dépendances Python et configurer le projet.
    ```bash
    bash scripts/setup_env.sh
    ```
    *Note : Le script utilise `brew` pour installer `protobuf` sur macOS. Si vous utilisez un autre système d'exploitation, vous devrez l'installer manuellement avec le gestionnaire de paquets approprié.*

3.  **Activer l'environnement virtuel :**
    À chaque nouvelle session dans le terminal, n'oubliez pas d'activer l'environnement :
    ```bash
    source venv/bin/activate
    ```

## Utilisation

### Lancer l'application de prédiction

Pour démarrer l'interface web Gradio, exécutez :
```bash
python scripts/app.py
```
Ouvrez ensuite votre navigateur à l'adresse `http://127.0.0.1:7860`.

### Entraîner un nouveau modèle

L'entraînement se fait en utilisant la classe `CloudsClassifier` de `src/cloudproject/classifier.py`. Vous pouvez créer un script Python ou utiliser un notebook Jupyter pour orchestrer l'entraînement.

Voici un exemple de code pour entraîner un modèle `ResNet50V2` :
```python
from cloudproject import CloudsClassifier

# Initialiser le classifieur avec les chemins vers les données
classifier = CloudsClassifier(
    train_dir="data/dataset/train",
    valid_dir="data/dataset/valid",
    img_height=200,
    img_width=200,
    batch_size=32,
)

# Construire le modèle
classifier.build_model(base_model_name="ResNet50V2", learning_rate=0.001)

# Lancer l'entraînement et le fine-tuning
classifier.train(
    epochs=20,
    fine_tune_epochs=10,
    model_name="MonSuperModele" # Le modèle sera sauvegardé dans models/MonSuperModele.keras
)

# Afficher les courbes d'apprentissage
classifier.plot_history()
```

### Étendre le dataset via scraping

Pour lancer le script qui capture une image de la webcam, la classifie et enregistre le résultat :
```bash
python src/cloudproject/expand_dataset.py
```
*Note : Ce script nécessite un `chromedriver` compatible avec votre version de Google Chrome et accessible dans votre `PATH`.*

## Dataset

Le projet s'attend à ce que les données soient structurées avec un dossier `train` et un dossier `valid`, chacun contenant les images et un fichier `_classes.csv`. Ce CSV doit mapper les noms de fichiers aux classes en utilisant un encodage one-hot.

Exemple de `_classes.csv`:
```csv
filename,Classe1,Classe2,Classe3
image_01.jpg,1,0,0
image_02.jpg,0,1,0
```

## Technologies Utilisées

*   **Langage :** Python 3.11
*   **Deep Learning :** TensorFlow / Keras
*   **Interface Web :** Gradio
*   **Manipulation de données :** Pandas, NumPy
*   **Web Scraping :** Selenium
*   **Visualisation :** Matplotlib