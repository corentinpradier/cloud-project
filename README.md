# Cloud Classifier

## About This Project

This project implements an end-to-end image classifier capable of recognizing and distinguishing different types of clouds. It leverages pre-trained deep learning models (via transfer learning) with TensorFlow and offers an interactive web interface built with Gradio to test predictions on new images.

A key feature is a Selenium-based script that automatically captures images from a live webcam feed, classifies them, and saves the results, allowing the dataset to be continuously expanded.

---

## Table of Contents
* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)

## Features

*   **Model Training:** Use the `CloudsClassifier` class to build, train, and fine-tune models based on well-known architectures like `ResNet50V2`, `MobileNetV2`, and `VGG16`.
*   **Web Interface:** A Gradio application (`scripts/app.py`) allows you to upload an image and get an instant prediction of the cloud type.
*   **Dataset Expansion:** A script (`src/cloudproject/expand_dataset.py`) uses Selenium to capture images from a live webcam, classify them, and save the results, enabling the dataset to grow over time.
*   **Model Management:** Easily save and load trained models in `.keras` format.
*   **Visualization:** Plot learning curves (loss and accuracy) with Matplotlib.

## Project Structure

```
.
├── data/
│   ├── dataset/          # Training and validation datasets
│   │   ├── train/
│   │   └── valid/
│   └── scraped/          # Images and predictions from scraping
├── models/               # Saved trained models (e.g., ResNet50V2.keras)
├── scripts/
│   ├── app.py            # Gradio application launch script
│   └── setup_env.sh      # Environment setup script
├── src/
│   └── cloudproject/
│       ├── __init__.py
│       ├── classifier.py     # Main classifier class
│       └── expand_dataset.py # Scraping script to expand the dataset
├── .gitignore
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── setup.py              # Local package configuration file (optional)
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/corentinpradier/cloud-project.git
    ```

2.  **Run the setup script:**
    This script will create a virtual environment, install Python dependencies, and configure the project.
    ```bash
    bash scripts/setup_env.sh
    ```
    *Note: The script uses `brew` to install `protobuf` on macOS. If you use another operating system, you will need to install it manually with the appropriate package manager.*

3.  **Activate the virtual environment:**
    At the start of each new terminal session, don't forget to activate the environment:
    ```bash
    source venv/bin/activate
    ```

## Usage

### Launch the prediction application

To start the Gradio web interface, run:
```bash
python scripts/app.py
```
Then open your browser at `http://127.0.0.1:7860`.

### Train a new model

Training is done using the `CloudsClassifier` class from `src/cloudproject/classifier.py`. You can create a Python script or use a Jupyter notebook to orchestrate the training.

Here is an example code to train a `ResNet50V2` model:
```python
from cloudproject import CloudsClassifier

# Initialize the classifier with data paths
classifier = CloudsClassifier(
    train_dir="data/dataset/train",
    valid_dir="data/dataset/valid",
    img_height=200,
    img_width=200,
    batch_size=32,
)

# Build the model
classifier.build_model(base_model_name="ResNet50V2", learning_rate=0.001)

# Start training and fine-tuning
classifier.train(
    epochs=20,
    fine_tune_epochs=10,
    model_name="MySuperModel" # The model will be saved in models/MySuperModel.keras
)

# Display learning curves
classifier.plot_history()
```

### Expand the dataset via scraping

To run the script that captures an image from the webcam, classifies it, and saves the result:
```bash
python src/cloudproject/expand_dataset.py
```
*Note: This script requires a `chromedriver` compatible with your version of Google Chrome and accessible in your `PATH`.*

## Dataset

The project expects the data to be structured with a `train` folder and a `valid` folder, each containing images and a `_classes.csv` file. This CSV should map filenames to classes using one-hot encoding.

Example `_classes.csv`:
```csv
filename,Class1,Class2,Class3
image_01.jpg,1,0,0
image_02.jpg,0,1,0
```

## Technologies Used

*   **Language:** Python 3.11
*   **Deep Learning:** TensorFlow / Keras
*   **Web Interface:** Gradio
*   **Data Manipulation:** Pandas, NumPy
*   **Web Scraping:** Selenium
*   **Visualization:** Matplotlib