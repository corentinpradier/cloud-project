#!/bin/bash

# Aller à la racine du projet

# Créer l'environnement virtuel
python3.11 -m venv venv

# Activer l’environnement virtuel
source venv/bin/activate

# Mise à jour des outils de base
pip install --upgrade pip setuptools wheel

# Installation des dépendances
pip install -r requirements.txt

# Installation du package local
pip install -e .

# Installation de protoc (compilateur protobuf)
brew install protobuf

# Vérification protoc
protoc --version

# Installation pour utiliser dans Jupyter
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python (venv)"

# Installation de JupyterLab
pip install --upgrade jupyterlab
