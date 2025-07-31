#!/bin/bash

# --- Cloud Classifier Setup for macOS/Linux ---

set -e # Exit immediately if a command exits with a non-zero status.

# --- 1. Check for Python ---
if ! command -v python3 &> /dev/null
then
    echo "ERREUR: python3 n'a pas été trouvé. Veuillez installer Python 3.8+."
    exit 1
fi
echo "Python 3 trouvé : $(python3 --version)"

# --- 2. Créer l'environnement virtuel ---
if [ -d "venv" ]; then
    echo "L'environnement virtuel 'venv' existe déjà. Création ignorée."
else
    echo "Création de l'environnement virtuel dans 'venv'..."
    python3 -m venv venv
fi

# --- 3. Activer l'environnement et installer les dépendances ---
echo "Activation de l'environnement virtuel..."
source venv/bin/activate

echo "Mise à jour des outils de base..."
pip install --upgrade pip setuptools wheel

echo "Installation des dépendances Python depuis requirements.txt..."
pip install -r requirements.txt

echo "Installation du package local en mode éditable..."
pip install -e .

# --- 4. Installation du compilateur Protobuf (protoc) ---
echo ""
echo "--- Installation du compilateur Protobuf (protoc) ---"
if command -v protoc &> /dev/null
then
    echo "'protoc' est déjà installé."
    protoc --version
else
    echo "'protoc' non trouvé. Tentative d'installation..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "Utilisation de apt-get (Debian/Ubuntu)..."
            sudo apt-get update && sudo apt-get install -y protobuf-compiler
        elif command -v dnf &> /dev/null; then
            echo "Utilisation de dnf (Fedora/CentOS)..."
            sudo dnf install -y protobuf-devel
        else
            echo "AVERTISSEMENT: Gestionnaire de paquets Linux non identifié. Veuillez installer 'protobuf-compiler' manuellement."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "Utilisation de Homebrew (macOS)..."
            brew install protobuf
        else
            echo "AVERTISSEMENT: Homebrew non trouvé. Veuillez installer 'protobuf' manuellement."
        fi
    else
        echo "AVERTISSEMENT: OS non supporté '$OSTYPE'. Veuillez installer 'protoc' manuellement."
    fi
fi

# --- 5. Configuration du noyau Jupyter ---
echo ""
echo "Installation du noyau Jupyter..."
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python (venv)"

pip install --upgrade jupyterlab

echo ""
echo "--- Installation terminée ! ---"
echo "Pour activer l'environnement dans un nouveau terminal, exécutez :"
echo "source venv/bin/activate"
echo ""
