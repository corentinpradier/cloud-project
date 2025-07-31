#!/bin/bash

# --- Cloud Classifier Setup for macOS/Linux with Python 3.11 forced ---

set -e # Exit immediately if a command exits with a non-zero status.

echo ""
echo "⚙️  --- Setup CloudProject avec Python 3.11 ---"
echo ""

# --- 1. Localiser Python 3.11 ---
PYTHON_BIN="/opt/homebrew/bin/python3.11"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ ERREUR : Python 3.11 non trouvé à $PYTHON_BIN"
    echo "➡️  Installez-le avec : brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python 3.11 détecté : version $PYTHON_VERSION"

# --- 2. Supprimer ancien environnement virtuel ---
rm -rf venv

# --- 3. Créer l'environnement virtuel avec Python 3.11 ---
echo "Création de l'environnement virtuel dans 'venv' avec Python 3.11..."
$PYTHON_BIN -m venv venv

# --- 4. Activer l'environnement virtuel ---
echo "Activation de l'environnement virtuel..."
source venv/bin/activate

# --- 5. Mettre à jour pip, setuptools, wheel ---
echo "Mise à jour des outils de base..."
pip install --upgrade pip setuptools wheel

# --- 6. Installer les dépendances depuis requirements.txt ---
echo "Installation des dépendances Python depuis requirements.txt..."
pip install -r requirements.txt

# --- 7. Installer le package local en mode éditable ---
echo "Installation du package local en mode éditable..."
pip install -e .

# --- 8. Installation du compilateur Protobuf (protoc) ---
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

# --- 9. Installation du noyau Jupyter ---
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
