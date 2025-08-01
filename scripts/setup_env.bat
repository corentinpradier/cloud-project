@echo off
setlocal

echo --- Installation de Cloud Classifier pour Windows ---

REM --- 1. Vérification de Python 3.11 ---
echo Verification de la presence de Python 3.11...
py -3.11 -c "import sys; sys.exit(0) if sys.version_info >= (3, 11) else sys.exit(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Python 3.11+ est requis. Veuillez l'installer et l'ajouter au PATH.
    pause
    exit /b 1
)

REM --- 2. Suppression et création de l'environnement virtuel ---
if exist venv (
    echo Suppression de l'ancien environnement virtuel 'venv'...
    rmdir /s /q venv
)
echo Creation de l'environnement virtuel dans 'venv'...
py -3.11 -m venv venv
if %errorlevel% neq 0 (
    echo ERREUR: La creation de l'environnement virtuel a echoue.
    pause
    exit /b 1
)

REM --- 3. Installation des dépendances Python ---
echo Installation des dependances Python depuis requirements.txt...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 exit /b 1
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b 1
pip install -e .
if %errorlevel% neq 0 exit /b 1

REM --- 4. Installation du compilateur Protobuf ---
echo.
echo --- Installation du compilateur Protobuf (protoc) ---
where protoc >nul 2>&1
if %errorlevel% equ 0 (
    echo 'protoc' est deja installe.
    protoc --version
) else (
    echo Installation de 'protoc' via Chocolatey...
    choco install protoc -y
    if %errorlevel% neq 0 (
        echo ERREUR: L'installation de 'protoc' a echoue. Veuillez l'installer manuellement.
        pause
        exit /b 1
    )
    echo Ajoutez le dossier 'bin' de protoc a votre PATH si necessaire.
)

REM --- 5. Configuration du noyau Jupyter ---
echo.
echo Installation du noyau Jupyter...
pip install ipykernel
if %errorlevel% neq 0 exit /b 1
python -m ipykernel install --user --name venv --display-name "Python (venv)"
if %errorlevel% neq 0 exit /b 1
pip install --upgrade jupyterlab
if %errorlevel% neq 0 exit /b 1

echo.
echo --- Installation terminee ! ---
echo Pour activer l'environnement dans un nouveau terminal, executez :
echo venv\Scripts\activate.bat
echo.
pause
endlocal
