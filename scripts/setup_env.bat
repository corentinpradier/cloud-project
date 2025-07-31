@echo off
setlocal

echo --- Installation de Cloud Classifier pour Windows ---

REM --- 1. V‚rification de Python ---
echo Verification de la presence de Python 3...
py -3 -c "import sys; sys.exit(0) if sys.version_info >= (3, 8) else sys.exit(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Python 3.8+ est requis. Veuillez l'installer et l'ajouter au PATH.
    pause
    exit /b 1
)

REM --- 2. Cr‚ation de l'environnement virtuel ---
if exist venv (
    echo L'environnement virtuel 'venv' existe deja. Creation ignoree.
) else (
    echo Creation de l'environnement virtuel dans 'venv'...
    py -3 -m venv venv
    if %errorlevel% neq 0 (
        echo ERREUR: La creation de l'environnement virtuel a echoue.
        pause
        exit /b 1
    )
)

REM --- 3. Installation des d‚pendances Python ---
echo Installation des ependances Python depuis requirements.txt...
call venv\Scripts\activate.bat
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

REM --- 4. Installation du compilateur Protobuf ---
echo.
echo --- Installation du compilateur Protobuf (protoc) ---
where protoc >nul 2>&1
if %errorlevel% equ 0 (
    echo 'protoc' est deja installe.
    protoc --version
) else (
    echo AVERTISSEMENT: Compilateur 'protoc' non trouve.
    echo Veuillez l'installer manuellement.
    echo Vous pouvez utiliser un gestionnaire de paquets comme Chocolatey ('choco install protoc')
    echo ou le telecharger depuis les releases GitHub officielles de Protobuf.
    echo Assurez-vous d'ajouter le dossier 'bin' … votre PATH systŠme.
)

REM --- 5. Configuration du noyau Jupyter ---
echo.
echo Installation du noyau Jupyter...
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python (venv)"
pip install --upgrade jupyterlab

echo.
echo --- Installation terminee ! ---
echo Pour activer l'environnement dans un nouveau terminal, executez :
echo venv\Scripts\activate.bat
echo.
pause
endlocal
