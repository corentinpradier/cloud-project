import gradio as gr
from clouds_classifier import CloudsClassifier

TRAIN_DIR = "data/dataset/train"
VALID_DIR = "data/dataset/valid"
IMG_HEIGHT = 200
IMG_WIDTH = 200
BATCH_SIZE = 32
MODEL_PATH = "ResNet50V2.keras"

print("Initialisation du classifieur...")
try:
    classifier = CloudsClassifier(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE,
    )
except FileNotFoundError:
    print(f"ERREUR: Impossible de trouver les datasets aux chemins spécifiés.")
    print(
        f"Vérifiez que les dossiers '{TRAIN_DIR}' et '{VALID_DIR}' existent et contiennent les fichiers _classes.csv."
    )
    exit()


print(f"Chargement du modèle depuis {MODEL_PATH}...")
try:
    classifier.load_model(MODEL_PATH)
except Exception as e:
    print(f"ERREUR: Impossible de charger le modèle depuis 'models/{MODEL_PATH}'.")
    print(f"Assurez-vous que le modèle a été entraîné et sauvegardé sous ce nom.")
    print(f"Erreur détaillée: {e}")
    exit()


def predict_image(image_array):
    """
    Fonction qui prend une image (fournie par Gradio) et renvoie la prédiction.
    Gradio fournit l'image comme un tableau NumPy (H, W, C).
    """
    if image_array is None:
        return "Veuillez fournir une image."

    predicted_class_name, confidence = classifier.predict_from_array(image_array)

    return f"Prédiction : {predicted_class_name}\nConfiance : {confidence:.2f}%"


print(
    "Lancement de l'interface Gradio... Ouvrez http://127.0.0.1:7860 dans votre navigateur."
)
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="numpy",
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        label="Faites glisser une image de nuage ici ou cliquez pour en choisir une",
    ),
    outputs=gr.Textbox(label="Résultat de la prédiction"),
    title="☁️ Classifieur de Nuages ☁️",
    description="Téléchargez une image d'un nuage et le modèle prédira de quel type il s'agit. Cette démo utilise un modèle pré-entraîné.",
    examples=[
        ["dataset/valid/Ac-8-_jpg.rf.382636684b876860c5a61aa85238104d.jpg"],
        ["dataset/valid/Sc-63-_jpg.rf.6b946a6cda82df3d5e9b3175181896f8.jpg"],
        ["dataset/valid/Cu-N133_jpg.rf.8b7d11e0036d5a3cf3bbe4084c2a1cf4.jpg"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch()
