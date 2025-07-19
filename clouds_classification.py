import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class CloudsClassifier:
    def __init__(
        self,
        train_dir,
        valid_dir,
        img_height,
        img_width,
        batch_size,
        autotune=tf.data.AUTOTUNE,
    ):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.train_csv_path = os.path.join(train_dir, "_classes.csv")
        self.valid_csv_path = os.path.join(valid_dir, "_classes.csv")
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.autotune = autotune
        self.model = None
        self.base_model = None

        print("Création du dataset d'entraînement...")
        self.train_data, self.class_names = self._create_dataset_from_csv(
            self.train_dir, self.train_csv_path, is_training=True
        )

        print("\nCréation du dataset de validation...")
        self.val_data, _ = self._create_dataset_from_csv(
            self.valid_dir, self.valid_csv_path, is_training=False
        )
        print("\nLes datasets sont prêts.")

    def _load_and_resize_image(self, path: str, label: int):
        """Charge une image depuis un chemin, la décode et la redimensionne."""
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        return img, label

    # --- 3. Fonction réutilisable pour créer un dataset depuis un dossier et un CSV ---
    def _create_dataset_from_csv(self, image_dir, csv_path, is_training=True):
        """
        Crée un tf.data.Dataset complet à partir d'un dossier d'images et d'un fichier CSV.

        Args:
            image_dir (str): Chemin vers le dossier contenant les images.
            csv_path (str): Chemin vers le fichier CSV contenant les noms et labels.
            is_training (bool): Si True, le dataset sera mélangé (shuffle).

        Returns:
            tf.data.Dataset: Le dataset final, prêt pour l'entraînement ou la validation.
            list: La liste des noms de classes.
        """
        print(f"Chargement des données depuis : {image_dir}")

        # Lire le fichier CSV
        df = pd.read_csv(csv_path)
        df.rename(columns={df.columns[0]: "filename"}, inplace=True)

        # Créer les chemins complets vers les images
        df["full_path"] = df["filename"].apply(
            lambda name: os.path.join(image_dir, name)
        )

        # Extraire les chemins et les labels
        image_paths = df["full_path"].values
        label_columns = df.drop(columns=["filename", "full_path"])
        labels = label_columns.values
        class_names = label_columns.columns.tolist()

        print(
            f"Chargement depuis '{image_dir}': {len(image_paths)} images trouvées pour les classes {class_names}."
        )

        # Créer le dataset de base (chemins, labels)
        path_label_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        # Appliquer la fonction de chargement et de prétraitement
        image_label_ds = path_label_ds.map(
            self._load_and_resize_image, num_parallel_calls=self.autotune
        )

        # Appliquer le brassage (shuffle) SEULEMENT pour l'ensemble d'entraînement
        if is_training:
            # La taille du buffer est souvent fixée à la taille totale du dataset
            # pour un brassage complet
            image_label_ds = image_label_ds.shuffle(buffer_size=len(image_paths))

        # Appliquer le batching et le prefetching
        final_dataset = image_label_ds.batch(self.batch_size).prefetch(
            buffer_size=self.autotune
        )

        return final_dataset, class_names

    def build_model(self, base_model_name: str, learning_rate=0.001):
        """
        Construit un modèle de classification en utilisant le transfert d'apprentissage.

        Args:
            base_model_name (str): Le nom du modèle pré-entraîné à utiliser (ex: "MobileNetV2").
            learning_rate (float): Le taux d'apprentissage pour l'optimiseur.
        """
        MODEL_REGISTRY = {
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "ResNet50V2": tf.keras.applications.ResNet50V2,
            "VGG16": tf.keras.applications.VGG16,
            # Ajoutez d'autres modèles ici
        }

        PREPROCESSORS = {
            tf.keras.applications.MobileNetV2: tf.keras.applications.mobilenet_v2.preprocess_input,
            tf.keras.applications.ResNet50V2: tf.keras.applications.resnet_v2.preprocess_input,
            tf.keras.applications.VGG16: tf.keras.applications.vgg16.preprocess_input,
            # Ajoutez d'autres modèles ici
        }

        base_model_class = MODEL_REGISTRY.get(base_model_name)
        if not base_model_class:
            raise ValueError(
                f"Le nom du modèle '{base_model_name}' n'est pas reconnu. "
                f"Modèles disponibles : {list(MODEL_REGISTRY.keys())}"
            )

        preprocess_input = PREPROCESSORS.get(base_model_class)

        self.base_model = base_model_class(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights="imagenet",
        )

        # 1. Freeze le modèle
        self.base_model.trainable = False

        # 3. Construire le nouveau modèle avec l'API Fonctionnelle
        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = preprocess_input(x)  # Utiliser le bon pré-processeur
        x = self.base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.class_names), activation="softmax")(x)

        self.model = tf.keras.Model(inputs, outputs)

        # 4. Compiler le modèle
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        print(f"Modèle construit avec {base_model_name} comme base.")
        self.model.summary()

    def train(self, epochs=50, fine_tune_epochs=10, model_name=None):
        """Entraîne le modèle construit."""
        if not self.model:
            raise RuntimeError(
                "Vous devez d'abord construire le modèle avec `build_model()`."
            )

        # Callbacks
        logdir = "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir, histogram_freq=1
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        callbacks = [tensorboard_callback, early_stopping_callback]

        # --- Phase 1: Entraînement de la tête de classification ---
        print("\n--- Phase 1: Entraînement de la tête de classification ---")
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=callbacks,
        )

        self.history = history

        # --- Phase 2: Fine-tuning ---
        print("\n--- Phase 2: Début de la phase de Fine-Tuning ---")

        if self.base_model and fine_tune_epochs > 0:
            self.base_model.trainable = True

            # Geler les premières couches et dégeler le reste
            fine_tune_at = 100
            for layer in self.base_model.layers[:fine_tune_at]:
                layer.trainable = False

            # Re-compiler avec un taux d'apprentissage très faible
            optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=1e-5)
            self.model.compile(
                optimizer=optimizer_fine_tune,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"],
            )

            # Continuer l'entraînement
            initial_epoch = history.epoch[-1] + 1
            fine_tune_history = self.model.fit(
                self.train_data,
                validation_data=self.val_data,
                epochs=initial_epoch + fine_tune_epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
            )

            self.fine_tune_history = fine_tune_history

        else:
            print("Aucun modèle de base trouvé, la phase de fine-tuning est sautée.")

        if model_name:
            save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(save_dir, f"{model_name}.keras")
            self.model.save(full_path)
            print(f"Modèle sauvegardé dans {full_path}")

    def load_model(self, model_path):
        """
        Charge un modèle préalablement sauvegardé.

        Args:
            model_path (str): Chemin vers le fichier du modèle sauvegardé.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Modèle chargé avec succès depuis {model_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            raise

    def _plot_history(self, history, title):
        """Trace les courbes d'apprentissage pour la perte et l'exactitude."""
        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title(f"{title} - Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title(f"{title} - Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.tight_layout()
        plt.show()

    def plot_history(self):
        if hasattr(self, "history"):
            self._plot_history(self.history, "Phase 1: Entraînement de la tête")
        if hasattr(self, "fine_tune_history"):
            self._plot_history(self.fine_tune_history, "Phase 2: Fine-Tuning")

    def _prepare_image(self, image_path):
        """Charge une image et la prépare pour le modèle."""
        img = tf.keras.utils.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        # Ajouter une dimension de lot (batch) car le modèle s'attend à un lot d'images
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return img_array_expanded

    def predict(self, image_path):
        """Effectue une prédiction sur une image donnée."""

        prepped_image = self._prepare_image(image_path)

        predictions = self.model.predict(prepped_image)

        predicted_class_index = np.argmax(predictions[0])
        confidence = 100 * np.max(predictions[0])
        predicted_class_name = self.class_names[predicted_class_index]

        print(
            f"La classe prédite est: {predicted_class_name} avec une confiance de {confidence:.2f}%"
        )

        return predicted_class_name, confidence
