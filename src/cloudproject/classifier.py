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

        self.model_registry = {
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "ResNet50V2": tf.keras.applications.ResNet50V2,
            "VGG16": tf.keras.applications.VGG16,
        }

        self.preprocessors = {
            tf.keras.applications.MobileNetV2: tf.keras.applications.mobilenet_v2.preprocess_input,
            tf.keras.applications.ResNet50V2: tf.keras.applications.resnet_v2.preprocess_input,
            tf.keras.applications.VGG16: tf.keras.applications.vgg16.preprocess_input,
        }

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
        """Charge une image depuis un chemin, la décode et la redimensionne.
        
        Args:
            path (str): Chemin vers l'image.
            label (int): Label de la classe de l'image.
        """
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        return img, label

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

        df = pd.read_csv(csv_path)
        df.rename(columns={df.columns[0]: "filename"}, inplace=True)

        df["full_path"] = df["filename"].apply(
            lambda name: os.path.join(image_dir, name)
        )

        image_paths = df["full_path"].values
        label_columns = df.drop(columns=["filename", "full_path"])
        labels = label_columns.values
        class_names = label_columns.columns.tolist()

        print(
            f"Chargement depuis '{image_dir}': {len(image_paths)} images trouvées pour les classes {class_names}."
        )

        path_label_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        image_label_ds = path_label_ds.map(
            self._load_and_resize_image, num_parallel_calls=self.autotune
        )

        if is_training:
            image_label_ds = image_label_ds.shuffle(buffer_size=len(image_paths))

        final_dataset = image_label_ds.batch(self.batch_size).prefetch(
            buffer_size=self.autotune
        )

        return final_dataset, class_names
    
    def add_model(self, name: str, model_class, preprocess_function):
        """
        Ajoute dynamiquement un nouveau modèle et sa fonction de pré-traitement au registre.

        Args:
            name (str): Le nom à donner au modèle (ex: "EfficientNetB0").
            model_class: La classe du modèle Keras (ex: tf.keras.applications.EfficientNetB0).
            preprocess_function: La fonction de pré-traitement associée (ex: tf.keras.applications.efficientnet.preprocess_input).
        """
        self.model_registry[name] = model_class
        self.preprocessors[model_class] = preprocess_function
        print(f"Modèle '{name}' ajouté au registre avec succès.")


    def build_model(self, base_model_name: str, learning_rate=0.001):
        """
        Construit un modèle de classification en utilisant le transfert d'apprentissage.

        Args:
            base_model_name (str): Le nom du modèle pré-entraîné à utiliser (ex: "MobileNetV2").
            learning_rate (float): Le taux d'apprentissage pour l'optimiseur.
        """
        base_model_class = self.model_registry.get(base_model_name)
        if not base_model_class:
            raise ValueError(
                f"Le nom du modèle '{base_model_name}' n'est pas reconnu. "
                f"Modèles disponibles : {list(self.model_registry.keys())}"
            )

        preprocess_input = self.preprocessors.get(base_model_class)

        self.base_model = base_model_class(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights="imagenet",
        )

        self.base_model.trainable = False

        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.class_names), activation="softmax")(x)

        self.model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        print(f"Modèle construit avec {base_model_name} comme base.")
        self.model.summary()

    def train(self, epochs=50, fine_tune_epochs=10, model_name=None, learning_rate_ft=0.001):
        """Entraîne le modèle construit."""
        if not self.model:
            raise RuntimeError(
                "Vous devez d'abord construire le modèle avec `build_model()`."
            )

        logdir = "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir, histogram_freq=1
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        callbacks = [tensorboard_callback, early_stopping_callback]

        print("\n--- Phase 1: Entraînement de la tête de classification ---")
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=callbacks,
        )

        self.history = history

        print("\n--- Phase 2: Début de la phase de Fine-Tuning ---")
        if self.base_model and fine_tune_epochs > 0:
            self.base_model.trainable = True

            fine_tune_at = 100
            for layer in self.base_model.layers[:fine_tune_at]:
                layer.trainable = False

            optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=learning_rate_ft)
            self.model.compile(
                optimizer=optimizer_fine_tune,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"],
            )

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
            full_model_path = os.path.join("../models", model_path)
            self.model = tf.keras.models.load_model(full_model_path)

            input_shape = self.model.input_shape
            self.img_height = input_shape[1]
            self.img_width = input_shape[2]

            print(f"Modèle chargé avec succès depuis {model_path}")
            print(f"Dimensions d'entrée du modèle mises à jour : {self.img_height}x{self.img_width}")
        
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            raise

    def _plot_history(self, history, title):
        """Trace les courbes d'apprentissage pour la perte et l'exactitude."""
        plt.figure(figsize=(12, 4))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title(f"{title} - Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title(f"{title} - Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.tight_layout()
        plt.show()

    def plot_history(self):
        if hasattr(self, "history"):
            self._plot_history(self.history, "Phase 1: Entraînement de la tête")
        if hasattr(self, "fine_tune_history"):
            self._plot_history(self.fine_tune_history, "Phase 2: Fine-Tuning")

    def _load_image_for_prediction(self, image_path):
        """Charge une image depuis un chemin et la convertit en tableau NumPy."""
        img = tf.keras.utils.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        return img, img_array
    
    def predict_from_array(self, img_array):
        """Effectue une prédiction sur un tableau NumPy d'image."""
        if not self.model:
            raise RuntimeError("Le modèle n'est pas chargé. Utilisez `build_model()` ou `load_model()` d'abord.")
        
        img_resized = tf.image.resize(img_array, [self.img_height, self.img_width])
        img_array_expanded = np.expand_dims(img_resized, axis=0)

        predictions = self.model.predict(img_array_expanded)
        predicted_class_index = np.argmax(predictions[0])
        confidence = 100 * np.max(predictions[0])
        predicted_class_name = self.class_names[predicted_class_index]

        return predicted_class_name, confidence

    def predict(self, image_path, show_image=True):
        """Effectue une prédiction sur une image donnée."""
        img, img_array = self._load_image_for_prediction(image_path)
        predicted_class_name, confidence = self.predict_from_array(img_array)

        title = f"Prédiction: {predicted_class_name} ({confidence:.2f}%)"
        print(title)

        if show_image:
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
            plt.show()

        return predicted_class_name, confidence
