"""This file implements CNN class pipeline to recognize medicine"""

import os
from typing import Literal

import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Recognizer:
    """
    Recognizer class that implements a convolutional neural network (CNN) pipeline
    to recognize medicine from image datasets.
    """

    def __init__(self, image_dataset: str = "medicine_databese_augmented") -> None:
        """
        Initializes the Recognizer instance.

        Parameters:
            image_dataset (str): Path to the image dataset directory.
        """
        self.set_image_dataset(image_dataset)
        self.__last_model_trained = None
        self.__train_data = None
        self.__validation_data = None
        self.__model = None

    @property
    def image_dataset(self) -> str:
        """Returns the path of the image dataset."""
        return self.__image_dataset

    @image_dataset.setter
    def image_dataset(self, image_dataset: str) -> None:
        """Sets the image dataset path with validation."""
        if not isinstance(image_dataset, str):
            raise TypeError(
                f"image_dataset must be a string, instead got {type(image_dataset)}"
            )
        self.__image_dataset = image_dataset

    def set_image_dataset(self, image_dataset: str) -> None:
        """Wrapper to set the image dataset."""
        self.image_dataset = image_dataset

    @property
    def last_model_trained(self) -> keras.models.Sequential:
        """Returns the last trained model."""
        return self.__last_model_trained

    @last_model_trained.setter
    def last_model_trained(self, last_model_trained):
        """Sets the last trained model with validation."""
        if not isinstance(last_model_trained, keras.models.Sequential):
            raise TypeError(
                f"last_model_trained must be a keras.models.Sequential or None, instead got {type(last_model_trained)}"
            )
        self.__last_model_trained = last_model_trained

    @property
    def train_data(self):
        """Returns the training data generator."""
        return self.__train_data

    @train_data.setter
    def train_data(self, train_data) -> None:
        """Sets the training data generator with validation."""
        if not isinstance(train_data, keras.preprocessing.image.DirectoryIterator):
            raise TypeError(
                f"train_data must be DirectoryIterator, instead got {type(train_data)}"
            )
        self.__train_data = train_data

    @property
    def validation_data(self):
        """Returns the validation data generator."""
        return self.__validation_data

    @validation_data.setter
    def validation_data(self, validation_data) -> None:
        """Sets the validation data generator with validation."""
        if not isinstance(validation_data, keras.preprocessing.image.DirectoryIterator):
            raise TypeError(
                f"validation_data must be DirectoryIterator, instead got {type(validation_data)}"
            )
        self.__validation_data = validation_data

    def create_train_validation_data(
        self,
        rescale: float = 1.0 / 255,
        rotation_range: int = 5,
        horizontal_flip: bool = True,
        zoom_range: float = 0.2,
        validation_split: float = 0.25,
        batch_size: int = 64,
        target_size: tuple = (256, 256),
        class_mode: Literal[
            "categorical", "binary", "input", "multi_output", "raw", "sparse"
        ] = "categorical",
    ) -> None:
        """
        Generates train and validation datasets from directory.

        Parameters:
            rescale (float): Rescaling factor for images.
            rotation_range (int): Degree range for random rotations.
            horizontal_flip (bool): Whether to randomly flip images horizontally.
            zoom_range (float): Range for random zoom.
            validation_split (float): Fraction of data to reserve for validation.
            batch_size (int): Number of samples per batch.
            target_size (tuple): Image size (height, width).
            class_mode (str): Type of classification.
        """
        image_generator = ImageDataGenerator(
            rescale=rescale,
            rotation_range=rotation_range,
            horizontal_flip=horizontal_flip,
            zoom_range=zoom_range,
            validation_split=validation_split,
        )

        train_generator = image_generator.flow_from_directory(
            self.image_dataset,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True,
            subset="training",
        )

        validation_generator = image_generator.flow_from_directory(
            self.image_dataset,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False,
            subset="validation",
        )

        self.train_data = train_generator
        self.validation_data = validation_generator

    @property
    def model(self) -> keras.models.Sequential:
        """Returns the compiled model."""
        return self.__model

    @model.setter
    def model(self, model) -> None:
        """Sets the compiled model with validation."""
        if not isinstance(model, keras.models.Sequential):
            raise TypeError(
                f"model must be a keras.models.Sequential or None, instead got {type(model)}"
            )
        self.__model = model

    def create_model(
        self,
        layers: list = [
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(256, 256, 3),
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=512, activation="relu"),
            Dense(units=256, activation="relu"),
            Dense(units=128, activation="relu"),
            Dense(units=4, activation="softmax"),
        ],
        optimizer: Literal[
            "sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"
        ] = "adam",
        loss: Literal[
            "mean_squared_error",
            "mse",
            "mean_absolute_error",
            "mae",
            "mean_absolute_percentage_error",
            "mape",
            "mean_squared_logarithmic_error",
            "msle",
            "squared_hinge",
            "hinge",
            "categorical_hinge",
            "logcosh",
            "huber",
            "huber_loss",
            "categorical_crossentropy",
            "sparse_categorical_crossentropy",
            "binary_crossentropy",
            "kullback_leibler_divergence",
            "kld",
            "poisson",
            "cosine_similarity",
        ] = "categorical_crossentropy",
        metrics: list = ["accuracy"],
    ) -> None:
        """
        Creates and compiles the CNN model.

        Parameters:
            layers (list): List of Keras layers.
            optimizer (str): Optimizer to use.
            loss (str): Loss function.
            metrics (list): List of metrics to monitor.
        """
        model = Sequential()
        for layer in layers:
            model.add(layer)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model = model

    def train_model(self, epochs: int = 100) -> None:
        """
        Trains the CNN model using training data.

        Parameters:
            epochs (int): Number of training epochs.
        """
        self.last_model_trained = self.model
        self.last_model_trained.fit(
            self.train_data, validation_data=self.validation_data, epochs=epochs
        )

    def save_model(self, path: str = "models/recognizer_model.h5") -> None:
        """
        Saves the last trained model to disk.

        Parameters:
            path (str): File path to save the model.
        """
        if self.last_model_trained is None:
            raise ValueError("No model has been trained yet to save.")
        self.last_model_trained.save(path)
        print(f"Model saved to: {path}")

    def run(
        self, epochs: int = 100, save_path: str = "models/recognizer_model.h5"
    ) -> None:
        """
        Runs the complete pipeline: data loading, model creation, training, and saving.

        Parameters:
            epochs (int): Number of epochs to train the model.
            save_path (str): Path to save the trained model.
        """
        print("Creating training and validation data...")
        self.create_train_validation_data()
        print("Creating model...")
        self.create_model()
        print("Training model...")
        self.train_model(epochs=epochs)
        print("Saving model...")
        self.save_model(path=save_path)
        print("Pipeline complete!")
