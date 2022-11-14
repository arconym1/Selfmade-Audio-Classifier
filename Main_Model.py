import os
import random

import numpy as np
from keras import models
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import cv2


def create_training_data(Categories: list, Data_dir: str, training_data: list):
    for category in Categories:
        path: str = os.path.join(Data_dir, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))  # Gray Scale / 2D
                new_array = cv2.resize(img_array, (64, 64))
                training_data.append([new_array, class_num])

            except Exception as e:
                print(e)


def train():
    Data_dir: str = "DirectoryOfYourData"
    Categories: list = ["Your", "Categories"] # As many Categories as you want
    training_data: list = []

    create_training_data(Categories, Data_dir, training_data)
    random.shuffle(training_data)
    X, y = [], []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3))) # You can try to get better results by changing the input shape ( maybe you find better network architecture )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))  # Or sigmoid

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=200) # Your epochs
    model.evaluate(X_test, y_test)

    model.save(
        "YourModelName.h5")


if __name__ == '__main__':
    train()
