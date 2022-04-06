import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from helper_functions import make_confusion_matrix, load_and_prep_image, create_tensorboard_callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import numpy as np

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

def create_model(model_url, num_classes=10):
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name="feature_extraction_layer",
                                             input_shape=IMAGE_SIZE + (3,))

    model = tf.keras.Sequential([
        feature_extractor_layer,
        Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model


train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                                          target_size=IMAGE_SIZE,
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=True,
                                                          seed=42,
                                                          class_mode="categorical")

test_data_10_percent = test_datagen.flow_from_directory(test_dir,
                                                        target_size=IMAGE_SIZE,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        seed=42,
                                                        class_mode="categorical")

# Resnet 50 V2 feature vector
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

# Original: EfficientNetB0 feature vector (version 1)
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

resnet_model = create_model(resnet_url, num_classes=train_data_10_percent.num_classes)
resnet_model.compile(loss="categorical_crossentropy",
                     optimizer=Adam(),
                     metrics=["accuracy"])
resnet_model.fit(train_data_10_percent,
                 epochs=5,
                 steps_per_epoch=len(train_data_10_percent),
                 validation_data=test_data_10_percent,
                 validation_steps=len(test_data_10_percent),
                 callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                        experiment_name="resnet50V2")])


efficientnet_model = create_model(efficientnet_url, num_classes=train_data_10_percent.num_classes)
efficientnet_model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(),
                           metrics=["accuracy"])
efficientnet_model.fit(train_data_10_percent,
                       epochs=5,
                       steps_per_epoch=len(train_data_10_percent),
                       validation_data=test_data_10_percent,
                       validation_steps=len(test_data_10_percent),
                       callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                              experiment_name="efficientnetB0")])


