import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Exercise 1
# region
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

tf.random.set_seed(42)

train_dir = "../10_food_classes_10_percent/train/"
test_dir = "../10_food_classes_10_percent/test/"
train_datagen = ImageDataGenerator(1/255.)
test_datagen = ImageDataGenerator(1/255.)

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=IMAGE_SIZE,
                                               class_mode="categorical",
                                               seed=42,
                                               batch_size=BATCH_SIZE)
test_data = test_datagen.flow_from_directory(test_dir,
                                            target_size=IMAGE_SIZE,
                                            class_mode="categorical",
                                            seed=42,
                                            batch_size=BATCH_SIZE)

mobilenet_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
mobilenet_layer = hub.KerasLayer(mobilenet_url, trainable=False, input_shape=IMAGE_SIZE + (3,))
model_mobilenet = tf.keras.Sequential([
    mobilenet_layer,
    Dense(train_data.num_classes, activation="softmax")
])
model_mobilenet.compile(loss="categorical_crossentropy",
                        optimizer=Adam(),
                        metrics=["accuracy"])

model_mobilenet.fit(train_data,
                    epochs=5,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))
# endregion

# Exercise 2
# region

# endregion