import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomZoom, RandomRotation, RandomHeight, RandomWidth
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from helper_functions import create_tensorboard_callback

# region
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

tf.random.set_seed(42)

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_data = image_dataset_from_directory(train_dir,
                                          image_size=IMAGE_SIZE,
                                          batch_size=BATCH_SIZE,
                                          label_mode="categorical")
test_data = image_dataset_from_directory(test_dir,
                                         image_size=IMAGE_SIZE,
                                         batch_size=BATCH_SIZE,
                                         label_mode="categorical")

callback_checkpoint = ModelCheckpoint(filepath="transfer_learning_checkpoint.ckpt",
                                      save_best_only=False,
                                      save_weights_only=True)

base_model = EfficientNetB0(include_top=False)
base_model.trainable = False

# Data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomHeight(0.2),
    RandomWidth(0.2)
])

inputs = Input(shpae=IMAGE_SIZE+(3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation="softmax")(x)
model = Model(inputs, outputs)
# endregion