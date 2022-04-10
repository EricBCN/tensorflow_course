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

callback_tensorboard = create_tensorboard_callback(dir_name="transfer_learning",
                                                   experiment_name="model_0")
callback_checkpoint = ModelCheckpoint(filepath="transfer_learning_checkpoint.ckpt",
                                      monitor="val_loss",
                                      save_best_only=False,
                                      save_weights_only=True,
                                      save_freq="epoch")

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

inputs = Input(shape=IMAGE_SIZE+(3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation="softmax")(x)
model_1 = Model(inputs, outputs)

model_1.compile(loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=int(0.25*len(test_data)),
                        callbacks=[callback_tensorboard,
                                   callback_checkpoint])
# endregion

# region
# Fine tuning
base_model.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False

# 需要重新编译模型
model_1.compile(loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

fine_tune_epochs = 10

history_fine = model_1.fit(train_data,
                           epochs=fine_tune_epochs,
                           steps_per_epoch=len(train_data),
                           validation_data=test_data,
                           validation_steps=int(len(test_data)*0.25),
                           initial_epoch=history_1.epoch[-1],
                           callbacks=[]
                           )
# endregion