import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomZoom, RandomRotation, RandomHeight, RandomWidth
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Activation
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from helper_functions import create_tensorboard_callback

# region
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

mixed_precision.set_global_policy(policy="mixed_float16")

# datasets_list = tfds.list_builders() # get all available datasets in TFDS

(train_data, test_data), ds_info = tfds.load("food101",
                                             split=["train", "validation"],
                                             batch_size=BATCH_SIZE,
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)
class_names = ds_info.features["label"].names
print(class_names[:10])

# batch & prepare datasets
def preprocess_img(image, label, img_shape=224):
    image = tf.image.resize(image, [img_shape, img_shape])
    return tf.cast(image, tf.float32), label

train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

base_model = EfficientNetB0(include_top=False)
base_model.trainable = False

callback_early_stopping = EarlyStopping(monitor="val_loss", patience=3)
callback_reduce_lr = ReduceLROnPlateau(factor=0.2,
                                       patience=2,
                                       min_lr=1e-7)

inputs = Input(shpae=IMAGE_SIZE+(3,), dtype=tf.float16)
# x = data_augmentation(inputs)
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(10),
outputs = Activation("softmax", dtype=tf.float32)(x)    # 将数据转换回float32
model = Model(inputs, outputs)

model.compile(loss="sparse_categorical_crossentropy",   # label不是one-hot编码形式
              optimizer=Adam(),
              metrics=["accuracy"])

model.fit(train_data,
          epochs=100,
          steps_per_epochs=len(train_data),
          validation_data=test_data,
          validation_steps=int(0.15*len(test_data)),
          callbacks=[callback_early_stopping,
                     callback_reduce_lr])


# endregion