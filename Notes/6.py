import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Activation
from tensorflow.keras import mixed_precision

# region 数据载入
# datasets_list = tfds.list_builders()  # get all available datasets in TFDS

# Load in the data (takes about 5-6 minutes in Google Colab)
(train_data, test_data), ds_info = tfds.load(name="food101",  # target dataset to get from TFDS
                                             split=["train", "validation"],  # what splits of data should we get? note: not all datasets have train, valid, test
                                             shuffle_files=True,  # shuffle files on download?
                                             as_supervised=True,  # download data in tuple format (sample, label), e.g. (image, label)
                                             with_info=True)

class_names = ds_info.features["label"].names
print(class_names)
# endregion

# region 预处理数据
# Make a function for preprocessing images 改变尺寸并将数据类型变成float32
def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])  # reshape to img_shape
    return tf.cast(image, tf.float32), label  # return (float32_image, label) tuple


# 数据集batch与prefetch
# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map preprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)
# endregion

# region 建立callback
checkpoint_path = "model_checkpoints/cp.ckpt"  # saving weights requires ".ckpt" extension
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      montior="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      verbose=1)
# endregion

# region 建立模型
# 精度策略
mixed_precision.set_global_policy(policy="mixed_float16")

input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB0(includ_top=False)
base_model.trainable = False

inputs = Input(shape=input_shape, dtype=tf.float16)
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(len(class_names))(x)
outputs = Activation(activation="softmax", dtype=tf.float32)(x)   # 这边将输出层的激活函数分开写，主要是为了将输出值类型转换回float32
model = tf.keras.Model(inputs, outputs)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history_101_food_classes_feature_extract = model.fit(train_data,
                                                     epochs=3,
                                                     steps_per_epoch=len(train_data),
                                                     validation_data=test_data,
                                                     validation_steps=int(0.15*len(test_data)),
                                                     callbacks=[model_checkpoint])
# endregion

# region 加载并评估checkpoint权重模型
cloned_model = tf.keras.models.clone_model(model)
cloned_model.load_weights(checkpoint_path)

# 需要重新编译模型
cloned_model.compile(loss="sparse_categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])

results_cloned_model_with_loaded_weights = cloned_model.evaluate(test_data)
# endregion

# region Fine-tuning
loaded_gs_model = tf.keras.models.load_model("07_efficientnetb0_feature_extract_model_mixed_precision")

# 注意：因为训练集非常大，所以可以将所有的层都设置为可训练的
# 如果训练集比较小，则只训练一小部分的层就行，否则会过拟合
for layer in loaded_gs_model.layers:
    layer.trainable = True  # set all layers to trainable

# callbacks
# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",  # watch the val loss metric
                                                  patience=3)  # if val loss decreases for 3 epochs in a row, stop training

# Create ModelCheckpoint callback to save best model during fine-tuning
checkpoint_path = "fine_tune_checkpoints/"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor="val_loss")

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.2,  # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1,  # print out when learning rate goes down
                                                 min_lr=1e-7)

# Compile the model
loaded_gs_model.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
                        optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
                        metrics=["accuracy"])

# Start to fine-tune (all layers)
history_101_food_classes_all_data_fine_tune = loaded_gs_model.fit(train_data,
                                                                  epochs=100, # fine-tune for a maximum of 100 epochs
                                                                  steps_per_epoch=len(train_data),
                                                                  validation_data=test_data,
                                                                  validation_steps=int(0.15 * len(test_data)),
                                                                  callbacks=[model_checkpoint,
                                                                             early_stopping,
                                                                             reduce_lr])
# endregion
