import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, \
    RandomZoom, RandomHeight, RandomWidth


# 准备数据
train_dir = "../10_food_classes_10_percent/train/"
test_dir = "../10_food_classes_10_percent/test/"

IMG_SIZE = (224, 224)  # define image size
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical",
                                                                            batch_size=32)
test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                           image_size=IMG_SIZE,
                                                                           label_mode="categorical")

# 基准模型
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = Input(shape=IMG_SIZE + (3,))
x = base_model(inputs)
x = GlobalMaxPool2D()(x)
outputs = Dense(10, activation="softmax")(x)
model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# model_0.fit(train_data_10_percent,
#             epochs=5,
#             steps_per_epoch=len(train_data_10_percent),
#             validation_data=test_data_10_percent,
#             validation_steps=int(0.25*len(test_data_10_percent)))

print(model_0.summary())

# 建立模型（使用10%的数据来训练，同时使用fine-tuning和data augmentation技术
data_augmentation = tf.keras.Sequential([
  RandomFlip("horizontal"),
  RandomRotation(0.2),
  RandomZoom(0.2),
  RandomHeight(0.2),
  RandomWidth(0.2),
  # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
], name="data_augmentation")

checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt"
callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, # set to False to save the entire model
                                                         save_best_only=False, # set to True to save only the best model instead of a model every epoch
                                                         save_freq="epoch", # save every epoch
                                                         verbose=0)

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation="softmax")(x)
model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

initial_epochs = 5

history_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                          epochs=initial_epochs,
                                          validation_data=test_data_10_percent,
                                          validation_steps=int(0.25 * len(test_data_10_percent)),
                                          callbacks=[callback_checkpoint])


# region Fine-tuning
base_model.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False

# Recompile the model (always recompile after any adjustments to a model)
model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Fine tune for another 5 epochs
fine_tune_epochs = initial_epochs + 5

# Refit the model (same as model_2 except with more trainable layers)
history_fine_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                               epochs=fine_tune_epochs,
                                               validation_data=test_data_10_percent,
                                               initial_epoch=history_10_percent_data_aug.epoch[-1], # start from previous last epoch
                                               validation_steps=int(0.25 * len(test_data_10_percent)))
# endregion

# 获取
y_labels = []
for images, labels in test_data_10_percent.unbatch():  # unbatch the test data and get images and labels
    y_labels.append(labels.numpy().argmax())  # append the index which has the largest value (labels are one-hot)
