import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


# region 二元分类问题
# 数据准备
# Set the seed
tf.random.set_seed(42)

# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
# Create ImageDataGenerator training instance with data augmentation
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20,  # rotate the image slightly between 0 and 20 degrees (note: this is an int not a float)
                                             shear_range=0.2,  # shear the image
                                             zoom_range=0.2,  # zoom into the image
                                             width_shift_range=0.2,  # shift the image width ways
                                             height_shift_range=0.2,  # shift the image height ways
                                             horizontal_flip=True)  # flip the image on the horizontal axis

valid_datagen = ImageDataGenerator(rescale=1./255)

# Setup the train and test directories
train_dir = "../pizza_steak/train/"
test_dir = "../pizza_steak/test/"

# Import data from directories and turn it into batches
# Import data and augment it from directories
train_data_augmented_shuffled = train_datagen_augmented.flow_from_directory(train_dir,
                                                                            target_size=(224, 224),
                                                                            batch_size=32,
                                                                            class_mode='binary',
                                                                            shuffle=True) # Shuffle data (default)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

# 模型建立
# Create a CNN model (same as Tiny VGG - https://poloclub.github.io/cnn-explainer/)
model_1 = tf.keras.models.Sequential([
    Conv2D(filters=10,
         kernel_size=3,  # can also be (3, 3)
         activation="relu",
         input_shape=(224, 224, 3)),  # first layer specifies input shape (height, width, colour channels)
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
            padding="valid"),  # padding can also be 'same'
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),  # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
    MaxPool2D(2),
    Flatten(),
    Dense(1, activation="sigmoid")  # binary activation output
])

# Compile the model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Fit the model
history_1 = model_1.fit(train_data_augmented_shuffled,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented_shuffled),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

print(model_1.evaluate(valid_data))
# endregion


# region 多元分类问题
train_dir = "../10_food_classes_all_data/train/"
test_dir = "../10_food_classes_all_data/test/"

# Create augmented data generator instance
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20, # note: this is an int not a float
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(224, 224),
                                                                   batch_size=32,
                                                                   class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='categorical')

# Try a simplified model (removed two layers)
model_10 = tf.keras.Sequential([
    Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    MaxPool2D(),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

model_10.compile(loss='categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['accuracy'])

history_10 = model_10.fit(train_data_augmented,
                          epochs=5,
                          steps_per_epoch=len(train_data_augmented),
                          validation_data=test_data,
                          validation_steps=len(test_data))

print(model_10.evaluate(test_data))
# endregion
