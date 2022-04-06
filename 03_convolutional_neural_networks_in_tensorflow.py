import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_moons
from tensorflow.keras.datasets import fashion_mnist
from helper_functions import make_confusion_matrix, load_and_prep_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import numpy as np

# train_dir = pathlib.Path("pizza_steak/train/")
# class_names = np.array(sorted([item.name for item in train_dir.glob("*")]))
# print(class_names)

# # region
# # 披萨/牛排 （图片二元分类问题）
# tf.random.set_seed(42)
#
# # 准备数据 （利用tensorflow.keras.preprocessing.image.ImageDataGenerator）
# # 文件夹已经按照种类分类好了，所以我们只需要指定文件夹地址就行了
# train_datagen = ImageDataGenerator(rescale=1/255.)
# test_datagen = ImageDataGenerator(rescale=1/255.)
#
# # 利用数据增强技术来防止过拟合
# train_datagen_agumented = ImageDataGenerator(rescale=1/255.,
#                                              rotation_range=20,
#                                              zoom_range=0.2,
#                                              shear_range=0.2,
#                                              width_shift_range=0.2,
#                                              height_shift_range=0.2,
#                                              horizontal_flip=True)
#
# train_dir = "pizza_steak/train/"
# test_dir = "pizza_steak/test/"
#
# train_data = train_datagen.flow_from_directory(train_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="binary",
#                                                seed=42,
#                                                shuffle=True)
#
# train_data_augmented = train_datagen_agumented.flow_from_directory(train_dir,
#                                                                    batch_size=32,
#                                                                    target_size=(224, 224),
#                                                                    class_mode="binary",
#                                                                    seed=42,
#                                                                    shuffle=True)
#
# test_data = test_datagen.flow_from_directory(test_dir,
#                                              batch_size=32,
#                                              target_size=(224, 224),
#                                              class_mode="binary",
#                                              seed=42,
#                                              shuffle=True)
#
# # 2层Conv2D，1层MaxPool2D，2层Conv2D，1层MaxPool2D，一层Flatten，输出层
# model_1 = tf.keras.Sequential([
#     Conv2D(filters=10,
#            kernel_size=3,
#            activation="relu",
#            input_shape=(224, 224, 3)),
#     Conv2D(10, 3, activation="relu"),
#     MaxPool2D(pool_size=2,
#               padding="valid"), # 防止过拟合
#     Conv2D(10, 3, activation="relu"),
#     Conv2D(10, 3, activation="relu"),
#     MaxPool2D(2),
#     Flatten(),
#     Dense(1, activation="sigmoid")
# ])
#
# model_1.compile(loss="binary_crossentropy",
#                 optimizer=Adam(),
#                 metrics=["accuracy"])
#
# model_1.fit(train_data_augmented,
#             epochs=5,
#             steps_per_epoch=len(train_data_augmented),
#             validation_data=test_data,
#             validation_steps=len(test_data))
# # endregion


# region
# 多元分类问题
train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               shuffle=True,
                                               seed=42,
                                               class_mode="categorical")

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             shuffle=True,
                                             seed=42,
                                             class_mode="categorical")

# 获取train_data中的图片（以batch的形式）
# images, labels = train_data.next()
# print(labels[:5])   # label已经是one-hot编码形式了

model_2 = tf.keras.Sequential([
    Conv2D(kernel_size=3, filters=10, input_shape=(224, 224, 3)),
    MaxPool2D(),
    Conv2D(kernel_size=3, filters=10),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax")
])

model_2.compile(loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_2.fit(train_data,
            epochs=5,
            steps_per_epoch=len(train_data),
            validation_data=test_data,
            validation_steps=len(test_data))

# 预测
# img = load_and_prep_image("...")
# y_pred = model_2.predict(tf.expand_dims(img, axis=0)) # 增加一个维度：batch

# endregion