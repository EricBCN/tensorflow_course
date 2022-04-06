# 神经网络分类
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import plot_model
from helper_functions import make_confusion_matrix
from sklearn.datasets import make_circles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# region
# A few activation functions
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


def relu(x):
    return tf.maximum(0, x)


# endregion

# # region
# # 例子：圆环（二元分类问题）
# n_samples = 1000
#
# X, y = make_circles(n_samples=n_samples,
#                     noise=0.03,
#                     random_state=42)
#
#
# circles = pd.DataFrame({"X0": X[:, 0],
#                         "X1": X[:, 1],
#                         "label": y})
# plt.figure(figsize=(10, 7))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()
#
# # Create the model
# tf.random.set_seed(42)
#
# model_1 = tf.keras.Sequential([
#     Dense(10, activation="relu"),  # 隐藏层的激活函数很多情况下要用非线性函数，分类问题用relu
#     Dense(10, activation="relu"),
#     Dense(1, activation="sigmoid")  # 输出层的激活函数，二元分类问题用sigmoid，多元分类用softmax
# ])
#
# model_1.compile(loss="binary_crossentropy",
#                 optimizer=Adam(),
#                 metrics=["accuracy"])
#
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch/20))
#
# history_1 = model_1.fit(X, y,
#                        epochs=100,
#                        callbacks=[lr_scheduler])
#
# pd.DataFrame(history_1.history).plot()  # loss, accuracy曲线
#
# # 学习速度与损失值之间的关系曲线
# lrs = 1e-4 * (10 ** (np.arange(100)/20))
# plt.figure(figsize=(10, 7))
# plt.semilogx(lrs, history_1.history["loss"])
# plt.xlabel("Learning Rate")
# plt.ylabel("Loss")
# plt.title("Learning rate vs. loss")
# # endregion

# region
# MNIST服饰的灰度照片集 （多元分类问题）
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
# train_data[0].shape: (28, 28)
# plt.imshow(train_data[0])
# plt.show()

# 数据归一化
train_data = train_data / 255.0
test_data = test_data / 255.0

# 类别列表
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# create the model
model_2 = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),  # 将二维数组平铺成一维
    Dense(10, activation="relu"),
    Dense(10, activation="relu"),
    Dense(10, activation="softmax") # 一共有10个种类
])

model_2.compile(loss="sparse_categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_2.fit(train_data, train_labels,
            epochs=10,
            validation_data=(test_data, test_labels))

print(model_2.evaluate(test_data, test_labels))

# 计算预测值（概率值数组）
y_probs = model_2.predict(test_data)

# 转换成类别名称
y_preds = y_probs.argmax(axis=1)    # 利用argmax函数

make_confusion_matrix(y_pred=y_preds,
                      y_true=test_labels)

plot_model(model_2)
plt.show()
# endregion
