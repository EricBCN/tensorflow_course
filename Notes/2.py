import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import pandas as pd
from sklearn.datasets import make_circles

# # region 二元分类问题
# # 数据准备
# n_samples = 1000
#
# X, y = make_circles(n_samples,
#                     noise=0.03,
#                     random_state=42)    # X有两个维度
#
# X_train, y_train = X[:800], y[:800] # 80% of the data for the training set
# X_test, y_test = X[800:], y[800:] # 20% of the data for the test set
#
# # 建立模型
# # Set random seed
# tf.random.set_seed(42)
#
# # Create a model (same as model_8)
# model_9 = tf.keras.Sequential([
#   tf.keras.layers.Dense(4, activation="relu"),
#   tf.keras.layers.Dense(4, activation="relu"),
#   tf.keras.layers.Dense(1, activation="sigmoid")
# ])
#
# # Compile the model
# model_9.compile(loss="binary_crossentropy", # we can use strings here too
#               optimizer="Adam", # same as tf.keras.optimizers.Adam() with default settings
#               metrics=["accuracy"])
#
# # Create a learning rate scheduler callback
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))  # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch
#
# # Fit the model (passing the lr_scheduler callback)
# history = model_9.fit(X_train,
#                       y_train,
#                       epochs=100,
#                       callbacks=[lr_scheduler])
#
# print(model_9.evaluate(X_test, y_test))
#
#
#
# # Set the random seed
# tf.random.set_seed(42)
#
# # Create the model
# model_10 = tf.keras.Sequential([
#   tf.keras.layers.Dense(4, activation="relu"),
#   tf.keras.layers.Dense(4, activation="relu"),
#   tf.keras.layers.Dense(1, activation="sigmoid")
# ])
#
# # Compile the model with the ideal learning rate
# model_10.compile(loss="binary_crossentropy",
#                 optimizer=tf.keras.optimizers.Adam(lr=0.02), # to adjust the learning rate, you need to use tf.keras.optimizers.Adam (not "adam")
#                 metrics=["accuracy"])
#
# # Fit the model for 20 epochs (5 less than before)
# model_10.fit(X_train, y_train, epochs=20)
# print(model_10.evaluate(X_test, y_test))
# # endregion

# region 多元分类问题
# 数据准备
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
train_data = train_data / 255.0
test_data = test_data / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 模型建立
# Set random seed
tf.random.set_seed(42)

# Create the model
model_12 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # input layer (we had to reshape 28x28 to 784)
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax")  # output shape is 10, activation is softmax
])

# Compile the model
model_12.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

# Fit the model (to the normalized data)
norm_history = model_12.fit(train_data,
                            train_labels,
                            epochs=20,
                            validation_data=(test_data, test_labels))

print(model_12.evaluate(test_data, test_labels))

y_probs = model_12.predict(test_data)  # "probs" is short for probabilities
y_preds = y_probs.argmax(axis=1)
# endregion