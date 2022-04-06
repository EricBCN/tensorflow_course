import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_moons
from tensorflow.keras.datasets import fashion_mnist
from helper_functions import make_confusion_matrix
import matplotlib.pyplot as plt

# region
# Exercise 2
model_2 = tf.keras.Sequential([
    Dense(6, activation="relu"),
    Dense(6, activation="relu"),
    Dense(6, activation="relu"),
    Dense(6, activation="relu"),
    Dense(6, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_2.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])
# endregion

# region
# Exercise 3
X, y = make_moons(n_samples=1000,
                  noise=0.02,
                  random_state=42)

model_3 = tf.keras.Sequential([
    Dense(10, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_3.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

# history_3 = model_3.fit(X, y, epochs=100)

# endregion

# region
# Exercise 6
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

model_6 = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax")
])

model_6.compile(loss="sparse_categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_6.fit(train_data, train_labels,
            epochs=20)

print(model_6.evaluate(test_data, test_labels))

y_probs = model_6.predict(test_data)
y_preds = y_probs.argmax(axis=1)
make_confusion_matrix(y_true=test_labels,
                      y_pred=y_preds)
plt.show()
# endregion

