import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = boston_housing.load_data(test_split=0.2)

print(X_train[:5])

model_boston = tf.keras.Sequential([
    Dense(100),
    Dense(100),
    Dense(10),
    Dense(1)
])

model_boston.compile(loss="mae",
                     optimizer=Adam(),
                     metrics=["mae"])

history_model_boston = model_boston.fit(X_train,
                                        y_train,
                                        epochs=200)

print(model_boston.evaluate(X_test, y_test))

pd.DataFrame(history_model_boston.history).plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

