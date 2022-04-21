import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# 1. 回归问题
# 数据准备
insurance = pd.read_csv("data/insurance.csv")

# Create column transformer (this will help us normalize/preprocess our data)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Create X & y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ct.fit(X_train)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# 模型建立
tf.random.set_seed(42)

# Add an extra layer and increase number of units
insurance_model = tf.keras.Sequential([
  tf.keras.layers.Dense(100),  # 100 units
  tf.keras.layers.Dense(10),  # 10 units
  tf.keras.layers.Dense(1)  # 1 unit (important for output layer)
])

insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't
                        metrics=['mae'])

insurance_model.fit(X_train_normal, y_train, epochs=200, verbose=1)

# 保存模型
insurance_model.save("best_model_HDF5_format.h5")

# 加载模型
loaded_h5_model = tf.keras.models.load_model("best_model_HDF5_format.h5")


