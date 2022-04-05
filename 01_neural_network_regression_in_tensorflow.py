import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

# # region
# # 回归问题
# tf.random.set_seed(42)
#
# X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
# y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
#
# model = tf.keras.Sequential([
#     Dense(1)
# ])
#
# model.compile(loss="mae",
#               optimizer=Adam(),
#               metrics=["mae"])
#
# history = model.fit(tf.expand_dims(X, axis=-1),
#                     y,
#                     epochs=10)
#
# pd.DataFrame(history.history).plot()
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()
#
# print(tf.squeeze(model.predict([17.0])).numpy())
#
#
# # Save the model
# model.save("01_model_0.h5")
#
# # Load the model
# loaded_model = tf.keras.models.load_model("01_model_0.h5")
# print(tf.squeeze(loaded_model.predict([17.0])).numpy())
# # endregion

# region
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# 将非数值转换成数值，如sex, smoker...
insurance_one_hot = pd.get_dummies(insurance)

X = insurance_one_hot.drop(["charges"], axis=1)
y = insurance_one_hot["charges"]

# 归一化和标准化
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

X = insurance.drop("charges", axis=1)
y = insurance["charges"]

print(X[:5])

# 使用train_test_split来分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42,
                                                    test_size=0.2)

ct.fit(X_train)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(X_train_normal[:5])

# 建立模型
tf.random.set_seed(42)

insurance_model = tf.keras.Sequential([
    Dense(100),
    Dense(10),
    Dense(1)
])

insurance_model.compile(loss="mae",
                        optimizer=Adam(),
                        metrics=["mae"])

history_insurance = insurance_model.fit(X_train_normal, y_train,
                                        epochs=200)

print(insurance_model.evaluate(X_test_normal, y_test))
# endregion