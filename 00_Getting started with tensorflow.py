import tensorflow as tf
import random
import numpy as np
import pandas as pd

print(tf.__version__)

scalar = tf.constant(7)
print(scalar.numpy())

vector = tf.constant([10, 10])
print(vector.ndim)

matrix = tf.constant([[10, 7],
                      [3, 4],
                      [5, 0]], dtype=tf.float16)

# 可变及不可变tensor
changeable_tensor = tf.Variable([10, 7])
print(changeable_tensor)

changeable_tensor[0].assign(8)
print(changeable_tensor)

# 随机生成tensor
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3, 2))

tf.random.set_seed(42)

# shuffle
print(matrix)
matrix = tf.random.shuffle(matrix)  # 此处可以设置操作层面上的seed，如：tf.random.shuffle(matrix, seed=12)
print(matrix)

# 其他生成tensor的方式
tf.ones(shape=(3, 2))
tf.zeros(shape=(4, 5))

# 从numpy数组生成tensor，tensor转变成bumpy
numpy_A = np.arange(1, 25, dtype=np.int32)
A = tf.constant(numpy_A, shape=[2, 3, 4])
print(A)

print(A.numpy())

# 获取tensor的属性
print(A.shape, A.ndim, tf.size(A))

# 增加维度
B = A[..., tf.newaxis]  # 或者
B = tf.expand_dims(A, axis=-1)

# 重塑形状，转置
tf.reshape(A, shape=(8, 3))
tf.transpose(A)

# 矩阵乘法
matrix_1 = tf.constant([[1, 3, 4],
                        [3, 5, 6],
                        [10, 8, 8]])
matrix_2 = tf.constant([[1, 2],
                        [4, 7],
                        [9, 1]])
print(tf.matmul(matrix_1, matrix_2))  # 或者
print(matrix_1 @ matrix_2)

# 最大，最小，平均，和，绝对值，标准差，方差，最大值所在位置，最小值所在位置
tf.reduce_min(A)
tf.reduce_max(A)
tf.reduce_mean(A)
tf.reduce_sum(A)
tf.abs(A)
tf.reduce_std(A)
tf.reduce_variant(A)
tf.argmax(A)
tf.argmin(A)

# 去除多余维度
tf.squeeze(A)

# One-hot编码
tf.one_hot(A, depth=10)
tf.one_hot(A, depth=10, on_value="True", off_value="False")

# 平方，平方根，对数
tf.square(A)
tf.sqrt(A)
tf.math.log(A)
