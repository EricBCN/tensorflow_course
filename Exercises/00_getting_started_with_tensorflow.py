import tensorflow as tf
import numpy as np

# Exercise 1
# region
vector_1 = tf.constant([1, 2, 3])
scalar_1 = tf.constant(8)
matrix_1 = tf.constant([[1, 3],
                        [2, 6]], dtype=tf.int32)
tensor_1 = tf.constant([[[1, 2],
                         [9, 8],
                         [8, 6]],
                        [[9, 3],
                         [3, 4],
                         [5, 6]]])
# endregion

# Exercise 2
# region
print(f"Shape: {vector_1.shape}, Rank: {vector_1.ndim}, Size: {tf.size(vector_1).numpy()}")
print(f"Shape: {scalar_1.shape}, Rank: {scalar_1.ndim}, Size: {tf.size(scalar_1).numpy()}")
print(f"Shape: {matrix_1.shape}, Rank: {tf.rank(matrix_1)}, Size: {tf.size(matrix_1).numpy()}")
print(f"Shape: {tensor_1.shape}, Rank: {tensor_1.ndim}, Size: {tf.size(tensor_1).numpy()}")
# endregion

# Exercise 3
# region
random_1 = tf.random.Generator.from_seed(1).uniform(shape=(5, 300), minval=0, maxval=1)
random_2 = tf.random.Generator.from_seed(5).uniform(shape=(5, 300), minval=0, maxval=1)
# print(random_1)
# print(random_2)
# endregion

# Exercise 4
# region
print(tf.matmul(random_1, tf.transpose(random_2)))
# endregion

# Exercise 5
# region
print(tf.tensordot(random_1, tf.transpose(random_2), axes=1))
# endregion

# Exercise 6
# region
random_3 = tf.random.Generator.from_seed(7).uniform(shape=(224, 224, 3), minval=0, maxval=1)
print(random_3)
# endregion

# Exercise 7
# region
print(f"Min: {tf.reduce_min(random_3)}, Max: {tf.reduce_max(random_3)}")
# endregion

# Exercise 8
# region
random_4 = tf.random.Generator.from_seed(7).uniform(shape=(1, 224, 224, 3), minval=0, maxval=1)
print(random_4)
print(tf.squeeze(random_4))
# endregion

# Exercise 9
# region
tensor_9 = [1, 4, 3, -3, 9, 1, 8, 3, -12, 1]
print(tf.argmax(tensor_9))
# endregion

# Exercise 10
# region
print(tf.one_hot(tensor_9, depth=10))
# endregion


