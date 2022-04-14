import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Conv1D, Input, GlobalMaxPooling1D, MaxPooling1D, Dropout
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import os
import numpy as np
import pathlib


data_dir = "20_newsgroup/"
dir_names = os.listdir(data_dir)

samples = []
labels = []
class_names = []
class_index = 0

for dir_name in sorted(os.listdir(data_dir)):
    class_names.append(dir_name)
    dir_path = data_dir + "/" + dir_name
    file_names = os.listdir(dir_path)
    print("Processing %s, %d files found" % (dir_name, len(file_names)))

    for file_name in file_names:
        file_path = dir_path + "/" + file_name
        f = open(file_path, encoding="latin-1")
        content = f.read()
        lines = content.split("\n")
        lines = lines[10:]
        content = "\n".join(lines)
        samples.append(content)
        labels.append(class_index)
        f.close()

    class_index += 1

# Shuffle the data
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(samples)
rng = np.random.RandomState(seed)
rng.shuffle(labels)

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(samples))
train_samples = samples[:-num_validation_samples]
val_samples = samples[-num_validation_samples:]
train_labels = labels[:-num_validation_samples]
val_labels = labels[-num_validation_samples:]

# Create a vocabulary index
vectorizer = TextVectorization(max_tokens=20000,
                               output_sequence_length=200)

text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128).prefetch(tf.data.AUTOTUNE)
vectorizer.adapt(text_ds)

vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

# Load pre-trained word embeddings
path_to_glove_file = "glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs


num_tokens = len(vocab) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1


embedding_layer = Embedding(input_dim=num_tokens,
                            output_dim=embedding_dim,
                            embeddings_initializer=tf.keras.initializer.Constant(embedding_matrix),
                            trainable=False)

inputs = Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(inputs)
x = Conv1D(128, 5, activation="relu")(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation="relu")(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation="relu")(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

y_train = np.array(train_labels)
y_val = np.array(val_labels)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["acc"])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_val, y_val))

