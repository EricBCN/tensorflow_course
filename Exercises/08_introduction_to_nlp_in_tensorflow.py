import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, Dense, Input, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM, GRU, \
    Bidirectional, Conv1D
from tensorflow.keras.optimizers import Adam
from helper_functions import create_tensorboard_callback, calculate_results
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

tf.random.set_seed(42)

train_df = pd.read_csv("nlp_getting_started/train.csv")
test_df = pd.read_csv("nlp_getting_started/test.csv")

train_df_shuffled = train_df.sample(frac=1, random_state=42)

train_sentences, val_sentences, \
train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                            train_df_shuffled["target"].to_numpy(),
                                            random_state=42)

max_vocab_length = 10000
max_length = 15

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    standardize="lower_and_strip_punctuation",
                                    split="whitespace",
                                    output_mode="int",
                                    output_sequence_length=max_length)
text_vectorizer.adapt(train_sentences)

embedding = Embedding(input_dim=max_vocab_length,
                      output_dim=128,
                      embeddings_initializer="uniform",
                      input_length=max_length,
                      name="embedding_1")

# region Exercise 1
model_1 = tf.keras.Sequential([
    Input(shape=(1,), dtype=tf.string),
    text_vectorizer,
    embedding,
    GlobalAveragePooling1D(),
    Dense(1, activation="sigmoid")
])

model_2 = tf.keras.Sequential([
    Input(shape=(1,), dtype=tf.string),
    text_vectorizer,
    embedding,
    LSTM(64),
    Dense(1, activation="sigmoid")
])

model_5 = tf.keras.Sequential([
    Input(shape=(1,), dtype=tf.string),
    text_vectorizer,
    embedding,
    Conv1D(32, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(1, activation="sigmoid")
])
# endregion

# region Exercise 3
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=True,
                                        input_shape=[],
                                        dtype=tf.string,
                                        name="USE")
model_USE = tf.keras.Sequential([
    sentence_encoder_layer,
    Dense(1, activation="sigmoid")
], name="model_USE")

model_USE.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_USE.fit(train_sentences, train_labels,
              epochs=5,
              validation_data=(val_sentences, val_labels))
# endregion


