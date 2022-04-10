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

# region
# 将文本转换成数字 (Tokenization & Embeddings)
# Tokenization: word-level, character-level, sub-word tokenization
# Embeddings: create your own embedding, reuse a pre-learned embedding
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
# endregion

# region Model_0: Naive Bayes
model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()),  # 用tfidf将单词转换成数字
    ("clf", MultinomialNB())  # model the text
])

model_0.fit(train_sentences, train_labels)
baseline_score = model_0.score(val_sentences, val_labels)
baseline_preds = model_0.predict(val_sentences)
# endregion

# region Model_1: A simple dense model
SAVE_DIR = "model_logs"
inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = GlobalAveragePooling1D()(x)
outputs = Dense(1, activation="sigmoid")(x)
model_1 = tf.keras.Model(inputs, outputs, name="model_1")

model_1.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_1.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name="model_1")])

model_1.evaluate(val_sentences, val_labels)
model_1_pred_probs = model_1.predict(val_sentences)
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_1_results = calculate_results(y_true=val_labels,
                                    y_pred=model_1_preds)

print(model_1.summary())
# endregion

# region Model_2: LSTM
# 重新建立一层embedding，因为之前在model_1中embedding层已经被训练过了，参数都已经被训练确定的
model_2_embedding = Embedding(input_dim=max_vocab_length,
                              output_dim=128,
                              input_length=max_length,
                              name="embedding_2")

inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_2_embedding(x)
x = LSTM(64)(x)
outputs = Dense(1, activation="sigmoid")
model_2 = tf.keras.Model(inputs, outputs, name="model_2")

model_2.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_2.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name="model_2")])
# endregion

# region Model_3: GRU
# 重新建立一层embedding，因为之前在model_1中embedding层已经被训练过了，参数都已经被训练确定的
model_3_embedding = Embedding(input_dim=max_vocab_length,
                              output_dim=128,
                              input_length=max_length,
                              name="embedding_3")

inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_3_embedding(x)
x = GRU(64)(x)
outputs = Dense(1, activation="sigmoid")
model_3 = tf.keras.Model(inputs, outputs, name="model_3")

model_3.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_3.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name="model_3")])
# endregion

# region Model_4: Bidirectional RNN model
# 重新建立一层embedding，因为之前在model_1中embedding层已经被训练过了，参数都已经被训练确定的
model_4_embedding = Embedding(input_dim=max_vocab_length,
                              output_dim=128,
                              input_length=max_length,
                              name="embedding_4")

inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_4_embedding(x)
x = Bidirectional(LSTM(64))(x)
outputs = Dense(1, activation="sigmoid")
model_4 = tf.keras.Model(inputs, outputs, name="model_4")

model_4.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_4.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name="model_4")])
# endregion

# region Model_5: Conv1D
# 重新建立一层embedding，因为之前在model_1中embedding层已经被训练过了，参数都已经被训练确定的
model_5_embedding = Embedding(input_dim=max_vocab_length,
                              output_dim=128,
                              input_length=max_length,
                              name="embedding_5")

inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_5_embedding(x)
x = Conv1D(filters=32, kernel_size=5, activation="relu")(x)
x = GlobalMaxPooling1D()(x)
outputs = Dense(1, activation="sigmoid")
model_5 = tf.keras.Model(inputs, outputs, name="model_5")

model_5.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_5.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name="model_5")])
# endregion

# region Model_6: Transfer learning (USE)
# 用Universal Sentence Encoder来代替tokenization和embedding层
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        input_shape=[],
                                        dtype=tf.string,
                                        name="USE")
model_6 = tf.keras.Sequential([
    sentence_encoder_layer,
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
], name="model_6")

model_6.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_6.fit(train_sentences, train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name="model_6")])
# endregion

# region Model_7: Transfer learning (USE, 10% of the training data)
train_sentences_90_percent, train_sentences_10_percent, \
train_labels_90_percent, train_labels_10_percent = train_test_split(np.array(train_sentences),
                                                                    train_labels,
                                                                    test_size=0.1,
                                                                    random_state=42)
model_7 = tf.keras.models.clone_model(model_6)

model_7.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_7.fit(train_sentences_10_percent,
            train_labels_10_percent,
            epochs=5,
            validation_data=(val_sentences, val_labels),
            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
                                                   experiment_name="model_7")])
# endregion
