import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, \
    LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPool1D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# region 准备数据
train_df = pd.read_csv("../nlp_getting_started/train.csv")
test_df = pd.read_csv("../nlp_getting_started/test.csv")
train_df_shuffled = train_df.sample(frac=1, random_state=42)

train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42)
# endregion

# region 将文本转换成数字 (Tokenization & Embeddings)
max_vocab_length = 10000  # max number of words to have in our vocabulary
max_length = 15  # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
text_vectorizer.adapt(train_sentences)

embedding = Embedding(input_dim=max_vocab_length,  # set input shape
                      output_dim=128,  # set size of embedding vector
                      embeddings_initializer="uniform",  # default, intialize randomly
                      input_length=max_length,  # how long is each input
                      name="embedding_1")
# endregion

# region 模型0：baseline（朴素贝叶斯）
# Create tokenization and modelling pipeline
model_0 = Pipeline([
                    ("tfidf", TfidfVectorizer()),  # convert words to numbers using tfidf
                    ("clf", MultinomialNB())  # model the text
])

# Fit the pipeline to the training data
model_0.fit(train_sentences, train_labels)
# endregion

# region 模型1：简单的全连接模型
inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)

x = GlobalAveragePooling1D()(x)
outputs = Dense(1, activation="sigmoid")(x)
model_1 = tf.keras.Model(inputs, outputs)

model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_1_history = model_1.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))
# endregion

# region 模型2：LSTM
inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = LSTM(64)(x)
outputs = Dense(1, activation="sigmoid")(x)
model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_2_history = model_2.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))
# endregion

# region 模型3：GRU
inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = GRU(64)(x)
outputs = Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs)

model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3_history = model_1.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))
# endregion

# region 模型4：Bidirectional RNN
inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = Bidirectional(LSTM(64))(x)
outputs = Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs)

model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_4_history = model_4.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))
# endregion

# region 模型5：卷积网络
inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = Conv1D(filters=32, kernel_size=5, activation="relu")(x)
x = GlobalMaxPool1D()(x)
outputs = Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.Model(inputs, outputs)

model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_5_history = model_5.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))
# endregion

# region 模型6：迁移学习（Universal Sentence Encoder）
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False)

model_6 = tf.keras.Sequential([
    sentence_encoder_layer,
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_6.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_6_history = model_6.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))
# endregion

