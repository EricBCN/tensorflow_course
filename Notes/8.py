import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, \
    LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPool1D, Concatenate, Dropout
import pandas as pd
import numpy as np
import string
import json
from spacy.lang.en import English
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from helper_functions import calculate_results


def get_lines(filename):
    with open(filename, "r") as f:
        return f.readlines()


# 返回字典列表
def preprocess_text_with_line_numbers(filename):
    input_lines = get_lines(filename)  # get all lines from filename
    abstract_lines = ""  # create an empty abstract
    abstract_samples = []  # create an empty list of abstracts

    # Loop through each line in target file
    for line in input_lines:
        if line.startswith("###"):  # check to see if line is an ID line
            abstract_lines = ""  # reset abstract string
        elif line.isspace():  # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines()  # split abstract into separate lines
            total_lines = len(abstract_line_split) - 1

            # Iterate through each line in abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}  # 空字典，用于存储行信息
                target_text_split = abstract_line.split("\t")  # 按照tab进行拆分
                line_data["target"] = target_text_split[0]  # 类别
                line_data["text"] = target_text_split[1].lower()  # 文本内容
                line_data["line_number"] = abstract_line_number  # 行数编号
                line_data["total_lines"] = total_lines  # 总行数
                abstract_samples.append(line_data)  # add line data to abstract samples list

        else:
            abstract_lines += line

    return abstract_samples


def split_chars(text):
  return " ".join(list(text))


# region 准备数据
data_dir = "../pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "dev.txt")  # dev is another name for validation set
test_samples = preprocess_text_with_line_numbers(data_dir + "test.txt")

# 转换成dataframe
train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)

# 获取语句列表
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

# 将标签变成one-hot编码
one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

# 将标签变成int编码
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
# endregion

# region 模型0：朴素贝叶斯
model_0 = Pipeline([
  ("tf-idf", TfidfVectorizer()),
  ("clf", MultinomialNB())
])

# model_0.fit(X=train_sentences,
#             y=train_labels_encoded)
# endregion

# region 建立Tokenization和Embedding层
sent_lens = [len(sentence.split()) for sentence in train_sentences]
output_seq_len = int(np.percentile(sent_lens, 95))  # 语句长度的95%分位数，作为输出大小

max_tokens = 68000
text_vectorizer = TextVectorization(max_tokens=max_tokens,
                                    output_sequence_length=output_seq_len)
text_vectorizer.adapt(train_sentences)
rct_20k_text_vocab = text_vectorizer.get_vocabulary()

token_embed = Embedding(input_dim=len(rct_20k_text_vocab), # length of vocabulary
                        output_dim=128, # Note: different embedding sizes result in drastically different numbers of parameters to train
                        # Use masking to handle variable sequence lengths (save space)
                        mask_zero=True,
                        name="token_embedding")
# endregion

# region 建立数据集（利用batch和prefetch使得数据载入更快）
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))

train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
# endregion

# region 模型1：卷积网络
inputs = Input(shape=(1, ), dtype=tf.string)
text_vector = text_vectorizer(inputs)
token_embedding = token_embed(text_vector)
x = Conv1D(64, 5, padding="same", activation="relu")(token_embedding)
x = GlobalAveragePooling1D()(x)
outputs = Dense(num_classes, activation="softmax")(x)
model_1 = tf.keras.Model(inputs, outputs)

model_1.compile(loss="categorical_crossentropy",  # if your labels are integer form (not one hot) use sparse_categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# model_1_history = model_1.fit(train_dataset,
#                               steps_per_epoch=int(0.1 * len(train_dataset)),  # only fit on 10% of batches for faster training time
#                               epochs=3,
#                               validation_data=valid_dataset,
#                               validation_steps=int(0.1 * len(valid_dataset)))  # only validate on 10% of batches
# endregion

# region 模型2：迁移学习（Universal Sentence Encoder）
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")

inputs = Input(shape=[], dtype=tf.string)
pretrained_embedding = tf_hub_embedding_layer(inputs)
x = Dense(128, activation="relu")(pretrained_embedding)
outputs = Dense(num_classes, activation="softmax")(x)
model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(loss="categorical_crossentropy",  # if your labels are integer form (not one hot) use sparse_categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# model_2_history = model_2.fit(train_dataset,
#                               steps_per_epoch=int(0.1 * len(train_dataset)),  # only fit on 10% of batches for faster training time
#                               epochs=3,
#                               validation_data=valid_dataset,
#                               validation_steps=int(0.1 * len(valid_dataset)))  # only validate on 10% of batches
# endregion

# region 模型3：卷积网络（token为字符）
train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]

char_lens = [len(sentence) for sentence in train_sentences]
output_seq_char_len = int(np.percentile(char_lens, 95))

alphabet = string.ascii_lowercase + string.digits + string.punctuation

NUM_CHAR_TOKENS = len(alphabet) + 2 # num characters in alphabet + space + OOV token
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")
char_vectorizer.adapt(train_chars)

char_embed = Embedding(input_dim=NUM_CHAR_TOKENS,  # number of different characters
                       output_dim=25,  # embedding dimension of each character (same as Figure 1 in https://arxiv.org/pdf/1612.05251.pdf)
                       mask_zero=False,  # don't use masks (this messes up model_5 if set to True)
                       name="char_embed")

train_char_dataset = tf.data.Dataset.from_tensor_slices((train_chars, train_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
val_char_dataset = tf.data.Dataset.from_tensor_slices((val_chars, val_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)

inputs = Input(shape=(1, ), dtype=tf.string)
char_vector = char_vectorizer(inputs)
char_embedding = char_embed(char_vector)
x = Conv1D(64, 5, padding="same", activation="relu")(char_embedding)
x = GlobalMaxPool1D()(x)
outputs = Dense(num_classes, activation="softmax")(x)
model_3 = tf.keras.Model(inputs, outputs)

model_3.compile(loss="categorical_crossentropy",  # if your labels are integer form (not one hot) use sparse_categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# model_3_history = model_1.fit(train_char_dataset,
#                               steps_per_epoch=int(0.1 * len(train_char_dataset)),  # only fit on 10% of batches for faster training time
#                               epochs=3,
#                               validation_data=val_char_dataset,
#                               validation_steps=int(0.1 * len(val_char_dataset)))  # only validate on 10% of batches
# endregion

# region 模型4：混合模型（预训练模型+字符模型）
# 1. Setup token inputs/model
token_inputs = Input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_output = Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs,
                             outputs=token_output)

# 2. Setup char inputs/model
char_inputs = Input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = Bidirectional(LSTM(25))(char_embeddings) # bi-LSTM shown in Figure 1 of https://arxiv.org/pdf/1612.05251.pdf
char_model = tf.keras.Model(inputs=char_inputs,
                            outputs=char_bi_lstm)

# 3. Concatenate token and char inputs (create hybrid token embedding)
token_char_concat = Concatenate(name="token_char_hybrid")([token_model.output,
                                                                  char_model.output])

# 4. Create output layers - addition of dropout discussed in 4.2 of https://arxiv.org/pdf/1612.05251.pdf
combined_dropout = Dropout(0.5)(token_char_concat)
combined_dense = Dense(200, activation="relu")(combined_dropout) # slightly different to Figure 1 due to different shapes of token/char embedding layers
final_dropout = Dropout(0.5)(combined_dense)
output_layer = Dense(num_classes, activation="softmax")(final_dropout)

# 5. Construct model with char and token inputs
model_4 = tf.keras.Model(inputs=[token_model.input, char_model.input],
                         outputs=output_layer,
                         name="model_4_token_and_char_embeddings")

# Combine chars and tokens into a dataset
train_char_token_data = tf.data.Dataset.from_tensor_slices((train_sentences, train_chars))  # make data
train_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)  # make labels
train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels))  # combine data and labels
train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Repeat same steps validation data
val_char_token_data = tf.data.Dataset.from_tensor_slices((val_sentences, val_chars))
val_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# model_4_history = model_4.fit(train_char_token_dataset,  # train on dataset of token and characters
#                               steps_per_epoch=int(0.1 * len(train_char_token_dataset)),
#                               epochs=3,
#                               validation_data=val_char_token_dataset,
#                               validation_steps=int(0.1 * len(val_char_token_dataset)))
# endregion

# region 模型5：混合模型（预训练模型+字符模型+行数信息）
# 用train_df["line_number"].value_counts()可以看到，大部分的行编号都在15以内
train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)

# 用np.percentile(train_df.total_lines, 98)可以看到，98%的总行数都在20行内
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)


# 1. Token inputs
token_inputs = Input(shape=[], dtype="string", name="token_inputs")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_outputs = Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs,
                             outputs=token_outputs)

# 2. Char inputs
char_inputs = Input(shape=(1,), dtype="string", name="char_inputs")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = Bidirectional(LSTM(32))(char_embeddings)
char_model = tf.keras.Model(inputs=char_inputs,
                            outputs=char_bi_lstm)

# 3. Line numbers inputs
line_number_inputs = Input(shape=(15,), dtype=tf.int32, name="line_number_input")
x = Dense(32, activation="relu")(line_number_inputs)
line_number_model = tf.keras.Model(inputs=line_number_inputs,
                                   outputs=x)

# 4. Total lines inputs
total_lines_inputs = Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
y = Dense(32, activation="relu")(total_lines_inputs)
total_line_model = tf.keras.Model(inputs=total_lines_inputs,
                                  outputs=y)

# 5. Combine token and char embeddings into a hybrid embedding
combined_embeddings = Concatenate(name="token_char_hybrid_embedding")([token_model.output,
                                                                       char_model.output])
z = Dense(256, activation="relu")(combined_embeddings)
z = Dropout(0.5)(z)

# 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
z = Concatenate(name="token_char_positional_embedding")([line_number_model.output,
                                                         total_line_model.output,
                                                         z])

# 7. Create output layer
output_layer = Dense(5, activation="softmax", name="output_layer")(z)

# 8. Put together model
model_5 = tf.keras.Model(inputs=[line_number_model.input,
                                 total_line_model.input,
                                 token_model.input,
                                 char_model.input],
                         outputs=output_layer)

# Compile token, char, positional embedding model
model_5.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),  # add label smoothing (examples which are really confident get smoothed a little)
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Create training and validation datasets (all four kinds of inputs)
train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,  # line numbers
                                                                train_total_lines_one_hot,  # total lines
                                                                train_sentences,  # train tokens
                                                                train_chars))  # train chars
train_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)  # train labels
train_pos_char_token_dataset = tf.data.Dataset.zip((train_pos_char_token_data, train_pos_char_token_labels))  # combine data and labels
train_pos_char_token_dataset = train_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Validation dataset
val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                              val_total_lines_one_hot,
                                                              val_sentences,
                                                              val_chars))
val_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_pos_char_token_dataset = tf.data.Dataset.zip((val_pos_char_token_data, val_pos_char_token_labels))
val_pos_char_token_dataset = val_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

history_model_5 = model_5.fit(train_pos_char_token_dataset,
                              steps_per_epoch=int(0.1 * len(train_pos_char_token_dataset)),
                              epochs=3,
                              validation_data=val_pos_char_token_dataset,
                              validation_steps=int(0.1 * len(val_pos_char_token_dataset)))

# 对测试集进行评估
# Create test dataset batch and prefetched
test_pos_char_token_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,
                                                               test_total_lines_one_hot,
                                                               test_sentences,
                                                               test_chars))
test_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_pos_char_token_dataset = tf.data.Dataset.zip((test_pos_char_token_data, test_pos_char_token_labels))
test_pos_char_token_dataset = test_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

test_pred_probs = model_5.predict(test_pos_char_token_dataset,
                                  verbose=1)
test_preds = tf.argmax(test_pred_probs, axis=1)
test_results = calculate_results(y_true=test_labels_encoded,
                                 y_pred=test_preds)
print(test_results)

# 读取json文件，并对其内的摘要进行分类
with open("skimlit_example_abstracts.json", "r") as f:
    example_abstracts = json.load(f)

abstracts = pd.DataFrame(example_abstracts)

nlp = English()
nlp.add_pipe('sentencizer')

doc = nlp(example_abstracts[0]["abstract"])  # create "doc" of parsed sequences, change index for a different abstract
abstract_lines = [str(sent) for sent in list(doc.sents)]  # return detected sentences from doc in string type (not spaCy token type)

# Get total number of lines
total_lines_in_sample = len(abstract_lines)

# Go through each line in abstract and create a list of dictionaries containing features for each line
sample_lines = []
for i, line in enumerate(abstract_lines):
    sample_dict = {}
    sample_dict["text"] = str(line)
    sample_dict["line_number"] = i
    sample_dict["total_lines"] = total_lines_in_sample - 1
    sample_lines.append(sample_dict)

test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)
test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

test_abstract_pred_probs = model_5.predict(x=(test_abstract_line_numbers_one_hot,
                                           test_abstract_total_lines_one_hot,
                                           tf.constant(abstract_lines),
                                           tf.constant(abstract_chars)))
test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]
print(test_abstract_pred_classes)
# endregion