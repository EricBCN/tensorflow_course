import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np
import string


def get_lines(filename):
    with open(filename, "r") as f:
        return f.readlines()


def preprocess_text_with_line_numbers(filename):
    input_lines = get_lines(filename)
    abstract_lines = ""
    abstract_samples = []
    num = 1

    for line in input_lines:
        if line.startswith("###"):
            abstract_id = line
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines()

            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_lines"] = len(abstract_line_split) - 1
                line_data["line_number_total"] = str(line_data["line_number"]) + "_of_" + str(line_data["total_lines"])
                abstract_samples.append(line_data)

            if num % 1000 == 0:
                print(num)

            num += 1
        else:
            abstract_lines += line

    return abstract_samples


def split_chars(text):
    return " ".join(list(text))


# region Preprocess data
data_dir = "../../pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "dev.txt")
test_samples = preprocess_text_with_line_numbers(data_dir + "test.txt")

train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)

# train_df['line_number_total'] = train_df['line_number'].astype(str) + '_of_' + train_df['total_lines'].astype(str)
# val_df['line_number_total'] = val_df['line_number'].astype(str) + '_of_' + val_df['total_lines'].astype(str)

train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

# 将属性line_number_total转换为数字：one-hot编码形式
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder.fit(np.expand_dims(train_df['line_number_total'], axis=1))
train_line_number_total_encoded = one_hot_encoder.transform(np.expand_dims(train_df['line_number_total'], axis=1))
val_line_number_total_encoded = one_hot_encoder.transform(np.expand_dims(val_df['line_number_total'], axis=1))

train_line_number_total_encoded = train_line_number_total_encoded.toarray()
val_line_number_total_encoded = val_line_number_total_encoded.toarray()

train_line_number_total_encoded = tf.cast(train_line_number_total_encoded, dtype=tf.int32)
val_line_number_total_encoded = tf.cast(val_line_number_total_encoded, dtype=tf.int32)


# 将标签label转换为数字：one-hot编码形式，或是int整型
one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

num_classes = len(label_encoder.classes_)

train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)

train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=15)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=15)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=15)
# endregion

# region Char Embedding
train_chars =[split_chars(sentence) for sentence in train_sentences]
val_chars =[split_chars(sentence) for sentence in val_sentences]
test_chars =[split_chars(sentence) for sentence in test_sentences]

char_len = [len(sentence) for sentence in train_sentences]
output_seq_char_len = int(np.percentile(char_len, 95))

alphabet = string.ascii_lowercase + string.digits + string.punctuation
NUM_CHAR_TOKEN = len(alphabet) + 2  # 多出来的2个是空格和OOV
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKEN,
                                    output_sequence_length=output_seq_char_len,
                                    name="char_vectorizer")
char_vectorizer.adapt(train_chars)
char_vocab = char_vectorizer.get_vocabulary()

char_embed = Embedding(input_dim=char_vocab,
                       output_dim=25,
                       mask_zero=False,
                       name="char_embed")
# endregion


preprocess_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                                  trainable=False)
bert_layer = hub.KerasLayer('https://tfhub.dev/google/experts/bert/pubmed/2',
                            trainable=False)

# region Build the model
# Buidling the tribid model using the functional api
preprocess_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                                  trainable=False)
bert_layer = hub.KerasLayer('https://tfhub.dev/google/experts/bert/pubmed/2',
                            trainable=False)

input_token = Input(shape=[], dtype =tf.string)
bert_inputs_token = preprocess_layer(input_token)
bert_embedding_char =bert_layer(bert_inputs_token)
output_token = Dense(64 , activation = 'relu')(bert_embedding_char['pooled_output'])
token_model = tf.keras.Model(input_token , output_token)

input_char = Input(shape=[], dtype=tf.string)
bert_inputs_char = preprocess_layer(input_char)
bert_embedding_char =bert_layer(bert_inputs_char)
output_char = Dense(64 , activation = 'relu')(bert_embedding_char['pooled_output'])
char_model = tf.keras.Model(input_char , output_char)

line_number_total_input = Input(shape=(460,), dtype=tf.int32)
dense = Dense(32, activation = 'relu')(line_number_total_input)
total_line_number_model = tf.keras.Model(line_number_total_input, dense)

# Concatenating the tokens amd chars output (Hybrid!!!)
combined_embeddings = Concatenate(name='token_char_hybrid_embedding')([token_model.output,
                                                                       char_model.output])

# Combining the line_number_total to our hybrid model (Time for Tribid!!)
z = Concatenate(name='tribid_embeddings')([total_line_number_model.output,
                                           combined_embeddings])

# Adding a dense + dropout and creating our output layer
dropout = Dropout(0.5)(z)
x = Dense(128 , activation='relu')(dropout)
output_layer = Dense(5 , activation='softmax')(x)

# Packing into a model
tribid_model = tf.keras.Model(inputs=[token_model.input,
                                      char_model.inpu,
                                      total_line_number_model.input],
                              outputs=output_layer)
# endregion

tribid_model.compile(loss="categorical_crossentropy",
                     optimizer=Adam(),
                     metrics=["accuracy"])

train_data = tf.data.Dataset.from_tensor_slices((train_sentences,
                                                 train_chars,
                                                 train_line_number_total_encoded))
train_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
train_dataset = tf.data.Dataset.zip((train_data, train_labels))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_tensor_slices((val_sentences,
                                               val_chars,
                                               val_line_number_total_encoded))
val_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_dataset = tf.data.Dataset.zip((val_data, val_labels))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

tribid_model.fit(train_dataset,
                 epochs=3,
                 steps_per_epoch=int(0.1 * len(train_dataset)),
                 validation_data=val_dataset,
                 validation_steps=int(0.1 * len(val_dataset)))

model_pred_probs = tribid_model.predict(val_dataset)
model_preds = tf.argmax(model_pred_probs, axis=1)
# endregion