import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Embedding, Dense, Input, Conv1D, GlobalAvgPool1D, \
    GlobalMaxPool1D, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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

train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

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

# region Char embedding layer
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

# region USE embedding layer
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")
# endregion

# region Callbacks
callback_ModelCheckpoint = ModelCheckpoint(filepath="best_weights/checkpoint.ckpt",
                                           save_best_only=True,
                                           save_weights_only=True)

callback_EarlyStopping = EarlyStopping(patience=3,
                                       restore_best_weights=True,
                                       min_delta=0.5)

call_ReduceLROnPlateau = ReduceLROnPlateau(factor=0.2,
                                           patience=3,
                                           min_lr=1e-7)
# endreion

# endregion

# 1. Setup token inputs model
token_inputs = Input(shape=[], dtype=tf.string)
x = tf_hub_embedding_layer(token_inputs)
token_outputs = Dense(128, activation="relu")(x)
token_model = tf.keras.Model(token_inputs, token_outputs)

# 2. Setup char inputs model (Bidirectional LSTM)
char_inputs = Input(shape=(1,), dtype=tf.string)
y = char_vectorizer(char_inputs)
y = char_embed(y)
char_outputs = Bidirectional(LSTM(32))(y)
char_model = tf.keras.Model(char_inputs, char_outputs)

# 3. Setup line numbers inputs model
line_number_inputs = Input(shape=(15,), dtype=tf.int32)
z = Dense(32, activation="relu")(line_number_inputs)
line_number_model = tf.keras.Model(line_number_inputs, z)

# 4. Setup total lines inputs model
total_lines_inputs = Input(shape=(20,), dtype=tf.int32)
w = Dense(32, activation="relu")(total_lines_inputs)
total_lines_model = tf.keras.Model(total_lines_inputs, w)

# 5. Combine token and char embeddings into a hybrid embedding
token_char_concat = Concatenate(name="token_char_hybrid")([token_model.output,
                                                           char_model.output])
s = Dense(256, activaion="relu")(token_char_concat)
s = Dropout(0.5)(s)

# 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
s = Concatenate(name="token_char_position_embedding")([line_number_model.output,
                                                       total_lines_model.output,
                                                       s])

# 7. Create output layer
output_layer = Dense(num_classes, activation="softmax")(s)
model_5 = tf.keras.Model(inputs=[line_number_model.input,
                                 total_lines_model.input,
                                 token_model.input,
                                 char_model.input],
                         outputs=output_layer)

model_5.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                optimizer=Adam(),
                metrics=["accuracy"])

# Combine chars and tokens into a dataset
train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,
                                                                train_total_lines_one_hot,
                                                                train_sentences,
                                                                train_chars))
train_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
train_pos_char_token_dataset = tf.data.Dataset.zip((train_pos_char_token_data, train_pos_char_token_labels))
train_pos_char_token_dataset = train_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                               val_total_lines_one_hot,
                                                               val_sentences,
                                                               val_chars))
val_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_pos_char_token_dataset = tf.data.Dataset.zip((val_pos_char_token_data, val_pos_char_token_labels))
val_pos_char_token_dataset = val_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

model_5.fit(train_pos_char_token_dataset,
            epochs=3,
            steps_per_epoch=int(0.1 * len(train_pos_char_token_dataset)),
            validation_data=val_pos_char_token_dataset,
            validation_steps=int(0.1 * len(val_pos_char_token_dataset)),
            callbacks=[callback_ModelCheckpoint,
                       callback_EarlyStopping,
                       call_ReduceLROnPlateau])

model_5_pred_probs = model_5.predict(val_pos_char_token_dataset)
model_5_preds = tf.argmax(model_5_pred_probs, axis=1)

