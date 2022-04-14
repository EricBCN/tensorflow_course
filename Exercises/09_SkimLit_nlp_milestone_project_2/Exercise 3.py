import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd


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

preprocess_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                                  trainable=False)
bert_layer = hub.KerasLayer('https://tfhub.dev/google/experts/bert/pubmed/2',
                            trainable=False)

# preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
# bert = hub.load('https://tfhub.dev/google/experts/bert/pubmed/2')
#
# bert_inputs = preprocess(sentences)
#
# bert_outputs = bert(bert_inputs, training=False)
# pooled_output = bert_outputs['pooled_output']
# sequence_output = bert_outputs['sequence_output']

inputs = Input(shape=[], dtype=tf.string)
x = preprocess_layer(inputs)
x = bert_layer(x)
x = Dense(128, activation="relu")(x['pooled_output'])
x = Dense(num_classes, activation="softmax")(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])

train_sentences_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
val_sentences_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))

train_sentences_dataset = train_sentences_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
val_sentences_dataset = val_sentences_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

model.fit(train_sentences_dataset,
          epochs=3,
          steps_per_epoch=int(0.1 * len(train_sentences_dataset)),
          validation_data=val_sentences_dataset,
          validation_steps=int(0.1 * len(val_sentences_dataset)))

model_pred_probs = model.predict(val_sentences_dataset)
model_preds = tf.argmax(model_pred_probs, axis=1)
# endregion