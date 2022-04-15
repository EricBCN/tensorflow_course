###
# Model 1, 2, 3: 结构上没有区别，就是一般的单层全连接隐藏层（128个神经元）
# Model 4: 是卷积网络，需要注意的是，输入的尺寸要符合网络要求
# Model 5: 是LSTM网络，需要注意的是，输入的尺寸要符合网络要求
# Model 6: 是一般的全连接网络，只是输入数据多了一个属性：比特币的奖励值
# Model 7: 手动实现N-BEATS算法
# Model 8: 混合模型，用不同的loss function创建多个模型
# Model 9: 全数据训练来预测未来数据
# Model 10: Turkey问题 (黑天鹅，这里没有创建）
###
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Dropout, Input, Lambda, subtract, add
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt


# Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (setting a value will index from end of timesteps & values)
    label : label to show on plot of values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)


def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data).
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Find MAE of naive forecast (no seasonality)
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))  # our seasonality is 1 day (hence the shifting of 1 day)

    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    # Account for different sized metrics (for longer horizons, reduce to single number)
    if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}


# Create function to label windowed data
def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]


# Create function to view NumPy arrays as windows
def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    # print(f"Window step:\n {window_step}")

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
    # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split))  # this will default to 80% train/20% test
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels


def make_preds(model, input_data):
    """
    Uses model to make predictions on input_data.

    Parameters
    ----------
    model: trained model
    input_data: windowed input data (same kind of data model was trained on)

    Returns model predictions on input_data.
    """
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)  # return 1D array of predictions


# Create a function to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path="model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),  # create filepath to save model
                                              verbose=0,  # only output a limited amount of text
                                              save_best_only=True) # save only the best model to file


filepath = "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
df = pd.read_csv(filepath,
                 parse_dates=["Date"],
                 index_col=["Date"])

bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})

timesteps = []
btc_price = []

with open("BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv", "r") as f:
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader)    # 跳过第一行标题

    for line in csv_reader:
        timesteps.append(datetime.strptime(line[1], "%Y-%m-%d"))
        btc_price.append(float(line[2]))

timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()

split_size = int(0.8 * len(prices))
X_train, y_train = timesteps[:split_size], prices[:split_size]  # 日期
X_test, y_test = timesteps[split_size:], prices[split_size:]    # 价格

# region Model 0: baseline
naive_forecast = y_test[:-1]
naive_results = evaluate_preds(y_pred=y_test[1:],
                               y_true=naive_forecast)
# endregion

# region Model 1: Dense model (w=7, h=1)
tf.random.set_seed(42)

HORIZON = 1
WINDOW_SIZE = 7

full_windows, full_labels = make_windows(prices, WINDOW_SIZE, HORIZON)
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

model_1 = tf.keras.Sequential([
    Dense(128, activation="relu"),
    Dense(HORIZON, activation="linear")
], name="model_1_dense")

model_1.compile(loss="mae",
                optimizer=Adam(),
                metrics=["mae"])

model_1.fit(train_windows, train_labels,
            epochs=100,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_1.name)])

model_1 = tf.keras.models.load_model("model_experiments/model_1_dense") # 加载训练出的最佳结果

model_1_preds = make_preds(model_1, test_windows)
model_1_results = evaluate_preds(test_labels, model_1_preds)
# endregion

# region Model 2: Dense model (w=30, h=1)
tf.random.set_seed(42)

HORIZON = 1
WINDOW_SIZE = 30

full_windows, full_labels = make_windows(prices, WINDOW_SIZE, HORIZON)
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

model_2 = tf.keras.Sequential([
    Dense(128, activation="relu"),
    Dense(HORIZON)
], name="model_2_dense")

model_2.compile(loss="mae",
                optimizer=Adam(),
                metrics=["mae"])

model_2.fit(train_windows, train_labels,
            epochs=100,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_2.name)])

model_2 = tf.keras.models.load_model("model_experiments/model_2_dense") # 加载训练出的最佳结果

model_2_preds = make_preds(model_2, test_windows)
model_2_results = evaluate_preds(test_labels, model_2_preds)
# endregion

# region Model 3: Dense model (w=30, h=7)
tf.random.set_seed(42)

HORIZON = 7
WINDOW_SIZE = 30

full_windows, full_labels = make_windows(prices, WINDOW_SIZE, HORIZON)
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

model_3 = tf.keras.Sequential([
    Dense(128, activation="relu"),
    Dense(HORIZON)
], name="model_2_dense")

model_3.compile(loss="mae",
                optimizer=Adam(),
                metrics=["mae"])

model_3.fit(train_windows, train_labels,
            epochs=100,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_3.name)])

model_3 = tf.keras.models.load_model("model_experiments/model_3_dense") # 加载训练出的最佳结果

model_3_preds = make_preds(model_3, test_windows)
model_3_results = evaluate_preds(test_labels, model_3_preds)
# endregion

# region Model 4: Conv1D (w=7, h=1)
# The Conv1D layer in TensorFlow takes an input of: (batch_size, timesteps, input_dim)
tf.random.set_seed(42)

HORIZON = 1
WINDOW_SIZE = 7

full_windows, full_labels = make_windows(prices, WINDOW_SIZE, HORIZON)
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)


model_4 = tf.keras.Sequential([
    Lambda(lambda x: tf.expand_dims(x, axis=1)),  # 增加一个维度，以满足Conv1的输入尺寸要求
    Conv1D(128, 5, padding="casual", activation="relu"),    # 注意：对于时间序列，padding需要设置为casual
    Dense(HORIZON)
], name="model_4_conv1D")

model_4.compile(loss="mae",
                optimizer=Adam(),
                metrics=["mae"])

model_4.fit(train_windows, train_labels,
            epochs=100,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_4.name)])

model_4 = tf.keras.models.load_model("model_experiments/model_4_conv1D") # 加载训练出的最佳结果

model_4_preds = make_preds(model_4, test_windows)
model_4_results = evaluate_preds(test_labels, model_4_preds)
# endregion

# region Model 5: LSTM (w=7, h=1)
# The tf.keras.layers.LSTM() layer takes a tensor with [batch, timesteps, feature] dimensions
tf.random.set_seed(42)

HORIZON = 1
WINDOW_SIZE = 7

full_windows, full_labels = make_windows(prices, WINDOW_SIZE, HORIZON)
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

inputs = Input(shape=(WINDOW_SIZE))
x = Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # 增加一个维度，以满足LSTM的输入尺寸要求
# x = LSTM(128, activation="relu", return_sequences=True)(x)  # this layer will error if the inputs are not the right shape
x = LSTM(128, activation="relu")(x)
outputs = Dense(HORIZON)(x)
model_5 = tf.keras.Model(inputs, outputs, name="model_5_lstm")

model_5.compile(loss="mae",
                optimizer=Adam(),
                metrics=["mae"])

model_5.fit(train_windows, train_labels,
            epochs=100,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_5.name)])

model_5 = tf.keras.models.load_model("model_experiments/model_5_lstm") # 加载训练出的最佳结果

model_5_preds = make_preds(model_5, test_windows)
model_5_results = evaluate_preds(test_labels, model_5_preds)
print(model_5_results)
# endregion

# region Make a multivariate time series
# Block reward values
block_reward_1 = 50 # 3 January 2009 (2009-01-03)
block_reward_2 = 25 # 28 November 2012
block_reward_3 = 12.5 # 9 July 2016
block_reward_4 = 6.25 # 11 May 2020

# Block reward dates (datetime form of the above date stamps)
block_reward_2_datetime = np.datetime64("2012-11-28")
block_reward_3_datetime = np.datetime64("2016-07-09")
block_reward_4_datetime = np.datetime64("2020-05-11")

# Get date indexes for when to add in different block dates
block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days

# Add block_reward column
bitcoin_prices_block = bitcoin_prices.copy()
bitcoin_prices_block["block_reward"] = None

# Set values of block_reward column (it's the last column hence -1 indexing on iloc)
bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

# Making a windowed dataset with pandas
HORIZON = 1
WINDOW_SIZE = 7

bitcoin_prices_windowed = bitcoin_prices_block.copy()

# Add windowed columns
for i in range(WINDOW_SIZE):  # Shift values for each step in WINDOW_SIZE
    bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32)
y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)

split_size = int(len(X) * 0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]
# endregion

# region Model 6: Dense model (multivariate time series, w=7, h=1)
tf.random.set_seed(42)

HORIZON = 1
WINDOW_SIZE = 7

full_windows, full_labels = make_windows(prices, WINDOW_SIZE, HORIZON)
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

model_6 = tf.keras.Sequential([
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(HORIZON)
], name="model_6_dense_multivariate")

model_6.compile(loss="mae",
                optimizer=Adam(),
                metrics=["mae"])

model_6.fit(X_train, y_train,
            epochs=100,
            batch_size=128,
            validation_data=(X_test, y_test),
            callbacks=[create_model_checkpoint(model_6.name)])

model_6 = tf.keras.models.load_model("model_experiments/model_6_dense_multivariate") # 加载训练出的最佳结果

model_6_preds = make_preds(model_6, X_test)
model_6_results = evaluate_preds(y_test, model_6_preds)
# endregion

# region Model 7: N-BEATS algorithm (w=7, h=1)
# # Create NBeatsBlock custom layer
# class NBeatsBlock(tf.keras.layers.Layer):
#     def __init__(self,
#                  input_size: int,
#                  theta_size: int,
#                  horizon: int,
#                  n_neurons: int,
#                  n_layers: int,
#                  **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
#     super().__init__(**kwargs)
#     self.input_size = input_size
#     self.theta_size = theta_size
#     self.horizon = horizon
#     self.n_neurons = n_neurons
#     self.n_layers = n_layers
#
#     # Block contains stack of 4 fully connected layers each has ReLU activation
#     self.hidden = [Dense(n_neurons, activation="relu") for _ in range(n_layers)]
#     # Output of block is a theta layer with linear activation
#     self.theta_layer = Dense(theta_size, activation="linear", name="theta")
#
#     def call(self, inputs): # the call method is what runs when the layer is called
#         x = inputs
#
#         for layer in self.hidden: # pass inputs through each hidden layer
#             x = layer(x)
#
#         theta = self.theta_layer(x)
#
#         # Output the backcast and forecast from theta
#         backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
#         return backcast, forecast
#
# # Set up dummy NBeatsBlock layer to represent inputs and outputs
# dummy_nbeats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE,
#                                        theta_size=WINDOW_SIZE+HORIZON, # backcast + forecast
#                                        horizon=HORIZON,
#                                        n_neurons=128,
#                                        n_layers=4)
#
bitcoin_prices_nbeats = bitcoin_prices.copy()
for i in range(WINDOW_SIZE):
    bitcoin_prices_nbeats[f"Price+{i+1}"] = bitcoin_prices_nbeats["Price"].shift(periods=i+1)

# Make features and labels
X = bitcoin_prices_nbeats.dropna().drop("Price", axis=1)
y = bitcoin_prices_nbeats.dropna()["Price"]

# Make train and test sets
split_size = int(len(X) * 0.8)
X_train, y_train = X[:split_size], y[:split_size]
X_test, y_test = X[split_size:], y[split_size:]

# 1. Turn train and test arrays into tensor Datasets
train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

# 2. Combine features & labels
train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

# 3. Batch and prefetch for optimal performance
BATCH_SIZE = 1024 # taken from Appendix D in N-BEATS paper
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
# # Values from N-BEATS paper Figure 1 and Table 18/Appendix D
# N_EPOCHS = 5000 # called "Iterations" in Table 18
# N_NEURONS = 512 # called "Width" in Table 18
# N_LAYERS = 4
# N_STACKS = 30
#
# INPUT_SIZE = WINDOW_SIZE * HORIZON # called "Lookback" in Table 18
# THETA_SIZE = INPUT_SIZE + HORIZON
#
# tf.random.set_seed(42)
#
# # 1. Setup N-BEATS Block layer
# nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
#                                  theta_size=THETA_SIZE,
#                                  horizon=HORIZON,
#                                  n_neurons=N_NEURONS,
#                                  n_layers=N_LAYERS,
#                                  name="InitialBlock")
#
# # 2. Create input to stacks
# stack_input = Input(shape=(INPUT_SIZE), name="stack_input")
#
# # 3. Create initial backcast and forecast input (backwards predictions are referred to as residuals in the paper)
# backcast, forecast = nbeats_block_layer(stack_input)
# # Add in subtraction residual link, thank you to: https://github.com/mrdbourke/tensorflow-deep-learning/discussions/174
# residuals = subtract([stack_input, backcast], name=f"subtract_00")
#
# # 4. Create stacks of blocks
# for i, _ in enumerate(range(N_STACKS-1)): # first stack is already creted in (3)
#     # 5. Use the NBeatsBlock to calculate the backcast as well as block forecast
#     backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
#                                           theta_size=THETA_SIZE,
#                                           horizon=HORIZON,
#                                           n_neurons=N_NEURONS,
#                                           n_layers=N_LAYERS,
#                                           name=f"NBeatsBlock_{i}"
#     )(residuals) # pass it in residuals (the backcast)
#
#     # 6. Create the double residual stacking
#     residuals = subtract([residuals, backcast], name=f"subtract_{i}")
#     forecast = add([forecast, block_forecast], name=f"add_{i}")
#
# # 7. Put the stack model together
# model_7 = tf.keras.Model(inputs=stack_input,
#                          outputs=forecast,
#                          name="model_7_N-BEATS")
#
# # 8. Compile with MAE loss and Adam optimizer
# model_7.compile(loss="mae",
#                 optimizer=tf.keras.optimizers.Adam(0.001),
#                 metrics=["mae", "mse"])
#
# # 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
# model_7.fit(train_dataset,
#             epochs=N_EPOCHS,
#             validation_data=test_dataset,
#             verbose=0, # prevent large amounts of training outputs
#             # callbacks=[create_model_checkpoint(model_name=stack_model.name)] # saving model every epoch consumes far too much time
#             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
#                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])
#
# model_7_preds = make_preds(model_7, test_dataset)
# model_7_results = evaluate_preds(y_test, model_7_preds)
# endregion

# region Model 8: Ensemble model (stacking different models together)
# 用不同的loss function创建多个模型
def get_ensemble_models(horizon=HORIZON,
                        train_data=train_dataset,
                        test_data=test_dataset,
                        num_iter=10,
                        num_epochs=100,
                        loss_fns=["mae", "mse", "mape"]):
    """
    Returns a list of num_iter models each trained on MAE, MSE and MAPE loss.

    For example, if num_iter=10, a list of 30 trained models will be returned:
    10 * len(["mae", "mse", "mape"]).
    """
    # Make empty list for trained ensemble models
    ensemble_models = []

    # Create num_iter number of models per loss function
    for i in range(num_iter):
        # Build and fit a new model with a different loss function
        for loss_function in loss_fns:
            print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}")

            # Construct a simple model (similar to model_1)
            model = tf.keras.Sequential([
                # Initialize layers with normal (Gaussian) distribution so we can use the models for prediction
                # interval estimation later: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
                Dense(128, kernel_initializer="he_normal", activation="relu"),
                Dense(128, kernel_initializer="he_normal", activation="relu"),
                Dense(HORIZON)
            ])

            # Compile simple model with current loss function
            model.compile(loss=loss_function,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae", "mse"])

            # Fit model
            model.fit(train_data,
                      epochs=num_epochs,
                      verbose=0,
                      validation_data=test_data,
                      # Add callbacks to prevent training from going/stalling for too long
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                  patience=200,
                                                                  restore_best_weights=True),
                                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                      patience=100,
                                                                      verbose=1)])

            # Append fitted model to list of ensemble models
            ensemble_models.append(model)

    return ensemble_models  # return list of trained models

# Create a function which uses a list of trained models to make and return a list of predictions
def make_ensemble_preds(ensemble_models, data):
    ensemble_preds = []

    for model in ensemble_models:
        preds = model.predict(data) # make predictions with current ensemble model
        ensemble_preds.append(preds)

    return tf.constant(tf.squeeze(ensemble_preds))

ensemble_models = get_ensemble_models(num_epochs=1000,
                                      num_iter=5)
ensemble_preds = make_ensemble_preds(ensemble_models,
                                     test_dataset)
ensemble_results = evaluate_preds(y_true=y_test,
                                  y_pred=np.median(ensemble_preds, axis=0)) # take the median across all ensemble predictions
# endregion

# region Model 9: Train a model on the full historical data
# Train model on entire data to make prediction for the next day
X_all = bitcoin_prices_windowed.drop(["Price", "block_reward"], axis=1).dropna().to_numpy()
y_all = bitcoin_prices_windowed.dropna()["Price"].to_numpy()

# 1. Turn X and y into tensor Datasets
features_dataset_all = tf.data.Dataset.from_tensor_slices(X_all)
labels_dataset_all = tf.data.Dataset.from_tensor_slices(y_all)

# 2. Combine features & labels
dataset_all = tf.data.Dataset.zip((features_dataset_all, labels_dataset_all))

# 3. Batch and prefetch for optimal performance
BATCH_SIZE = 1024  # taken from Appendix D in N-BEATS paper
dataset_all = dataset_all.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

tf.random.set_seed(42)

# Create model (nice and simple, just to test)
model_9 = tf.keras.Sequential([
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(HORIZON)
])

# Compile
model_9.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam())

# Fit model on all of the data to make future forecasts
model_9.fit(dataset_all,
            epochs=100,
            verbose=0)  # don't print out anything, we've seen this all before


def make_future_forecast(values, model, into_future, window_size=WINDOW_SIZE) -> list:
    """
    Makes future forecasts into_future steps after values ends.

    Returns future forecasts as list of floats.
    """
    # 2. Make an empty list for future forecasts/prepare data to forecast on
    future_forecast = []
    last_window = values[-window_size:]  # only want preds from the last window (this will get updated)

    # 3. Make INTO_FUTURE number of predictions, altering the data which gets predicted on each time
    for _ in range(into_future):
        # Predict on last window then append it again, again, again (model starts to make forecasts on its own forecasts)
        future_pred = model.predict(tf.expand_dims(last_window, axis=0))
        print(f"Predicting on: \n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")

        # Append predictions to future_forecast
        future_forecast.append(tf.squeeze(future_pred).numpy())
        # print(future_forecast)

        # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
        last_window = np.append(last_window, future_pred)[-window_size:]

    return future_forecast

def get_future_dates(start_date, into_future, offset=1):
    """
    Returns array of datetime values from ranging from start_date to start_date+horizon.

    start_date: date to start range (np.datetime64)
    into_future: number of days to add onto start date for range (int)
    offset: number of days to offset start_date by (default 1)
    """
    start_date = start_date + np.timedelta64(offset, "D")  # specify start date, "D" stands for day
    end_date = start_date + np.timedelta64(into_future, "D")  # specify end date
    return np.arange(start_date, end_date, dtype="datetime64[D]")  # return a date range between start date and end date


INTO_FUTURE = 14
future_forecast = make_future_forecast(values=y_all,
                                       model=model_9,
                                       into_future=INTO_FUTURE,
                                       window_size=WINDOW_SIZE)
last_timestep = bitcoin_prices.index[-1]
next_time_steps = get_future_dates(start_date=last_timestep,
                                   into_future=INTO_FUTURE)
# endregion
