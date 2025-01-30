import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import optuna
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from tensorflow.keras.layers import Input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'rateLimit': 1200,
    'enableRateLimit': True
})

symbol = 'BTC/USDT'
timeframe = '1m'
limit = 2000

def get_data():
    try:
        logging.info("Récupération des données depuis Binance...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(ohlcv, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df['RSI'] = calculate_rsi(df['close'], timeperiod=14).fillna(50)
        df['SMA_20'] = df['close'].rolling(window=20).mean().bfill()
        df['SMA_50'] = df['close'].rolling(window=50).mean().bfill()
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean().bfill()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean().bfill()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean().fillna(0)

        if df.isnull().values.any():
            logging.warning("Les données contiennent des valeurs manquantes, suppression en cours.")
            df = df.dropna()

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50',
                                               'EMA_12', 'EMA_26', 'MACD', 'MACD_signal']])
        df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD',
            'MACD_signal']] = scaled_data

        return df, scaler
    except ccxt.BaseError as e:
        logging.error(f"Erreur API : {e}")
        return None, None

def calculate_rsi(data, timeperiod=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=timeperiod).mean()
    avg_loss = loss.rolling(window=timeperiod).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df, scaler = get_data()

def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        labels.append(data[i + window_size, 3])
    return np.array(sequences), np.array(labels)

window_size = 60
data = df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD',
           'MACD_signal']].values
sequences, labels = create_sequences(data, window_size)

train_size = int(0.8 * len(sequences))
X_train, X_test = sequences[:train_size], sequences[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

assert X_train.shape[1:] == (window_size, data.shape[1]), "Les dimensions des séquences ne correspondent pas."

def load_saved_model(filename="model.h5"):
    return load_model(filename) if os.path.exists(filename) else None


model = load_saved_model()

if model is None:
    def objective(trial):
        num_units = trial.suggest_int("num_units", 64, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate) if optimizer_name == "rmsprop" else tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Bidirectional(
                LSTM(units=num_units, return_sequences=True, kernel_initializer="he_normal",
                     input_shape=(X_train.shape[1], X_train.shape[2]))),
            Dropout(dropout_rate),
            LSTM(units=num_units),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        return mae


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_trial.params

    model = Sequential([
        Bidirectional(LSTM(units=best_params["num_units"], return_sequences=True,
                           input_shape=(X_train.shape[1], X_train.shape[2]))),
        Dropout(best_params["dropout_rate"]),
        LSTM(units=best_params["num_units"]),
        Dropout(best_params["dropout_rate"]),
        Dense(1)
    ])
    model.compile(optimizer=best_params["optimizer"], loss="mean_squared_error", metrics=["mae"])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    model.save("30_01_25_model.h5")

early_stopping = EarlyStopping(monitor="val_loss", patience=5)
model_checkpoint = ModelCheckpoint("best_model2.keras", save_best_only=True, monitor="val_loss", mode="min")

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[early_stopping, model_checkpoint])

model.save("30_01_25_model.h5")

y_pred = model.predict(X_test)
y_test_real = scaler.inverse_transform(np.column_stack((np.zeros((y_test.shape[0], 11)), y_test)))[:, -1]
y_pred_real = scaler.inverse_transform(np.column_stack((np.zeros((y_pred.shape[0], 11)), y_pred)))[:, -1]

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print(f"Erreur moyenne absolue (MAE): {mae}")
print(f"Erreur quadratique moyenne (RMSE): {rmse}")
print(f"Erreur pourcentage absolue moyenne (MAPE): {mape}%")

plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label="Prix réel", color="blue")
plt.plot(y_pred_real, label="Prédiction", color="red")
plt.title("Comparaison des prix réels et prédits (BTC/USDT)")
plt.xlabel("Temps")
plt.ylabel("Prix")
plt.legend()
plt.grid()
plt.savefig("prediction_vs_real.png")
plt.show()

with open("model_performance.txt", "w") as f:
    f.write(f"MAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}%\n")
