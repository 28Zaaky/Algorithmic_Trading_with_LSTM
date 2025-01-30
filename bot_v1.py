# Ce programme sert de simulation. 

import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import time
import os

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
limit = 200 

def fetch_live_data(symbol, timeframe, limit=60):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(ohlcv, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Erreur lors de la récupération des données en temps réel : {e}")
        return None

# Calcul du RSI
def calculate_rsi(data, timeperiod=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=timeperiod).mean()
    avg_loss = loss.rolling(window=timeperiod).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_live_data(df, scaler):
    df['RSI'] = calculate_rsi(df['close'], timeperiod=14).fillna(50)
    df['SMA_20'] = df['close'].rolling(window=20).mean().bfill() 
    df['SMA_50'] = df['close'].rolling(window=50).mean().bfill()  
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean().bfill()  
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean().bfill()  
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean().fillna(0)

    df_normalized = scaler.transform(df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50',
                                          'EMA_12', 'EMA_26', 'MACD', 'MACD_signal']])
    return df_normalized

def make_prediction(model, live_data, window_size):
    if len(live_data) >= window_size:
        input_sequence = np.array(live_data[-window_size:]).reshape(1, window_size, live_data.shape[1])
        predicted_price = model.predict(input_sequence)
        return predicted_price[0][0]
    else:
        print("Pas assez de données pour une prédiction.")
        return None

class SimulatedTrader:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.position = 0
        self.trade_log = []

    def execute_trade(self, prediction, current_price):
        if prediction > current_price * 1.01:
            if self.balance > 0:
                self.position = self.balance / current_price
                self.balance = 0
                self.trade_log.append(f"Achat à {current_price:.2f} USD")
        elif prediction < current_price * 0.99: 
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
                self.trade_log.append(f"Vente à {current_price:.2f} USD")

    def get_status(self, current_price):
        total_value = self.balance + (self.position * current_price)
        return f"Valeur totale : {total_value:.2f} USD (Balance : {self.balance:.2f}, Position : {self.position:.4f} BTC)"


# EXEMPLE !!

model = load_model("30_01_25_model.h5")

scaler = MinMaxScaler()

scaler.fit(np.random.rand(200, 12))  

trader = SimulatedTrader()
window_size = 60

print("Début de la simulation en temps réel...")
while True:
    live_data = fetch_live_data(symbol, timeframe)
    if live_data is not None:
        live_data_preprocessed = preprocess_live_data(live_data, scaler)
        predicted_price = make_prediction(model, live_data_preprocessed, window_size)
        if predicted_price:
            current_price = live_data['close'].iloc[-1]
            print(f"Prix actuel : {current_price:.2f} USD, Prédiction : {predicted_price:.2f} USD")
            trader.execute_trade(predicted_price, current_price)
            print(trader.get_status(current_price))

            print(f"Prédiction brute du modèle : {predicted_price}")
            print(f"Données normalisées pour la prédiction : {live_data_preprocessed}")
    time.sleep(60)
