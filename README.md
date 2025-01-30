# Trading Bot LSTM pour BTC/USDT
## 🤖 Vue d'ensemble

Ce projet implémente un bot de trading qui prédit les prix du Bitcoin (BTC) en utilisant un réseau de neurones LSTM (Long Short-Term Memory). Le système collecte des données en temps réel depuis Binance, les prétraite, et utilise des indicateurs techniques comme le RSI, le MACD et les moyennes mobiles pour entraîner un modèle de prédiction des prix futurs.

## 📁 Structure du Projet

- `LSTM_2.py` : Script principal pour la construction et l'entraînement du modèle LSTM
  - Récupération des données historiques via Binance
  - Prétraitement des données
  - Entraînement du modèle avec optimisation des hyperparamètres via Optuna
  - Sauvegarde du modèle entraîné

- `bot_v1.py` : Bot de trading simulé
  - Utilise le modèle LSTM entraîné
  - Prédit le prochain prix BTC/USDT en temps réel
  - Simule des trades basés sur ces prédictions

## 🛠 Installation

### Dépendances

```bash
pip install ccxt pandas numpy tensorflow scikit-learn matplotlib optuna python-dotenv
```

### Configuration des clés API

1. Créez un fichier `.env` à la racine du projet
2. Ajoutez vos clés API Binance :
```
API_KEY=votre_clé_api
API_SECRET=votre_clé_secrète
```

## 📊 Structure des Dossiers

```
/TradingBotDeepLearning
├── .venv
├── LSTM_2.py
├── bot_v1.py
├── .env
├── /Model
│   └── model.h5
├── model_performance.txt
└── /Images
    └── prediction_vs_real.png
```

## 🚀 Utilisation

### Entraînement du Modèle

```bash
python LSTM_2.py
```

### Trading Simulé

```bash
python bot_v1.py
```

⚠️ Note : Le bot ne réalise pas de trades réels - il simule uniquement les transactions basées sur les prédictions du modèle LSTM.

## 📈 Métriques de Performance

Le modèle est évalué selon plusieurs métriques :
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

Les métriques sont sauvegardées dans `model_performance.txt` et les graphiques comparatifs sont générés dans le dossier `Images`.

## 🔄 Exemple de Sortie

```
Prix actuel : 106127.30 USD, Prédiction : 2.01 USD
Valeur totale : 1000.00 USD (Balance : 1000.00, Position : 0.0000 BTC)
Prédiction brute du modèle : 2.009071111679077
Données normalisées pour la prédiction : [[ 1.06250431e+05  1.05060124e+05  1.05542602e+05  1.05902704e+05
1.68641519e+01  5.02680800e+01  1.05286280e+05  1.06709744e+05
1.05712029e+05  1.05049347e+05 -4.16026237e-03 -6.28936785e-03]..........]
```

## 🔜 Améliorations Futures

- Implémentation du trading en direct avec transactions réelles
- Exploration d'autres modèles (GRU, Transformer)
- Optimisation de la stratégie basée sur le backtesting historique
