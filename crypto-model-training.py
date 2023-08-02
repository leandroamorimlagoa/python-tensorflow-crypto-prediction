from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time

from Repository.CandlestickRepository import CandlestickRepository
from Repository.MarketRepository import MarketRepository

# Símbolo da criptomoeda
symbol = "USDT"
hoursInTheFurure = 24
today = datetime.today()

def train_model(symbol):
    seq_length = 48

    # Load candlestick data from repository
    candlestick_repo = CandlestickRepository()
    candlestick_data = candlestick_repo.get_candlestick_by_market_symbol(symbol)

    # Remover o último registro, pois ele não possui o valor de fechamento real
    candlestick_data = candlestick_data[:-1]

    # Converter para DataFrame
    df = pd.DataFrame(
        candlestick_data,
        columns=[
            "Id",
            "MarketSymbol",
            "PriceOpen",
            "PriceHighest",
            "PriceLowest",
            "PriceClose",
            "Volume",
            "DateTime",
            "DayWeek",
        ],
    )

    # Selecionar as colunas de interesse
    selected_columns = ["PriceHighest","PriceClose", "Volume", "DateTime", "DayWeek"]
    df = df[selected_columns]

    # Converter a coluna DateTime para o tipo datetime
    last_date = df["DateTime"].iloc[-1]
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["Year"] = df["DateTime"].dt.year
    df["Hour"] = df["DateTime"].dt.hour
    df["Minute"] = df["DateTime"].dt.minute

    df.drop(columns=["DateTime"], axis=1, inplace=True)

    df["PriceHighest"] = pd.to_numeric(df["PriceHighest"], errors="coerce")
    df["PriceClose"] = pd.to_numeric(df["PriceClose"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    day = []
    month = []
    year = []
    hour = []
    minute = []
    price_high = []
    price_close = []
    volume = []
    day_week = []
    y = []

    for i in range(0, df.shape[0] - seq_length):
        day.append(df.iloc[i : i + seq_length]["Day"])
        month.append(df.iloc[i : i + seq_length]["Month"])
        year.append(df.iloc[i : i + seq_length]["Year"])
        hour.append(df.iloc[i : i + seq_length]["Hour"])
        minute.append(df.iloc[i : i + seq_length]["Minute"])
        price_high.append(df.iloc[i : i + seq_length]["PriceHighest"])
        price_close.append(df.iloc[i : i + seq_length]["PriceClose"])
        volume.append(df.iloc[i : i + seq_length]["Volume"])
        day_week.append(df.iloc[i : i + seq_length]["DayWeek"])
        y.append(df.iloc[i + seq_length]["PriceClose"])

    day, month, year, hour, minute, price_high, price_close, volume, day_week, y = (
        np.array(day),
        np.array(month),
        np.array(year),
        np.array(hour),
        np.array(minute),
        np.array(price_high),
        np.array(price_close),
        np.array(volume),
        np.array(day_week),
        np.array(y),
    )

    y = np.reshape(y, (len(y), 1))

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    day = scaler.fit_transform(day)
    month = scaler.fit_transform(month)
    year = scaler.fit_transform(year)
    hour = scaler.fit_transform(hour)
    minute = scaler.fit_transform(minute)
    price_high = scaler.fit_transform(price_high)
    price_close = scaler.fit_transform(price_close)
    volume = scaler.fit_transform(volume)
    day_week = scaler.fit_transform(day_week)
    y = scaler.fit_transform(y)

    X = np.stack((day, month, year, hour, minute, price_high, price_close, volume, day_week), axis=2)

    X_train, X_test = X[:-480], X[-480:]
    y_train, y_test = y[:-480], y[-480:]

    # Criar o modelo
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))

    # Compilar o modelo
    modelFile = f"models\crypto-model-{symbol}.h5"
    callbacks = [EarlyStopping(monitor="val_loss", patience=20), 
                ModelCheckpoint(filepath=modelFile, monitor="val_loss", save_best_only=True, mode="min")]

    keras.optimizers.SGD(momentum=0.9) # ???????????????????
    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, validation_split=0.2, epochs=300, callbacks=callbacks, batch_size=16)

    MSE, MAE = model.evaluate(X_test, y_test)
    print(MSE, MAE)

    # predictions = model.predict(X_test)
    # predictions = scaler.inverse_transform(predictions)
    # print("Predictions")
    # print("=====================================")
    # print(predictions)
    # # predictions

    # # Plotar o gráfico
    # plt.style.use('fivethirtyeight')
    # plt.figure(figsize=(24,12))
    # plt.title('30 Minutes ahead Price Forecasting')
    # plt.xlabel('Step (30 Minutes)', fontsize=14)
    # plt.ylabel('Crypto Price (AUD)', fontsize=18)
    # plt.plot(predictions)
    # plt.plot(scaler.inverse_transform(y_test))
    # plt.legend(['Forecast', 'Actual'])
    # plt.show()
    

market_repo = MarketRepository()
market_data = market_repo.get_all_markets()
df = pd.DataFrame(
        market_data,
        columns=[
            "Id",
            "Symbol",
            "QuantityMin",
            "QuantityIncrement",
            "PriceMin",
            "PriceIncrement",
            "CurrencyBaseId",
            "CurrencyQuoteId",
            "CurrencyBaseSymbol",
            "CurrencyQuoteSymbol",
        ],
    )

crypto_list = 'grt', 'htr', 'ilv', 'imx', 'knc', 'looks', 'lpt', 'lrc', 'mana', 'mco2', 'mkr', 'nexo', 'okb', 'omg', 'qnt', 'ren', 'rsr', 'sand', 'skl', 'slp', 'snx', 'storj', 'sushi', 'sxp', 'tusd', 'uma', 'xtz', 'yfi', 'ygg', 'zrx'
# errors: 'gala2', 
# 'ape', 'mana', 'sand', 'ftm', 'xtz', 'axs', 'aave', 'eos', 'grt', 'mkr', 'tusd', 'nexo', 'chz', 'qnt', 'gala', 'okb', 'enj', 'crv', 'bat', 'comp', 'knc', 'amp', 'zrx', 'audio', 'yfi', 'snx', 'omg', 'sxp', 'storj', 'sushi', 'ilv', 'dydx', 'ygg', 'alice', 'alpha', 'htr', 'slp', 'efi', 'looks', 'abfy', 'mco2', 'cofbr', 'lrc', '1inch', 'rsr', 'lpt', 'bal', 'imx', 'uma', 'skl', 'cvc', 'dao', 'ren', 'celr', 'furea', 'fmap', 'fkcl', 'bnb', 'ftpc34645', 'cake', 'ftpc35782', 'atom', 'ftpcl0053', 'buddha', 'fxmusics01', 'ftcec01', 'gala2'
crypto_list = sorted(crypto_list)

for symbol in crypto_list:
    print(symbol)
    train_model(symbol)
# # train_model('btc', daysBefore)