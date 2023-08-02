from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt

from Repository.CandlestickRepository import CandlestickRepository

# Substitua pela sua criptomoeda
symbol = "USDT"  # Exemplo de symbol
seq_length = 48

# Definir a data futura para a predição
datetime_future = pd.to_datetime('2023-05-22 12:00:00')  # Substitua pela sua data futura desejada

# Carregar o modelo treinado
modelFile = f"crypto-model-{symbol}.h5"
model = keras.models.load_model(modelFile)
# Carregar os dados do repositório CandlestickRepository
candlestick_repo = CandlestickRepository()
candlestick_data = candlestick_repo.get_candlestick_by_market_symbol(symbol)

# Remover o último registro, pois ele não possui o valor de fechamento real
candlestick_data = candlestick_data[:-1]

# Carregar o modelo treinado
model = load_model(f"trained-model-{symbol}.h5")
selected_columns = ["PriceHighest", "PriceClose", "Volume", "DayWeek", "DayMonth"]

# Preparar dados para predição
def prepare_data_for_prediction(data, seq_length):
    x_pred = []
    for i in range(len(data) - seq_length + 1):
        x_pred.append(data[i:i+seq_length])
    return np.array(x_pred)

# Carregar dados para predição
candlestick_data = candlestick_repo.get_candlestick_by_market_symbol(symbol)
df = pd.DataFrame(candlestick_data, columns=["Id", "MarketSymbol", "PriceOpen", "PriceHighest", "PriceLowest", "PriceClose", "Volume", "DateTime", "DayWeek", "DayMonth"])
data = df[selected_columns]
last_price_close = df["PriceClose"].iloc[-1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data)
scaled_data = scaler.transform(data)

# Preparar dados para predição
x_pred = prepare_data_for_prediction(scaled_data, seq_length)

# Fazer a predição
predicted_data = model.predict(x_pred)

# Desnormalizar os dados
predicted_data = scaler.inverse_transform(predicted_data)

lastprediction = predicted_data[-1,0]
max_predicted_value = np.max(predicted_data)

# print(predicted_data)
print("=====================================")
print(f"Last data: {last_price_close}")
print(f"Last prediction: {lastprediction}")
print(f"Highest predicted: {max_predicted_value}")



# Extrair os valores de PriceClose previstos
price_close_pred = predicted_data[:, 0]

# Criar uma data inicial com base na última data registrada nos dados
last_datetime = df.iloc[-1]["DateTime"]
start_datetime = datetime.strptime(str(last_datetime), "%Y-%m-%d %H:%M:%S")

# Definir a data e hora atual
current_datetime = datetime.now()

# Definir a data e hora final da predição
end_datetime = start_datetime + timedelta(hours=len(price_close_pred)/2)

# Criar uma lista de datas e horas para os índices
indices_datetime = [start_datetime + timedelta(minutes=i*30) for i in range(len(price_close_pred))]

# Limitar o índice para exibir somente a partir da data e hora atual até a última hora da predição
indices_datetime = [dt for dt in indices_datetime if current_datetime <= dt <= end_datetime]
price_close_pred = price_close_pred[:len(indices_datetime)]

# Plotar o gráfico de linha com os índices em formato de data e hora
plt.plot(indices_datetime, price_close_pred)
plt.xlabel('Data e Hora')
plt.ylabel('PriceClose Predito')
plt.title('Valores Previstos de PriceClose')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()