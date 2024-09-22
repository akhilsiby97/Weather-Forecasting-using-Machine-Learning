import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import seaborn as sns

class WeatherForecastModel:
    def __init__(self, data_path, n_steps=10, epochs=50):
        self.data_path = data_path
        self.n_steps = n_steps
        self.epochs = epochs
        self.scaler_max = MinMaxScaler()
        self.scaler_min = MinMaxScaler()
        self.data = self.load_and_prepare_data()

    def load_and_prepare_data(self):
        data = pd.read_excel(self.data_path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data['County'].fillna('Unknown', inplace=True)
        data['Station'].fillna('Unknown', inplace=True)
        data['g_rad'] = pd.to_numeric(data['g_rad'], errors='coerce')
        data['soil'] = pd.to_numeric(data['soil'], errors='coerce')
        return data

    def create_sequences(self, data):
        sequences = []
        labels = []
        for i in range(len(data) - self.n_steps):
            seq = data[i:i + self.n_steps]
            label = data[i + self.n_steps]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(self.n_steps, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, data_scaled):
        X, y = self.create_sequences(data_scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = self.build_model()
        model.fit(X, y, epochs=self.epochs, verbose=0)
        return model

    def forecast(self, model, data, n_forecast):
        predictions = []
        current_seq = data[-self.n_steps:]
        for _ in range(n_forecast):
            pred = model.predict(current_seq.reshape(1, self.n_steps, 1), verbose=0)
            predictions.append(pred[0, 0])
            current_seq = np.append(current_seq[1:], pred, axis=0)
        return np.array(predictions)

    def get_forecast(self, model_max, model_min, data_scaled_max, data_scaled_min, n_forecast):
        forecast_max = self.forecast(model_max, data_scaled_max, n_forecast)
        forecast_min = self.forecast(model_min, data_scaled_min, n_forecast)

        # Properly inverse scale the predictions
        forecast_max = self.scaler_max.inverse_transform(forecast_max.reshape(-1, 1)).flatten()
        forecast_min = self.scaler_min.inverse_transform(forecast_min.reshape(-1, 1)).flatten()

        return forecast_max, forecast_min

    def train_and_forecast(self, county, station, start_date, end_date):
        county_data = self.data[self.data['County'] == county]
        station_data = county_data[county_data['Station'] == station]

        max_temp_data = station_data['maxtp'].dropna().values
        min_temp_data = station_data['mintp'].dropna().values

        # Fit scalers only on the training data
        data_scaled_max = self.scaler_max.fit_transform(max_temp_data.reshape(-1, 1))
        data_scaled_min = self.scaler_min.fit_transform(min_temp_data.reshape(-1, 1))

        model_max = self.train_model(data_scaled_max)
        model_min = self.train_model(data_scaled_min)

        # Calculate the number of forecast days based on the date range
        n_forecast = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

        forecast_max, forecast_min = self.get_forecast(model_max, model_min, data_scaled_max, data_scaled_min, n_forecast)
        
        forecast_dates = pd.date_range(start=start_date, periods=n_forecast)

        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Max Temperature Forecast': forecast_max,
            'Min Temperature Forecast': forecast_min
        })

        return forecast_df
