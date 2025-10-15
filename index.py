import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

# ======================
# Baca data
# ======================
df = pd.read_csv("portfolio_data.csv")

# Ubah tanggal agar mulai 2020
n = len(df)
df['Date'] = pd.date_range(start="2020-01-01", periods=n, freq="B")
df.set_index('Date', inplace=True)

# Pilih salah satu kolom, misalnya AMZN
series = df['AMZN']

# Cari parameter ARIMA terbaik
# ======================
model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
print("Best ARIMA order:", model.order)

# ======================
# Prediksi sampai akhir 2028
# ======================
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, end="2028-12-31", freq="B")
steps = len(future_dates)

forecast = model.predict(n_periods=steps)
forecast_index = future_dates

# Plot grafik

plt.figure(figsize=(14,7))

# Data aktual
plt.plot(series.index, series, label="Data Aktual (AMZN)", color="blue")

# Data prediksi
plt.plot(forecast_index, forecast, label="Prediksi sampai 2028", color="red")

plt.title("Prediksi Harga AMZN dengan ARIMA (2020–2028) [auto_arima]")
plt.xlabel("Tahun")
plt.ylabel("Harga AMZN")
plt.legend()
plt.grid(True)
plt.show()
