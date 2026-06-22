# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# ticker = "AAPL"
# data = yf.download(
#     ticker,
#     start = "2021-01-01",
#     end = "2026-01-01"
# )
# #analysing data
# print(data.head())
# print(data.shape)
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
# print(data.columns)
# #making a copy of data
# df = data.copy()

# df["Target"] = df["Close"].shift(-1)
# print(df.head())
# print(df.tail())
# #drop values which are NA
# df = df.dropna()
# print(df.shape())
# print(df.isnull().sum())

# # Creating features and Target
# X = df[["Open", "High", "Low", "Volume"]]
# y = df["Target"]

# split_index = int(len(df) * 0.8)
# X_train = X[:split_index]
# X_test = X[split_index:]
# y_train = y[:split_index]
# y_test = y[split_index:]

# # We didn't use train_test_split because stock prices are 
# # time-series data. Randomly shuffling would leak future 
# # information into training and produce unrealistic results.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import joblib

# -------------------------
# 1. Download stock data
# -------------------------

ticker = "AAPL"

data = yf.download(
    ticker,
    start="2021-01-01",
    end="2026-01-01"
)

# Optional: flatten columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# -------------------------
# 2. Create working dataframe
# -------------------------

df = data.copy()

# -------------------------
# 3. Create target
# Tomorrow's close price
# -------------------------

df["Target"] = df["Close"].shift(-1)

# Remove last NaN row
df = df.dropna()

# -------------------------
# 4. Create features and target
# -------------------------

X = df[["Open", "High", "Low", "Volume"]]

y = df["Target"]

# -------------------------
# 5. Time-series split
# -------------------------

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# -------------------------
# 6. Train model
# -------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -------------------------
# 7. Predictions
# -------------------------

predictions = model.predict(X_test)

# -------------------------
# 8. Evaluate
# -------------------------

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("MAE:", mae)

print("R2:", r2)

# -------------------------
# 9. Actual vs Predicted plot
# -------------------------

plt.figure(figsize=(12, 6))

plt.plot(y_test.values, label="Actual")

plt.plot(predictions, label="Predicted")

plt.xlabel("Days")

plt.ylabel("Closing Price")

plt.title("Actual vs Predicted Closing Prices")

plt.legend()

plt.show()

# -------------------------
# 10. Results table
# -------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})

print(results.head(10))

# -------------------------
# 11. Save model
# -------------------------

joblib.dump(model, "stock_model.pkl")

print("Model saved successfully.")