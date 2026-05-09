import pandas as pd
import joblib

from app.models.linear_regression import LinearRegression


df  = pd.read_csv("data/raw/house_data.csv")

print(df.head())

# X = df[["size","bedrooms"]].values
# y = df["price"].values


model = LinearRegression(learning_rate=0.001, iterations= 10000)

# model.fit(X,y)

# joblib.dump(model,"saved_models/linear_regression.pkl")

# print("Model trained successfully!")
# print("Weights:", model.weights)
# print("Bias:", model.bias)


# # Test prediction
# sample_house = [[2800, 5]]

# predicted_price = model.predict(sample_house)

# print(f"Predicted Price: {predicted_price[0]}")





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


X = df[["size", "bedrooms"]].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X, y)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")