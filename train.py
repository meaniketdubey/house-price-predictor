import pandas as pd
import joblib

from app.models.linear_regression import LinearRegression


df  = pd.read_csv("data/raw/house_data.csv")

print(df.head())

X = df[["size","bedrooms"]].values
y = df["price"].values


model = LinearRegression(learning_rate=0.00000001, iterations= 1000)

model.fit(X,y)

joblib.dump(model,"saved_models/linear_regression.pkl")

print("Model trained successfully!")