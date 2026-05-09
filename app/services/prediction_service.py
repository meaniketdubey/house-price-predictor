import joblib
import numpy as np

from app.core.config import config
from app.core.logger import logger

class Predictservice:
     def __init__(self):
          model_path = config['model']['path']
          logger.info(f"loading model from: {model_path}")

          self.model = joblib.load("saved_models/linear_regression.pkl")
          logger.info(f"loading model from: {model_path}")

     def predict(self, size, bedrooms):
          logger.info(
            f"Prediction request received: "
            f"size={size}, bedrooms={bedrooms}"
            )
          features = np.array([[size, bedrooms]])
          predictions = self.model.predict(features)
          logger.info(
            f"Prediction generated: {predictions[0]}"
            )

          return predictions[0]