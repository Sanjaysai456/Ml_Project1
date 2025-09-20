import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import pickle

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths to saved model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Preprocess the input features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 step:int,
                 type: str,
                 amount: float,
                 oldbalanceOrg: float,
                 newbalanceOrig: float,
                 oldbalanceDest: float,
                 newbalanceDest: float,
                 isFlaggedFraud: int):
        
        self.step=step
        self.type = type
        self.amount = amount
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig
        self.oldbalanceDest = oldbalanceDest
        self.newbalanceDest = newbalanceDest
        self.isFlaggedFraud = isFlaggedFraud

    def get_data_as_data_frame(self):
        try:
            # Convert the input features to a DataFrame
            custom_data_input_dict = {
                "step":[self.step],
                "type": [self.type],
                "amount": [self.amount],
                "oldbalanceOrg": [self.oldbalanceOrg],
                "newbalanceOrig": [self.newbalanceOrig],
                "oldbalanceDest": [self.oldbalanceDest],
                "newbalanceDest": [self.newbalanceDest],
                "isFlaggedFraud": [self.isFlaggedFraud],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
