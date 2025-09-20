# CORRECTED data_transformation.py

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X_features: pd.DataFrame):
        """
        This function is responsible for data transformation
        """
        try:
            # No longer need to drop columns here, as we pass in X_train
            cat_cols = X_features.select_dtypes(include="object").columns
            num_cols = X_features.select_dtypes(exclude="object").columns

            numeric_pipeline = Pipeline(
                steps=[("scaler", StandardScaler(with_mean=False))]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numeric_pipeline, num_cols),
                    ("cat_pipelines", categorical_pipeline, cat_cols),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            target_column_name = "isFraud"
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_column_name, "nameOrig", "nameDest"], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name, "nameOrig", "nameDest"], axis=1)
            y_test = test_df[target_column_name]

            # Get preprocessing object based on X_train features
            preprocessing_obj = self.get_data_transformer_object(X_train)
            
            # Apply transformation
            X_train_processed = preprocessing_obj.fit_transform(X_train)
            X_test_processed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_processed, np.array(y_train)]
            test_arr = np.c_[X_test_processed, np.array(y_test)]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)