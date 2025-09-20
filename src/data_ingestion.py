# CORRECTED data_ingestion.py

import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from src.logger import logging
from src.data_trasformation import DataTransformation
from src.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join("artifacts")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")
        try:
            # Recommended to change this path
            df = pd.read_csv(r'notebook\data\data.csv')
            logging.info('Read the dataset as a dataframe')

            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved.")

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise e

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_training = ModelTrainer()
    model_training.initiate_model_trainer(train_arr, test_arr)
    print("Model training is complete.")