# CORRECTED model_trainer.py

import os
import sys
from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info(f"Before SMOTE, class distribution: {np.bincount(y_train.astype(int))}")

            # Apply SMOTE to handle imbalance
            smote = SMOTE(sampling_strategy=0.05, random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            logging.info(f"After SMOTE, class distribution: {np.bincount(y_res.astype(int))}")

            # Train RandomForest
            clf = RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            )
            clf.fit(X_res, y_res)

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=clf
            )
            logging.info("Model saved successfully.")

            # Predictions
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            # Evaluation
            report = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_prob)

            logging.info(f"Classification Report: {report}")
            logging.info(f"ROC AUC Score: {roc_auc}")

            return {"classification_report": report, "roc_auc": roc_auc}

        except Exception as e:
            raise CustomException(e, sys)