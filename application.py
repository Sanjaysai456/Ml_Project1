from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Create CustomData object from form inputs
        data = CustomData(
            step=int(request.form.get('step',0)),
            type=request.form.get('type'),
            amount=float(request.form.get('amount')),
            oldbalanceOrg=float(request.form.get('oldbalanceOrg')),
            newbalanceOrig=float(request.form.get('newbalanceOrig')),
            oldbalanceDest=float(request.form.get('oldbalanceDest')),
            newbalanceDest=float(request.form.get('newbalanceDest')),
            isFlaggedFraud=int(request.form.get('isFlaggedFraud', 0))  # default 0 if not provided
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:\n", pred_df)
        print("Before Prediction")

        # Make prediction
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Pass the result to the template
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
