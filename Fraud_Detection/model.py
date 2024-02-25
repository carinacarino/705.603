from flask import Flask, request, jsonify, current_app
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib  # For model persistence
import os  # For checking file existence

from data_pipeline import ETL_Pipeline
from dataset import Fraud_Dataset
from metrics import Metrics

app = Flask(__name__)


class Fraud_Detector_Model:
    def __init__(self, dataset_path=None, model_path='model.pkl', scaler_path='scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.dataset_path = dataset_path
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.scaler = StandardScaler()
        self.etl_pipeline = ETL_Pipeline()
        self.metrics = Metrics()
        # Ensure the model is loaded or trained right after initialization
        self.load_and_train_model()

    def load_and_train_model(self):
        """Loads a pre-trained model or trains a new one."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            print("Trained model, scaler, and encoder states exist on the directory.")
            # Load the pre-trained model and scaler
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

            self.etl_pipeline.load_state('etl_pipeline_state')

            print("Loaded trained model, scaler and encoder states.")

        elif self.dataset_path:
            # Load dataset
            print("Preprocessing...")
            original_data = self.etl_pipeline.extract('transactions.csv')
            # Transform the original dataset
            processed_og = self.etl_pipeline.transform(original_data, training=True)
            print("Transforming complete. Now Splitting data using Fraud_Dataset class.")

            # Initialize Fraud_Dataset with the loaded dataset to split intro train and test
            fraud_dataset = Fraud_Dataset(processed_og, test_size=0.2, random_state=42)
            # Get training and testing data
            X_train, y_train = fraud_dataset.get_training_data()
            X_test, y_test = fraud_dataset.get_testing_data()

            print("Splitting  complete. Now performing scaling transformations.")
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)  # For later evaluation

            print('Preprocessing done.')
            print('Training Random forest model. This process might take a while.')
            # Train the model with scaled training data
            self.train(X_train_scaled, y_train)
            self.etl_pipeline.save_state('etl_pipeline_state')

            print('Testing...')
            self.test(X_test_scaled, y_test)

            # Save the trained model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print("Trained and saved the model, scaler, and encoder states.")
            print("Model evaluated. Metrics saved in /results")
        else:
            raise FileNotFoundError("No pre-trained model found, and no dataset path provided for training.")

    def train(self, X_train, y_train):
        """Trains the XGBoost model."""
        self.model.fit(X_train, y_train)
        print("Model training completed.")

    def test(self, X_test, y_test):
        """Evaluates the model on the test set."""
        y_pred = self.model.predict(X_test)
        self.metrics.generate_report(y_pred, y_test)


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.get_json(force=True)

    # Ensure the model and scaler are loaded
    if not hasattr(fdm, 'model') or not hasattr(fdm, 'scaler'):
        return jsonify({"error": "Model or scaler not loaded properly."}), 500

    # Process the input data through the ETL pipeline and scale it
    input_data_df = pd.DataFrame([input_data])
    processed_data = fdm.etl_pipeline.transform(input_data_df, training=False)
    processed_data_scaled = fdm.scaler.transform(processed_data)
    current_app.logger.info(processed_data.columns)
    # Make a prediction
    prediction = fdm.model.predict(processed_data_scaled)
    prediction_result = "Fraud" if prediction[0] == 1 else "Not Fraud"

    # Return the prediction result
    return jsonify({"prediction": prediction_result})


if __name__ == '__main__':
    flaskPort = 8786
    fdm = Fraud_Detector_Model(dataset_path='transactions.csv')
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)
