# Multiple Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
        self.ordinal_encoder = None
        self.onehot_encoder_transmission = None
        self.onehot_encoder_color = None
        self.scaler = None
        self.model = None

    def model_learn(self):
        file_path = 'cars.csv'
        df = pd.read_csv(file_path)
        
        X = df[['transmission', 'color', 'odometer_value', 'year_produced', 'body_type', 'price_usd']]
        y = df['duration_listed']
        
        # Ordinal Encoding for 'body_type'
        body_type_order = ['universal', 'hatchback', 'cabriolet', 'coupe', 'sedan', 'liftback', 'suv', 'minivan', 'van', 'pickup', 'minibus', 'limousine']
        self.ordinal_encoder = OrdinalEncoder(categories=[body_type_order])
        X.loc[:, 'body_type_encoded'] = self.ordinal_encoder.fit_transform(X[['body_type']])

        
        # OneHot Encoding for 'transmission'
        self.onehot_encoder_transmission = OneHotEncoder(sparse=False, drop='first')
        transmission_encoded = self.onehot_encoder_transmission.fit_transform(X[['transmission']])
        transmission_df = pd.DataFrame(transmission_encoded, columns=self.onehot_encoder_transmission.get_feature_names_out(['transmission']))
        
        # OneHot Encoding for 'color'
        self.onehot_encoder_color = OneHotEncoder(sparse=False, drop='first')
        color_encoded = self.onehot_encoder_color.fit_transform(X[['color']])
        color_df = pd.DataFrame(color_encoded, columns=self.onehot_encoder_color.get_feature_names_out(['color']))
        
        # Combining all features
        X.drop(['transmission', 'color', 'body_type'], axis=1, inplace=True)
        
        X_encoded = pd.concat([X, transmission_df, color_df], axis=1)
        
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)
        
        
        
        # Feature Scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)  # Correct use of transform for the test set
        

        # Model Training
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)  # Use scaled training data
        
        # Model Evaluation
        y_pred = self.model.predict(X_test_scaled)
        self.stats = r2_score(y_test, y_pred)
        
        self.modelLearn = True
      
    def model_infer(self, transmission, color, odometer, year, bodytype, price):
        if not self.modelLearn:
            print("Model has not been trained. Please train the model first.")
            return

        # Preprocess inputs
        try:
            bodytype_encoded = self.ordinal_encoder.transform([[bodytype]])
        except ValueError as e:
            print(f"Error: {e}. Unknown category '{bodytype}' found for 'bodytype'.")
            return

        transmission_encoded = self.onehot_encoder_transmission.transform([[transmission]])
        color_encoded = self.onehot_encoder_color.transform([[color]])

        features = np.hstack([bodytype_encoded, transmission_encoded, color_encoded, np.array([[odometer, year, price]])])
        features_scaled = self.scaler.transform(features)

        # Predict the outcome
        y_pred = self.model.predict(features_scaled)
        return str(y_pred[0])
        
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)
