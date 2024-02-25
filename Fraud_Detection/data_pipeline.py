# data_pipeline.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import calendar
import joblib

class ETL_Pipeline:
    """
    Automates the extraction, transformation, and loading (ETL) process for data preprocessing in machine learning workflows.
    """
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_columns = None
        self.categorical_columns = ['sex', 'city_state', 'job', 'category', 'state']  # Add more if needed

    def extract(self, filepath):
        """Extracts data from a CSV file."""
        data = pd.read_csv(filepath, usecols=lambda column: column not in ['Unnamed: 0'])
        return data

    def transform(self, data, training=True):
        """Transforms the data for training or prediction."""
        data = self._transform_dates(data)
        data = self._calculate_age(data)
        data = self._extract_date_parts(data)
        data = self._calculate_distances(data)
        data = self._create_city_state_column(data)
        data = self._encode_categorical(data, training=training)
        data = self._one_hot_encode(data, training=training)
        data = self._drop_unnecessary_columns(data)

        # Check if it's the prediction phase and drop 'is_fraud' if present
        if not training and 'is_fraud' in data.columns:
            data = data.drop(columns=['is_fraud'])

        return data

    def _transform_dates(self, data):
        """Converts dates from string to datetime objects."""
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['trans_date'] = data['trans_date_trans_time'].dt.date
        data['trans_date'] = pd.to_datetime(data['trans_date'])
        data['trans_hour'] = data['trans_date_trans_time'].dt.hour
        data['trans_minute'] = data['trans_date_trans_time'].dt.minute
        return data

    def _calculate_age(self, data):
        """Calculates age in years."""
        data['age'] = (data['trans_date'] - data['dob']) / pd.Timedelta(days=365.25)
        data['age'] = data['age'].astype('int')
        return data

    def _extract_date_parts(self, data):
        """Extracts month and year from the transaction date."""
        data['trans_month'] = data['trans_date'].dt.month
        data['trans_year'] = data['trans_date'].dt.year
        data['month_name'] = data['trans_month'].apply(lambda x: calendar.month_abbr[x])
        return data

    def _calculate_distances(self, data):
        """Calculates latitudinal and longitudinal distances."""
        data['latitudinal_distance'] = abs(round(data['merch_lat'] - data['lat'], 3))
        data['longitudinal_distance'] = abs(round(data['merch_long'] - data['long'], 3))
        return data

    def _create_city_state_column(self, data):
        """Creates a 'city_state' column by concatenating 'city' and 'state' columns."""
        data['city_state'] = data['city'] + ', ' + data['state']
        return data

    def _encode_categorical(self, data, training=True):
        """Encodes categorical variables using label encoding."""
        for column in self.categorical_columns:
            if training and column in data:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                self.label_encoders[column] = le
            elif not training and column in data and column in self.label_encoders:
                le = self.label_encoders[column]
                data[column] = le.transform(data[column])
        return data

    def _one_hot_encode(self, data, training=True):
        if training:
            data_encoded = pd.get_dummies(data, columns=['category', 'state'], drop_first=True)
            # Save columns for prediction phase
            self.one_hot_columns = data_encoded.columns
        else:
            if self.one_hot_columns is None:
                # Initialize self.one_hot_columns if it's None
                # This is a fallback and might not be ideal; consider other solutions.
                self.one_hot_columns = data.columns
            data_encoded = pd.get_dummies(data, columns=['category', 'state'], drop_first=True)
            for col in set(self.one_hot_columns) - set(data_encoded.columns):
                data_encoded[col] = 0
            data_encoded = data_encoded.reindex(columns=self.one_hot_columns, fill_value=0)
        return data_encoded

    def _drop_unnecessary_columns(self, data):
        """Drops columns that are not needed for modeling."""
        columns_to_drop = ['cc_num','merchant','first','last','street','trans_num','trans_date_trans_time','city','lat',
                           'long','dob','merch_lat','merch_long','trans_date','month_name','city', 'job']
        return data.drop(columns=columns_to_drop, axis=1)

    def load(self, filename=None):
        """Loads the transformed data into a new CSV file and returns the transformed DataFrame."""
        if filename:
            self.data.to_csv(filename, index=False)
        return self.data

    def save_state(self, file_prefix):
        """Saves the state of the ETL pipeline to disk."""
        # Save the label encoders
        joblib.dump(self.label_encoders, f'{file_prefix}_label_encoders.pkl')
        # Save the one-hot columns
        joblib.dump(self.one_hot_columns, f'{file_prefix}_one_hot_columns.pkl')

    def load_state(self, file_prefix):
        """Loads the state of the ETL pipeline from disk."""
        # Load the label encoders
        self.label_encoders = joblib.load(f'{file_prefix}_label_encoders.pkl')
        # Load the one-hot columns
        self.one_hot_columns = joblib.load(f'{file_prefix}_one_hot_columns.pkl')
