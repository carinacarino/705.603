from sklearn.model_selection import train_test_split, KFold

class Fraud_Dataset:
    def __init__(self, data, test_size=0.2, validation_size=None, random_state=42):
        """
        Initializes the Fraud_Dataset object. Split the data into training and testing datasets, and validation datasets.

        """
        self.data = data
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

        # Split the data into training and testing datasets
        self.train_data, self.test_data = train_test_split(data, test_size=test_size, random_state=random_state)

        # Further split the training data into actual training and validation datasets
        self.actual_train_data, self.validation_data = train_test_split(self.train_data, test_size=validation_size,
                                                                        random_state=random_state)

    def get_training_data(self):
        """

        Returns the training dataset.

        """
        X_train = self.actual_train_data.drop('is_fraud', axis=1)
        y_train = self.actual_train_data['is_fraud']
        return X_train, y_train

    def get_validation_data(self):
        """

        Returns the validation dataset.

        """
        X_val = self.validation_data.drop('is_fraud', axis=1)
        y_val = self.validation_data['is_fraud']
        return X_val, y_val

    def get_testing_data(self):
        """

        Returns the testing dataset.

        """

        X_test = self.test_data.drop('is_fraud', axis=1)
        y_test = self.test_data['is_fraud']
        return X_test, y_test

    def get_k_fold_data(self, k):
        """
        Generates indices for k-fold cross-validation.

        """
        kf = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
        return kf.split(self.data)
