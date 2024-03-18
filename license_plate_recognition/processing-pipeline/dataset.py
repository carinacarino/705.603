import os
import random
import numpy as np
from sklearn.model_selection import KFold

class Object_Detection_Dataset:
    def __init__(self, dataset_dir, split_ratios=(0.7, 0.2, 0.1), k_folds=5, random_state=None):
        self.dataset_dir = dataset_dir
        self.split_ratios = split_ratios
        self.k_folds = k_folds
        self.random_state = random_state
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        # Load the dataset from the dataset directory
        dataset = []
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(self.dataset_dir, filename)
                dataset.append(file_path)
        return dataset

    def _split_dataset(self, dataset, split_ratios):
        # Split the dataset into training, testing, and validation sets based on the split ratios
        train_ratio, test_ratio, val_ratio = split_ratios
        assert sum(split_ratios) == 1.0, "Split ratios should sum up to 1.0"

        dataset_size = len(dataset)
        train_size = int(dataset_size * train_ratio)
        test_size = int(dataset_size * test_ratio)
        val_size = dataset_size - train_size - test_size

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:train_size+test_size]
        val_dataset = dataset[train_size+test_size:]

        return train_dataset, test_dataset, val_dataset

    def get_training_dataset(self):
        # Return the training dataset
        train_dataset, _, _ = self._split_dataset(self.dataset, self.split_ratios)
        return train_dataset

    def get_testing_dataset(self):
        # Return the testing dataset
        _, test_dataset, _ = self._split_dataset(self.dataset, self.split_ratios)
        return test_dataset

    def get_validation_dataset(self):
        # Return the validation dataset
        _, _, val_dataset = self._split_dataset(self.dataset, self.split_ratios)
        return val_dataset

    def get_kfold_datasets(self):
        # Generate k-fold cross-validation datasets
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        kfold_datasets = []
        for train_index, test_index in kfold.split(self.dataset):
            train_dataset = [self.dataset[i] for i in train_index]
            test_dataset = [self.dataset[i] for i in test_index]
            kfold_datasets.append((train_dataset, test_dataset))
        return kfold_datasets