"""
Fitness function for optimizing data preprocessing methods.
Objectives: Accuracy (maximize) and AOD (minimize)
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

import parameter
from pipeline_builder import DataPreprocessor


def generate_model_name(model, used_names):
    base = model.__class__.__name__.lower()
    index = 1
    name = f"{base}_{index}"
    while name in used_names:
        index += 1
        name = f"{base}_{index}"
    return name


def build_voting_classifier(models):
    named_estimators = []
    used_names = set()
    for model in models:
        name = generate_model_name(model, used_names)
        used_names.add(name)
        named_estimators.append((name, model))
    return VotingClassifier(estimators=named_estimators)


def compute_aod_vectorized(x_val, y_val, y_pred_val):
    """
    Vectorized computation of Average Odds Difference (AOD)
    """
    if hasattr(x_val, 'values'):
        protected_attr = x_val['protected_attribute'].values.astype(int)
    else:
        raise ValueError("x_val must be a pandas DataFrame with 'protected_attribute' column")

    if hasattr(y_val, 'values'):
        y_true = y_val.values.ravel().astype(int)
    else:
        y_true = np.array(y_val).ravel().astype(int)

    y_pred = np.array(y_pred_val).astype(int)

    TPR = np.zeros(2)
    FPR = np.zeros(2)

    for group_id in [0, 1]:
        mask = (protected_attr == group_id)
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        TP = np.sum((y_true_group == 1) & (y_pred_group == 1))
        TN = np.sum((y_true_group == 0) & (y_pred_group == 0))
        FP = np.sum((y_true_group == 0) & (y_pred_group == 1))
        FN = np.sum((y_true_group == 1) & (y_pred_group == 0))

        TPR[group_id] = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR[group_id] = FP / (FP + TN) if (FP + TN) > 0 else 0

    average_odds_difference = 0.5 * (abs(TPR[1] - TPR[0]) + abs(FPR[1] - FPR[0]))

    return average_odds_difference


class Fitness():

    maximise = [True, False]  # ACC: maximize, AOD: minimize
    num_obj = len(maximise)

    def __init__(self):
        self.train_data, self.val_data, self.test_data = self.get_data(
            parameter.Dataset_train, parameter.Dataset_val, parameter.Dataset_test
        )

    def __call__(self, phenotype, **kwargs):
        """Evaluate on validation set."""
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        try:
            # Parse phenotype
            pipeline_config = eval(phenotype)
            data_preprocessing_config = pipeline_config['data_preprocessing']
            model = pipeline_config['model']

            # If model is a list, build ensemble
            if isinstance(model, list):
                model = build_voting_classifier(model)

            # Apply preprocessing
            preprocessor = DataPreprocessor(data_preprocessing_config)
            X_train, y_train, X_val, y_val = preprocessor.fit_transform(
                self.train_data.copy(), self.val_data.copy()
            )

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred_val = model.predict(X_val)

            # ACC
            accuracy_val = round(float(accuracy_score(y_val, y_pred_val)), 3)

            # AOD
            average_odds_difference = round(float(compute_aod_vectorized(X_val, y_val, y_pred_val)), 3)

            return [accuracy_val, average_odds_difference]

        except Exception as e:
            print(f"Error: {e}")
            return [np.nan] * len(self.maximise)

    def evaluate_on_test(self, phenotype):
        """Evaluate on test set."""
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        try:
            # Parse phenotype
            pipeline_config = eval(phenotype)
            data_preprocessing_config = pipeline_config['data_preprocessing']
            model = pipeline_config['model']

            # If model is a list, build ensemble
            if isinstance(model, list):
                model = build_voting_classifier(model)

            # Apply preprocessing
            preprocessor = DataPreprocessor(data_preprocessing_config)
            X_train, y_train, X_test, y_test = preprocessor.fit_transform(
                self.train_data.copy(), self.test_data.copy()
            )

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred_test = model.predict(X_test)

            # ACC
            accuracy_test = round(float(accuracy_score(y_test, y_pred_test)), 3)

            # AOD
            average_odds_difference = round(float(compute_aod_vectorized(X_test, y_test, y_pred_test)), 3)

            return [accuracy_test, average_odds_difference]

        except Exception as e:
            print(f"Error on test set: {e}")
            return [np.nan] * len(self.maximise)

    def get_data(self, train_path, val_path, test_path):
        """Load raw data (preprocessing will be applied per phenotype)."""
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)
        return train_data, val_data, test_data
