import warnings
from sklearn.exceptions import ConvergenceWarning

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
from sklearn.metrics import confusion_matrix

import parameter

def compute_spd_vectorized(x_val, y_val, y_pred_val):
    """
    Vectorized computation of Statistical Parity Difference (SPD)
    Performance optimized version - avoids creating DataFrame
    
    :param x_val: Feature data (includes protected_attribute column)
    :param y_val: True labels (not used for SPD, but kept for interface consistency)
    :param y_pred_val: Predicted labels
    :return: SPD value
    """
    # Extract protected_attribute using column name
    if hasattr(x_val, 'values'):  # If pandas DataFrame
        protected_attr = x_val['protected_attribute'].values.astype(int)
    else:
        raise ValueError("x_val must be a pandas DataFrame with 'protected_attribute' column")
    
    y_pred = np.array(y_pred_val).astype(int)
    
    # Calculate positive prediction rate for both groups
    mask_group1 = (protected_attr == 1)
    mask_group2 = (protected_attr == 0)
    
    P_y1_group1 = np.mean(y_pred[mask_group1] == 1)
    P_y1_group2 = np.mean(y_pred[mask_group2] == 1)
    
    spd = abs(P_y1_group1 - P_y1_group2)
    
    return spd

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

class Fitness():

    # If it is MOO, then use list like [False, False] to represent each objective
    maximise = [True, False]
    num_obj = len(maximise)

    def __init__(self):

        self.training_in, self.training_exp, self.val_in, self.val_exp, self.test_in, self.test_exp, self.column_names = self.get_data(parameter.Dataset_train, parameter.Dataset_val, parameter.Dataset_test)


    def __call__(self, phenotype, **kwargs):

        warnings.filterwarnings("ignore", category=UserWarning)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        x_train = self.training_in
        y_train = self.training_exp
        x_val = self.val_in
        y_val = self.val_exp
        column_name = self.column_names
        model_list = eval(phenotype)

        model = build_voting_classifier(model_list) 

        try:
            # Training
            model.fit(x_train, y_train)

            y_pred_val = model.predict(x_val)

            # ACC
            accuracy_val = accuracy_score(y_val, y_pred_val)

            # Fairness - SPD calculation using vectorized method
            spd = compute_spd_vectorized(x_val, y_val, y_pred_val)

            return [accuracy_val, spd]

        except Exception as e:
            print(f"Model training error: {e}")
            if not isinstance(self.maximise, list):
                return np.nan
            else:
                return [np.nan] * len(self.maximise)

    def evaluate_on_test(self, phenotype):

        warnings.filterwarnings("ignore", category=UserWarning)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        x_train = self.training_in
        y_train = self.training_exp
        x_test = self.test_in
        y_test = self.test_exp
        column_name = self.column_names
        model_list = eval(phenotype)

        model = build_voting_classifier(model_list)

        # Training on training set
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)

        # Calculate accuracy on test set
        accuracy_test = accuracy_score(y_test, y_pred_test)

        # Calculate fairness metric on test set using vectorized method
        spd = compute_spd_vectorized(x_test, y_test, y_pred_test)

        return [accuracy_test, spd]


    def get_data(self, train, val, test):

        train_final = pd.read_csv(train)
        val_final = pd.read_csv(val)
        test_final = pd.read_csv(test)

        column_names = list(train_final.columns)

        x_train = train_final.drop(columns='Class')
        y_train = train_final['Class']

        x_val = val_final.drop(columns='Class')
        y_val = val_final['Class']

        x_test = test_final.drop(columns='Class')
        y_test = test_final['Class']

        return x_train, y_train, x_val, y_val, x_test, y_test, column_names

