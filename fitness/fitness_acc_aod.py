import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import parameter

def compute_aod_vectorized(x_val, y_val, y_pred_val):
    """
    Vectorized computation of Average Odds Difference (AOD)
    Performance optimized version - avoids creating DataFrame
    
    :param x_val: Feature data (includes protected_attribute column)
    :param y_val: True labels
    :param y_pred_val: Predicted labels
    :return: AOD value
    """
    # Extract protected_attribute using column name
    if hasattr(x_val, 'values'):  # If pandas DataFrame/Series
        protected_attr = x_val['protected_attribute'].values.astype(int)
    else:
        # If numpy array, need to know the column index
        # This should not happen in normal usage
        raise ValueError("x_val must be a pandas DataFrame with 'protected_attribute' column")
        
    if hasattr(y_val, 'values'):
        y_true = y_val.values.ravel().astype(int)
    else:
        y_true = np.array(y_val).ravel().astype(int)
    
    y_pred = np.array(y_pred_val).astype(int)
    
    # Calculate metrics for both groups
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
        model = eval(phenotype)

        try:
            # Training
            model.fit(x_train, y_train)
            y_pred_val = model.predict(x_val)

            # ACC
            accuracy_val = round(float(accuracy_score(y_val, y_pred_val)), 3)

            # Fairness - AOD calculation using vectorized method
            average_odds_difference = round(float(compute_aod_vectorized(x_val, y_val, y_pred_val)), 3)

            return [accuracy_val, average_odds_difference]

        except Exception as e:
            print(f"Model training error: {e}")
            return [np.nan] * len(self.maximise)

    def evaluate_on_test(self, phenotype):

        warnings.filterwarnings("ignore", category=UserWarning)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        x_train = self.training_in
        y_train = self.training_exp
        x_test = self.test_in
        y_test = self.test_exp
        column_name = self.column_names
        model = eval(phenotype)

        try:
            # Training on training set
            model.fit(x_train, y_train)
            y_pred_test = model.predict(x_test)

            # Calculate accuracy on test set
            accuracy_test = round(float(accuracy_score(y_test, y_pred_test)), 3)

            # Calculate fairness metric on test set using vectorized method
            average_odds_difference = round(float(compute_aod_vectorized(x_test, y_test, y_pred_test)), 3)

            return [accuracy_test, average_odds_difference]

        except Exception as e:
            print(f"Model training error on test set: {e}")
            return [np.nan] * len(self.maximise)

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

