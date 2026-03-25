import warnings
from sklearn.exceptions import ConvergenceWarning

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
            accuracy_val = accuracy_score(y_val, y_pred_val)

            b = 1 + y_pred_val - y_val
            entropy_index = np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))

            return [accuracy_val, entropy_index]

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
            accuracy_test = accuracy_score(y_test, y_pred_test)

            b = 1 + y_pred_test - y_test
            entropy_index = np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))

            return [accuracy_test, entropy_index]

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

