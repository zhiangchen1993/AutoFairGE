"""
Pipeline Builder for End-to-End AutoML
"""

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing.lfr import LFR


class DataPreprocessor:
    """
    Data preprocessor: encoding -> order(sel/samp/scale).
    Input: DataFrame with target column.
    """

    ENCODER_MAP = {
        'OneHotEncoder': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        'LabelEncoder': LabelEncoder(),
        'OrdinalEncoder': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        'TargetEncoder': TargetEncoder(),
    }
    SCALER_MAP = {
        'MaxAbsScaler': MaxAbsScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'Normalizer': Normalizer(),
    }
    FEATURE_SELECTOR_MAP = {
        'SelectKBest': SelectKBest(k='all'),
    }
    SAMPLER_MAP = {
        'RandomOverSampler': RandomOverSampler(random_state=0),
        'RandomUnderSampler': RandomUnderSampler(random_state=0),
    }

    def __init__(self, config, protected_attribute='protected_attribute', target_col='Class'):
        self.protected_attribute = protected_attribute
        self.target_col = target_col
        self.order_steps = config.get('order', 'sel_samp_scale').split('_')

        # Components
        self.encoder = self._clone_from_map(config.get('encoder'), self.ENCODER_MAP)
        self.scaler = self._clone_from_map(config.get('scalers'), self.SCALER_MAP)
        self.feature_selector = self._clone_from_map(config.get('feature_selectors'), self.FEATURE_SELECTOR_MAP)
        self.sampler = self._clone_from_map(config.get('samplers'), self.SAMPLER_MAP)

        self._step_funcs = {'samp': self._apply_samp, 'scale': self._apply_scale, 'sel': self._apply_sel}
        self._encoder_transformer = None
        self._label_encoders = {}  # For LabelEncoder: {col: encoder}
        self._cat_cols = None

    def _clone_from_map(self, name, component_map):
        return clone(component_map[name]) if name and name in component_map else None

    def _encode(self, X, y=None, fit=False):
        if fit:
            self._cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if self.protected_attribute in self._cat_cols:
                self._cat_cols.remove(self.protected_attribute)

            if not self._cat_cols or self.encoder is None:
                self._encoder_transformer = None
                return X.copy()

            # LabelEncoder: 对每列单独处理
            if isinstance(self.encoder, LabelEncoder):
                X_encoded = X.copy()
                self._label_encoders = {}
                for col in self._cat_cols:
                    self._label_encoders[col] = LabelEncoder()
                    X_encoded[col] = self._label_encoders[col].fit_transform(X_encoded[col])
                return X_encoded

            # 其他 encoder: 使用 ColumnTransformer
            self._encoder_transformer = ColumnTransformer(
                transformers=[('cat', self.encoder, self._cat_cols)],
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            encoded = self._encoder_transformer.fit_transform(X, y)
            return pd.DataFrame(encoded, columns=self._encoder_transformer.get_feature_names_out(), index=X.index)
        else:
            if not self._cat_cols or self.encoder is None:
                return X.copy()

            # LabelEncoder
            if isinstance(self.encoder, LabelEncoder):
                X_encoded = X.copy()
                for col in self._cat_cols:
                    le = self._label_encoders[col]
                    # 处理未知类别：映射为 -1
                    mapping = {v: i for i, v in enumerate(le.classes_)}
                    X_encoded[col] = X_encoded[col].map(lambda x: mapping.get(x, -1))
                return X_encoded

            # 其他 encoder
            encoded = self._encoder_transformer.transform(X)
            return pd.DataFrame(encoded, columns=self._encoder_transformer.get_feature_names_out(), index=X.index)

    def _apply_samp(self, X_train, y_train, X_test):
        if self.sampler is None:
            return X_train, y_train, X_test
        X_res, y_res = self.sampler.fit_resample(X_train, y_train)
        return pd.DataFrame(X_res, columns=X_train.columns), y_res, X_test

    def _apply_scale(self, X_train, y_train, X_test):
        if self.scaler is None:
            return X_train, y_train, X_test

        cols = [c for c in X_train.columns if c != self.protected_attribute]
        self.scaler.fit(X_train[cols])

        X_train, X_test = X_train.copy(), X_test.copy()
        X_train[cols] = self.scaler.transform(X_train[cols])
        X_test[cols] = self.scaler.transform(X_test[cols])
        return X_train, y_train, X_test

    def _apply_sel(self, X_train, y_train, X_test):
        if self.feature_selector is None:
            return X_train, y_train, X_test

        cols = [c for c in X_train.columns if c != self.protected_attribute]

        # 设置 k 值：特征数 > 10 则 k=10，否则 k=特征数-1
        n_features = len(cols)
        k = 10 if n_features > 10 else max(1, n_features - 1)
        self.feature_selector.set_params(k=k)

        self.feature_selector.fit(X_train[cols], y_train)

        selected = [c for c, m in zip(cols, self.feature_selector.get_support()) if m]
        if self.protected_attribute:
            selected.append(self.protected_attribute)
        return X_train[selected].copy(), y_train, X_test[selected].copy()

    def fit_transform(self, train_data, test_data):
        """
        Returns: X_train, y_train, X_test, y_test
        """
        # 检查 protected_attribute 是否存在
        if self.protected_attribute and self.protected_attribute not in train_data.columns:
            raise ValueError(f"protected_attribute '{self.protected_attribute}' not found in data columns: {list(train_data.columns)}")

        X_train = train_data.drop(columns=self.target_col)
        y_train = train_data[self.target_col].values
        X_test = test_data.drop(columns=self.target_col)
        y_test = test_data[self.target_col].values

        # Encoding first (pass y for TargetEncoder)
        X_train = self._encode(X_train, y=y_train, fit=True)
        X_test = self._encode(X_test, fit=False)

        # Apply steps in order
        for step in self.order_steps:
            if step in self._step_funcs:
                X_train, y_train, X_test = self._step_funcs[step](X_train, y_train, X_test)

        # 限制小数位数（保留4位）
        X_train = X_train.round(4)
        X_test = X_test.round(4)

        return X_train, y_train, X_test, y_test


class FairPreprocessor:
    """
    Fair preprocessing step applied AFTER DataPreprocessor.
    Converts data to aif360 StandardDataset, applies fair method, converts back.
    Currently supports: LFR, None (pass through).
    """

    def __init__(self, method_name, protected_attribute='protected_attribute',
                 label_name='Class'):
        self.method_name = method_name
        self.protected_attribute = protected_attribute
        self.label_name = label_name
        self.privileged_groups = [{protected_attribute: 1}]
        self.unprivileged_groups = [{protected_attribute: 0}]

    def _to_standard_dataset(self, X, y):
        """Convert X (DataFrame) + y (array) to aif360 StandardDataset."""
        df = X.copy()
        df[self.label_name] = y
        return StandardDataset(
            df=df,
            label_name=self.label_name,
            favorable_classes=[1],
            protected_attribute_names=[self.protected_attribute],
            privileged_classes=[[1]]
        )

    def _from_standard_dataset(self, dataset):
        """Convert aif360 StandardDataset back to X (DataFrame) + y (array)."""
        df = dataset.convert_to_dataframe()[0]
        y = df[self.label_name].values
        X = df.drop(columns=self.label_name)
        return X, y

    def fit_transform(self, X_train, y_train, X_test, y_test):
        """
        Apply fair preprocessing.
        Returns: X_train, y_train, X_test, y_test
        """
        if self.method_name is None:
            return X_train, y_train, X_test, y_test

        if self.method_name == 'LFR':
            return self._apply_lfr(X_train, y_train, X_test, y_test)

        raise ValueError(f"Unknown fair preprocessing method: {self.method_name}")

    def _apply_lfr(self, X_train, y_train, X_test, y_test):
        train_ds = self._to_standard_dataset(X_train, y_train)
        test_ds = self._to_standard_dataset(X_test, y_test)

        lfr = LFR(
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
            k=10, Ax=0.1, Ay=1.0, Az=2.0,
            verbose=0
        )
        lfr = lfr.fit(train_ds, maxiter=5000, maxfun=5000)

        train_transf = lfr.transform(train_ds)
        test_transf = lfr.transform(test_ds)

        X_train_out, y_train_out = self._from_standard_dataset(train_transf)
        X_test_out, y_test_out = self._from_standard_dataset(test_transf)

        return X_train_out, y_train_out, X_test_out, y_test_out
