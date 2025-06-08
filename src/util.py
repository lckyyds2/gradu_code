import math

import numpy as np
import pandas as pd
from fancyimpute import KNN
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

from src.miNNseq import miNNseq
from src.rpca.extendedRCPA import extendedRPCA


def load_data(
    file_path_or_data,
    preprocess_func: str,
    result_col: str = "2years",
    ignore_column: list = [],
):
    if type(file_path_or_data) is str:
        data = pd.read_excel(file_path_or_data, header=[0])
    else:
        data = file_path_or_data
    data = data.drop(columns=ignore_column)

    if preprocess_func:
        data = data.dropna(subset=[result_col])

        filter_data = Preprocess().do(data, result_col, preprocess_func)
        y = np.array(filter_data[result_col])
        X = np.array(filter_data.drop([result_col], axis=1))
        return X, y, filter_data
    else:
        return data


class Preprocess:
    def __init__(self):
        self.function_map = {
            "default": self.default,
            "rpca": self.rpca,
            "KNN": self.KNN,
            "miNNseq": self.miNNseq,
        }

    def do(self, data: pd.DataFrame, result_col: str, func_name: str):
        filter_data = self.function_map[func_name](data, result_col)

        for column in filter_data.columns:
            filter_data[column] = pd.to_numeric(filter_data[column])

        return filter_data

    def _common_process(self, data: pd.DataFrame) -> pd.DataFrame:
        _data = data.copy()

        nan_ratio = _data.isna().mean()
        columns_with_high_nan_ratio = nan_ratio[nan_ratio > 0.3].index.tolist()
        _data = _data.drop(columns=columns_with_high_nan_ratio)

        return _data

    def default(self, data: pd.DataFrame, result_col: str) -> pd.DataFrame:
        filter_data = self._common_process(data)

        filter_data = filter_data.dropna()
        return filter_data

    def rpca(self, data: pd.DataFrame, result_col: str, iter=10000) -> pd.DataFrame:
        filter_data = self._common_process(data)

        y = filter_data[result_col].copy()
        X = filter_data.drop([result_col], axis=1)

        m, n = X.shape
        omega = np.where(X.notnull() == True)
        _lambda = 1.0 / math.sqrt(max(m, n))

        fill_input_data = X.copy()
        fill_input_data = fill_input_data.fillna(1000)
        rpca_output_data, e, iter, stop_conv = extendedRPCA(
            fill_input_data, omega, _lambda, 1e-7, iter
        )
        X[:] = rpca_output_data

        filter_data = X.copy()
        filter_data[result_col] = y
        return filter_data

    def KNN(self, data: pd.DataFrame, result_col: str) -> pd.DataFrame:
        filter_data = self._common_process(data)

        # find K
        _filter_data = filter_data.dropna()
        _y = _filter_data[result_col].copy()
        _X = _filter_data.drop([result_col], axis=1)
        grid = GridSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors": np.arange(
                1, int(np.sqrt(_filter_data.shape[0]))+1)},
            cv=5
        )
        grid.fit(np.array(_X), np.array(_y))
        print("best k: ", grid.best_params_["n_neighbors"])

        y = filter_data[result_col].copy()
        X = filter_data.drop([result_col], axis=1)

        filled_data = KNN(k=grid.best_params_["n_neighbors"]).fit_transform(X)
        X[:] = filled_data

        filter_data = X.copy()
        filter_data[result_col] = y
        return filter_data

    def miNNseq(self, data: pd.DataFrame, result_col: str) -> pd.DataFrame:
        filter_data = self._common_process(data)

        # find K
        _filter_data = filter_data.dropna()
        _y = _filter_data[result_col].copy()
        _X = _filter_data.drop([result_col], axis=1)
        grid = GridSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors": np.arange(
                1, int(np.sqrt(_filter_data.shape[0]))+1)},
            cv=5
        )
        grid.fit(np.array(_X), np.array(_y))
        print("best k: ", grid.best_params_["n_neighbors"])

        y = filter_data[result_col].copy()
        X = filter_data.drop([result_col], axis=1)

        filled_data = miNNseq(X.values, grid.best_params_["n_neighbors"])
        X[:] = filled_data

        filter_data = X.copy()
        filter_data[result_col] = y
        return filter_data


def train_model(clf, X_train, y_train, X_test, y_test, n_class):
    clf.fit(X_train, y_train)

    # evaluate
    accuracy = clf.score(X_test, y_test)
    y_pred_proba = clf.predict_proba(X_test)

    if n_class == 2:
        fpr, tpr, thersholds = roc_curve(y_test, y_pred_proba[:, 1])
    else:
        # multi class
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        fpr, tpr, thersholds = roc_curve(
            y_test_one_hot.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    roc = [fpr, tpr, roc_auc]

    return accuracy, roc


def train_ordered_model(clf, X_test, y_test, n_class):
    clf = clf.fit()

    # evaluate
    y_pred_proba = clf.predict(X_test)
    accuracy = accuracy_score(y_test, np.argmax(y_pred_proba, axis=1))

    if n_class == 2:
        fpr, tpr, thersholds = roc_curve(y_test, y_pred_proba[:, 1])
    else:
        # multi class
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        fpr, tpr, thersholds = roc_curve(
            y_test_one_hot.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    roc = [fpr, tpr, roc_auc]

    return accuracy, roc


def train_regression_model(clf, X_train, y_train, X_test, y_test, n_class):
    clf.fit(X_train, y_train)

    # evaluate
    y_pred_proba = np.around(clf.predict(X_test))
    accuracy = accuracy_score(y_test, y_pred_proba)

    if n_class == 2:
        fpr, tpr, thersholds = roc_curve(y_test, y_pred_proba)
    else:
        # multi class
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        y_pred_one_hot = label_binarize(
            y_pred_proba, classes=np.arange(n_class))
        fpr, tpr, thersholds = roc_curve(
            y_test_one_hot.ravel(), y_pred_one_hot.ravel())
    roc_auc = auc(fpr, tpr)
    roc = [fpr, tpr, roc_auc]

    return accuracy, roc
