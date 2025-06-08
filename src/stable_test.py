import click
import numpy as np
import pandas as pd
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.dbn import SupervisedDBNClassification
from src.util import load_data, train_model, train_ordered_model


def calculate(X, y):
    n_class = np.unique(y).size

    functions = [
        # Boosting
        AdaBoostClassifier,
        GradientBoostingClassifier,
        # Bagging
        RandomForestClassifier,
        ExtraTreesClassifier,
        # Other
        SupervisedDBNClassification,
        SVC,
        OrderedModel
    ]
    results, rocs = dict(), dict()
    for function in functions:
        accuracy, roc = [], [[], [], []]
        for random_state in range(1, 51):
            '''
            The hyperparameters of GB, RF, ET and SVC are set based on:
            R. S. Olson, W. L. Cava, Z. Mustahsan, A. Varik, J. H. Moore,
            Data-driven advice for applying machine learning to bioinformatics problems.
            Pacific Symposium on Biocomputing. 23, 192–203 (2018).
            '''
            if function is AdaBoostClassifier:
                clf = function(n_estimators=1000, random_state=random_state)
            elif function is GradientBoostingClassifier:
                clf = function(loss='log_loss', n_estimators=500, max_features='log2',
                               random_state=random_state)  # deviance==log_loss
            elif function is RandomForestClassifier:
                clf = function(n_estimators=500, max_features=0.25,
                               criterion='entropy', random_state=random_state)
            elif function is ExtraTreesClassifier:
                clf = function(n_estimators=1000, max_features='log2',
                               criterion='entropy', random_state=random_state)
            elif function is SupervisedDBNClassification:
                clf = function(hidden_layers_structure=[
                               128, 64], learning_rate=5e-6, learning_rate_rbm=5e-6, n_epochs_rbm=10, dropout_p=0, verbose=False)
            elif function is SVC:
                clf = function(C=0.01, gamma=0.1, kernel="poly", coef0=10.0,
                               random_state=random_state, max_iter=1000, probability=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=random_state, stratify=y
            )

            if function is OrderedModel:
                clf = function(y_train, X_train)
                _accuracy, _roc = train_ordered_model(
                    clf, X_test, y_test, n_class
                )
            else:
                _accuracy, _roc = train_model(
                    clf, X_train, y_train, X_test, y_test, n_class
                )

            accuracy.append(_accuracy)
            roc[0].append(_roc[0].tolist())
            roc[1].append(_roc[1].tolist())
            roc[2].append(_roc[2])

        results[clf.__class__.__name__] = accuracy
        rocs[clf.__class__.__name__] = roc

    return results, rocs


@click.command()
@click.option("--data_path", help=".xlsx file path", type=str)
@click.option("--output_dir", help="Folder path for results output", type=str)
@click.option("--preprocess_func", default="default", type=str)
@click.option("--result_col", default="result", type=str)
@click.option("--ignore_column", multiple=True, default=[])
def run(data_path, output_dir, preprocess_func, result_col, ignore_column):
    ignore_column = list(ignore_column)
    X, y, filter_data = load_data(
        data_path,
        preprocess_func,
        result_col=result_col,
        ignore_column=ignore_column,
    )

    _y = y.copy()
    results, rocs = calculate(X, _y)

    pd.DataFrame(results).to_csv("{}/accuracy.csv".format(output_dir))
    np.save("{}/ROC.npy".format(output_dir), rocs)


if __name__ == "__main__":
    run()
