from joblib import dump
from .pathhandler import make_abs_path

import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


def train(pipeline: Pipeline, data: tuple, parameters: dict, config: dict) -> dict:
    scoring = ['balanced_accuracy', 'f1_weighted', 'roc_auc_ovo_weighted']
    with mlflow.start_run(run_name=f"{config['model']} (folds={config['eval']}, rand={config['randomstate']})"):
        scores = cross_validate(pipeline, data[0], data[1], cv=config['eval'], scoring=scoring,
                                return_train_score=False, error_score="raise")
        mlflow.log_param("SCALER", config["scaler"])
        mlflow.log_param("FEATENG", config["feateng"])
        mlflow.log_param("DIMREDUCT", config["dimreduct"])
        mlflow.log_params(parameters)
        mlflow.log_metrics(
            {"accuracy_balanced": float(np.mean(scores['test_balanced_accuracy'])),
             "F1_weighted": float(np.mean(scores['test_f1_weighted'])),
             "ROC_AUC": float(np.mean(scores['test_roc_auc_ovo_weighted']))}
        )
        mlflow.sklearn.log_model(pipeline, 'logit')
        dump(pipeline, make_abs_path(config['dumppath']))
        return scores
