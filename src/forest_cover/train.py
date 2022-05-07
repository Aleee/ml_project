from joblib import dump
from .pathhandler import make_abs_path

import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

SCORING = ['balanced_accuracy', 'f1_weighted', 'roc_auc_ovo_weighted']


def train(pipeline: Pipeline, data: tuple, parameters: dict, config: dict) -> dict:
    with mlflow.start_run(run_name=f"{config['model']} (folds={config['eval']}, rand={config['randomstate']})"):
        cv_procedure = KFold(n_splits=5, shuffle=True, random_state={config['randomstate']})
        scores = cross_validate(pipeline, data[0], data[1], cv=cv_procedure, scoring=SCORING, n_jobs=-1)
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
