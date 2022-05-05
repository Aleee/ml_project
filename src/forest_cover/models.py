from typing import Union

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


MODELS = {'logit': LogisticRegression(),
          'tree': DecisionTreeClassifier(),
          'forest': RandomForestClassifier(),
          'knn': KNeighborsClassifier()}


def set_model(model: str, parameters: dict) -> Union[LogisticRegression, DecisionTreeClassifier,
                                                     RandomForestClassifier, KNeighborsClassifier]:
    return MODELS[model].set_params(**parameters)


def clean_parameters(parameters: dict) -> dict:
    for key in list(parameters.keys()):
        if key.startswith('cmd'):
            del parameters[key]
    return parameters
