from typing import Union, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


MODELS: dict[str, Any] = {'logit': LogisticRegression(),
                          'tree': DecisionTreeClassifier(),
                          'forest': RandomForestClassifier(),
                          'knn': KNeighborsClassifier()}

NOT_SO_DEFAULT_PARAMETERS: dict[str, Any] = {
    'logit': {'max_iter': 1000},
    'tree': {},
    'forest': {},
    'knn': {}
}


def set_model(model: str,
              parameters: dict[str, Any]) \
        -> Union[LogisticRegression, DecisionTreeClassifier,
                 RandomForestClassifier, KNeighborsClassifier]:
    return MODELS[model].set_params(**parameters)


def clean_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    for key in list(parameters.keys()):
        if key.startswith('cmd'):
            del parameters[key]
    return parameters
