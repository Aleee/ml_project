from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from .models import set_model, NOT_SO_DEFAULT_PARAMETERS
from .train import SCORING
from .pipeline import create_pipeline


DEFAULT_SEARCH_GRID = {
    'logit': {
        'clf__C': [0.1, 1, 10]
    },
    'tree': {
        'clf__max_depth': [3, 5, 10, 20, 35],
        'clf__min_samples_leaf': [5, 10, 20, 50]
    },
    'forest': {
        'clf__n_estimators': [50, 150, 250],
        'clf__max_depth': [3, 15, 50]
    },
    'knn': {
        'clf__n_neighbors': [5, 10, 20],
        'clf__leaf_size': [10, 25, 40]
    }
}


def hypersearch(config, data, parameters) -> tuple[dict, dict]:
    cv_inner = KFold(n_splits=4, shuffle=True,
                     random_state=config['randomstate'])
    cv_outer = KFold(n_splits=8, shuffle=True,
                     random_state=config['randomstate'])
    model = set_model(config['model'],
                      NOT_SO_DEFAULT_PARAMETERS[config['model']])
    if not parameters:
        parameters = DEFAULT_SEARCH_GRID[config['model']]
    pipeline = create_pipeline(scaler=config['scaler'],
                               dimreduct=config['dimreduct'], model=model)
    clf = GridSearchCV(pipeline, parameters, scoring='accuracy', n_jobs=-1,
                       cv=cv_inner, refit=True)
    scores = cross_validate(clf, data[0], data[1], cv=cv_outer,
                            scoring=SCORING, n_jobs=-1)

    search_final = GridSearchCV(pipeline, parameters, scoring='accuracy',
                                n_jobs=-1, cv=cv_inner, refit=True)

    clf = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=cv_outer)
    clf.fit(data[0], data[1])

    return search_final.fit(data[0], data[1]).best_params_, scores


def check_params_validity(config: dict, parameters: dict) -> None:
    model = set_model(config['model'], {})
    pipeline = create_pipeline(scaler=config['scaler'],
                               dimreduct=config['dimreduct'], model=model)
    pipeline.set_params(**parameters)


def append_parameter_profixes(parameters: dict) -> dict:
    return {f'clf__{k}': v for k, v in parameters.items()}
