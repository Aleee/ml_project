from typing import Union

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

SCALERS = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'maxabs': MaxAbsScaler(),
    'robust': RobustScaler()
}
DIMREDUCTS = {
    'pca': PCA(),
    'lda': LinearDiscriminantAnalysis()
}


def create_pipeline(scaler: str, dimreduct: str, model: Union[LogisticRegression, DecisionTreeClassifier,
                                                              RandomForestClassifier, NearestNeighbors]) -> Pipeline:
    pipeline_steps = []

    if scaler != 'none':
        pipeline_steps.append(('scaler', SCALERS[scaler]))
    if dimreduct != 'none':
        pipeline_steps.append(('dimreduct', DIMREDUCTS[dimreduct]))
    pipeline_steps.append(('classifier', model))

    return Pipeline(steps=pipeline_steps)
