import featuretools as ft
import pandas as pd


def make_new_features(df: pd.DataFrame) -> pd.DataFrame:

    es = ft.EntitySet()
    es.add_dataframe(
        dataframe_name="data", dataframe=df, make_index=True, index="index"
    )

    col_ignore = []
    for i in range(1, 41):
        col_ignore.append("Soil_Type" + str(i))
    feature_m, feature_d = ft.dfs(
        entityset=es,
        target_dataframe_name="data",
        trans_primitives=[
            "multiply_numeric_boolean",
            "percentile",
            "modulo_numeric_scalar",
            "add_numeric",
            "subtract_numeric",
        ],
        ignore_columns={"data": col_ignore},
        max_depth=1,
    )
    new_dataframe, new_features = ft.selection.remove_highly_correlated_features(
        feature_m, features=feature_d, pct_corr_threshold=0.96
    )
    for col in col_ignore:
        new_dataframe[col] = df[col]

    return new_dataframe
