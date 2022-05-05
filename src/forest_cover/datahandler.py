import pandas as pd
from .pathhandler import make_abs_path


def load_data(csv_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    try:
        dataset = pd.read_csv(make_abs_path(csv_path))
    except FileNotFoundError:
        raise FileNotFoundError(f'Не удалось загрузить файл с датасетом по указанному пути({make_abs_path(csv_path)}). '
                                f'Обновите путь командой \'setpath load\'')
    try:
        features = dataset.drop(target_column, axis=1)
        target = dataset[target_column]
    except ValueError:
        raise ValueError(f'В датасете отсутсвует столбец {target_column}. '
                         f'Укажите название столбца с независимой переменной командой \'targetcolumn\'')

    return features, target
