import cmd2
import argparse
import warnings
import pandas as pd
import hashlib
from typing import Any

from .pathhandler import (
    make_abs_path,
    check_file_exists,
    check_dir_exists,
    check_extension,
)
from .loadable import (
    LoadableLogit,
    LoadableTree,
    LoadableForest,
    LoadablekNN,
    LoadableLogitHyperSearch,
    LoadableTreeHyperSearch,
    LoadableForestHyperSearch,
    LoadableKnnHyperSearch,
)
from .featureeng import make_new_features
from .datahandler import load_data


CONFIG_DEFAULTS: dict[str, Any] = {
    "loadpath": "data/train.csv",
    "dumppath": "data/model.joblib",
    "model": "logit",
    "scaler": "none",
    "dimreduct": "none",
    "feateng": "none",
    "eval": 10,
    "targetcolumn": "Cover_Type",
    "randomstate": 42,
}

FOREST_COVER_HASH = "44b7913f39ca108f39febca9f0aa00df821827a8"


class MLApp(cmd2.Cmd):  # type: ignore
    def __init__(self) -> None:
        super().__init__(auto_load_commands=False)

        self.hidden_commands.extend(
            [
                "alias",
                "edit",
                "macro",
                "run_pyscript",
                "run_script",
                "set",
                "shell",
                "shortcuts",
            ]
        )
        self.default_category = "Встроенные команды"

        self.intro = cmd2.style(
            "Начните работу с выбора алгоритма "
            "('setmodel'). Справка: '?' или "
            "'help'. Выход: 'quit'.",
            bold=True,
        )

        self.config = CONFIG_DEFAULTS
        self.data: tuple[Any, Any] = (None, None)

        self._logit = LoadableLogit(self)
        self._tree = LoadableTree(self)
        self._forest = LoadableForest(self)
        self._knn = LoadablekNN(self)

        self._hypersearchlogit = LoadableLogitHyperSearch(self)
        self._hypersearchtree = LoadableTreeHyperSearch(self)
        self._hypersearchforest = LoadableForestHyperSearch(self)
        self._hypersearchknn = LoadableKnnHyperSearch(self)

    # SETMODEL

    setmodel_parser = cmd2.Cmd2ArgumentParser()
    setmodel_parser.add_argument(
        "algorythm",
        type=str,
        choices=["logit", "tree", "forest", "knn"],
        help="Выберите один из следующих алгоритмов: "
        "логистическая регрессия (logit), "
        "дерево решений (tree), случайный лес "
        "(forest) или k-ближайших соседей (knn)",
    )

    @cmd2.with_argparser(setmodel_parser)
    @cmd2.with_category("Выбор алгоритма")
    def do_setmodel(self, ns: argparse.Namespace) -> None:
        self.config["model"] = ns.algorythm

        for command_set in [
            (self._logit, self._hypersearchlogit),
            (self._tree, self._hypersearchtree),
            (self._forest, self._hypersearchforest),
            (self._knn, self._hypersearchknn),
        ]:
            try:
                self.unregister_command_set(command_set[0])
                self.unregister_command_set(command_set[1])
            except ValueError:
                pass

        try:
            self.register_command_set(getattr(self, f"_{ns.algorythm}"))
            self.register_command_set(getattr(self, f"_hypersearch{ns.algorythm}"))
            self.poutput(
                f"Выбрана модель: {ns.algorythm}. Настройте "
                f'препроцессинг (список команд по "?") '
                f"или сразу перейдите к обучению (train) "
                f"или автоматическому подбору гиперпараметров "
                f"(hypersearch)."
            )
        except ValueError:
            pass

    # SETPATH

    setpath_parser = cmd2.Cmd2ArgumentParser()
    setpath_subparsers = setpath_parser.add_subparsers(
        title="подкоманды", help="справка по " "подкомандам:"
    )

    parser_load = setpath_subparsers.add_parser(
        "load", help="задать путь к " "анализируемому датасету"
    )
    parser_load.add_argument(
        "input_file", type=str, help="csv-файл с анализируемыми данными"
    )
    parser_dump = setpath_subparsers.add_parser(
        "dump", help="задать путь для сохране" "ния параметров модели"
    )
    parser_dump.add_argument(
        "dump_file",
        type=str,
        help="joblib-файл, в который будут " "сохраняться параметры модели",
    )

    def setpath_load(self, args: argparse.Namespace) -> None:
        filepath = make_abs_path(args.input_file)
        if check_file_exists(filepath):
            if check_extension(filepath, "csv"):
                self.config["loadpath"] = filepath
                self.poutput(f"Установлен путь к файлу датасета: {filepath}")
            else:
                raise ValueError("Указан путь к файлу " "с форматом, отличным от csv")
        else:
            raise FileNotFoundError(
                f"Файл по указанному пути " f"({filepath}) не найден"
            )

    parser_load.set_defaults(func=setpath_load)

    def setpath_dump(self, args: argparse.Namespace) -> None:
        filepath = make_abs_path(args.dump_file)
        if check_dir_exists(filepath):
            if check_extension(filepath, "joblib"):
                self.config["dumppath"] = filepath
                self.poutput(
                    f"Установлен путь для выгрузки данных " f"модели: {filepath}"
                )
            else:
                raise ValueError(
                    "Указан путь к файлу с " "форматом, отличным от joblib"
                )
        else:
            raise FileNotFoundError(
                f"Указанная папка для размещения " f"файла не найдена ({filepath})"
            )

    parser_dump.set_defaults(func=setpath_dump)

    @cmd2.with_argparser(setpath_parser)
    @cmd2.with_category("Управление файлами")
    def do_setpath(self, args: argparse.Namespace) -> None:
        func = getattr(args, "func", None)
        if func is not None:
            func(self, args)
        else:
            self.do_help("setpath")

    # PATHS

    paths_parser = cmd2.Cmd2ArgumentParser()
    paths_parser.add_argument(
        "-r", "--reset", action="store_true", help="сброс на значения по умолчанию"
    )

    @cmd2.with_category("Управление файлами")  # type: ignore
    @cmd2.with_argparser(paths_parser)
    def do_paths(self, args: argparse.Namespace) -> None:
        if args.reset:
            for key in ["loadpath", "exportpath", "dumppath"]:
                self.config[key] = CONFIG_DEFAULTS[key]
            self.poutput("Установлены значения путей к файлам по умолчанию")
        else:
            self.poutput(
                f"Путь к датасету:"
                f" {make_abs_path(self.config['loadpath'])}\n"
                f"Путь для выгрузки данных модели:"
                f" {make_abs_path(self.config['dumppath'])}\n"
            )

    # SCALER

    scaler_parser = cmd2.Cmd2ArgumentParser()
    scaler_parser.add_argument(
        "scaler",
        type=str,
        choices=["none", "standard", "minmax", "maxabs", "robust"],
        help="Выберите один из доступных алгоритмов "
        "машстабирования данных или отключите его "
        "(none).",
    )

    @cmd2.with_category("Препроцессинг")  # type: ignore
    @cmd2.with_argparser(scaler_parser)
    def do_scaler(self, args: argparse.Namespace) -> None:
        self.config["scaler"] = args.scaler
        if args.scaler == "none":
            self.poutput("Масштабирование данных отключено")
        else:
            self.poutput(
                f"Установлен алгоритм масштабирования " f"данных: {args.scaler}"
            )

    # DIMREDUCT

    dimreduct_parser = cmd2.Cmd2ArgumentParser()
    dimreduct_parser.add_argument(
        "dimreduct",
        type=str,
        choices=["none", "pca", "lda"],
        help="Выберите один из доступных алгоритмов "
        "уменьшения размерности "
        "или отключите его (none).",
    )

    @cmd2.with_category("Препроцессинг")  # type: ignore
    @cmd2.with_argparser(dimreduct_parser)
    def do_dimreduct(self, args: argparse.Namespace) -> None:
        self.config["dimreduct"] = args.dimreduct
        if args.dimreduct == "none":
            self.poutput("Уменьшение размерности данных отключено")
        else:
            self.poutput(
                f"Установлен алгоритм уменьшения размерности "
                f"данных: {args.dimreduct}"
            )

    # FEATENG

    feateng_parser = cmd2.Cmd2ArgumentParser()
    feateng_parser.add_argument(
        "feateng",
        type=str,
        choices=["none", "auto"],
        help="auto = набор методов feature engineering"
        " с использованием featuretools, "
        "подготовленный специально для датасета "
        "из Kaggle Forest Cover Type Prediction",
    )

    @cmd2.with_category("Препроцессинг")  # type: ignore
    @cmd2.with_argparser(feateng_parser)
    def do_feateng(self, args: argparse.Namespace) -> None:
        if args.feateng == "none":
            self.data = load_data(self.config["loadpath"], self.config["targetcolumn"])
            self.poutput("Теперь будет использоваться оригинальный датасет")
        elif args.feateng == "auto":
            if self.data[0] is None or self.config["feateng"] == "none":
                self.data = load_data(
                    self.config["loadpath"], self.config["targetcolumn"]
                )
                if (
                    hashlib.sha1(
                        pd.util.hash_pandas_object(self.data[0]).values
                    ).hexdigest()
                    != FOREST_COVER_HASH
                ):
                    self.poutput(
                        "Эта опция доступна только для датасета "
                        "Forest Cover Type Prediction"
                    )
                else:
                    self.poutput("Проводим feature engineering...")
                    self.data = make_new_features(self.data[0]), self.data[1]
                    self.poutput(
                        "Успешно! Теперь будет использоваться "
                        "датасет с кастомными признаками"
                    )
                    self.config["feateng"] = args.feateng
            else:
                self.poutput("Feature engineering уже проведен")


def start() -> None:
    warnings.filterwarnings("ignore")
    app = MLApp()
    app.cmdloop()


if __name__ == "__main__":
    start()
