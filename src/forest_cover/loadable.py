import cmd2
from cmd2 import CommandSet, with_default_category
import argparse
import numpy as np
from ast import literal_eval
from typing import Any

from .models import set_model, clean_parameters, MODELS
from .pipeline import create_pipeline
from .datahandler import load_data
from .train import train, SCORING
from .hypersearch import hypersearch, check_params_validity, append_parameter_profixes


def finilize(app: Any, parameters: dict[str, Any]) -> None:
    if app.data[0] is None:
        app.data = load_data(app.config["loadpath"], app.config["targetcolumn"])
    model = set_model(app.config["model"], parameters)
    app.poutput(
        f"Строим модель {app.config['model']} c параметрами "
        f"{parameters} (scaler: {app.config['scaler']}, "
        f"feateng: {app.config['feateng']}, dimreduct: "
        f"{app.config['dimreduct']})..."
    )
    pipeline = create_pipeline(app.config["scaler"], app.config["dimreduct"], model)
    scores = train(pipeline, app.data, parameters, app.config)
    app.poutput(
        f"Успешно! Accuracy (balanced): "
        f"{round(float(np.mean(scores['test_balanced_accuracy'])), 4)}"
    )


def parse_unknown_args(
    model: str, u_args: list[Any], k_args: argparse.Namespace
) -> dict[str, Any]:
    uknown_parameters = {}
    if len(u_args) % 2 != 0:
        raise ValueError(
            "Список дополнительных аругментов имеет "
            "неверную длину (заданы аргументы без значений)"
        )
    for name, value in zip(u_args[::2], u_args[1::2]):
        print(name, value)
        try:
            if name[:2] != "--":
                raise ValueError(
                    f"Неверный аргумент {name}: аргумент должен начинаться с --"
                )
        except ValueError as e:
            raise ValueError(
                f"Неверный аргумент {name}: аргумент должен начинаться с --"
            ) from e
        try:
            value = int(value)
            uknown_parameters[name[2:]] = value
        except ValueError:
            try:
                value = float(value)
                uknown_parameters[name[2:]] = value
            except ValueError:
                if str.lower(value) == "none":
                    value = None
                elif str.lower(value) == "true":
                    value = True
                elif str.lower(value) == "false":
                    value = False
                try:
                    uknown_parameters[name[2:]] = value
                except KeyError as e:
                    raise KeyError(
                        f"Не удалось разместить параметр {name} (возможно, дубликат?)"
                    ) from e

    print(uknown_parameters)
    known_parameters = clean_parameters(vars(k_args))
    all_parameters = uknown_parameters | known_parameters

    for key, value in all_parameters.items():
        try:
            print(f"{key}: {value}")
            MODELS[model].set_params(**{key: value})
        except ValueError as e:
            raise ValueError(
                f"Не удалось распознать введеный дополнительный " f"параметр {key}"
            ) from e

    return all_parameters


@with_default_category("Обучение и оценка (логистическая регрессия)")
class LoadableLogit(CommandSet):  # type: ignore

    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument(
        "-p",
        "--penalty",
        type=str,
        choices=["none", "l1", "l2", "elasticnet"],
        default="l2",
        help="норма регуляризации [по умолчанию: l2]",
    )
    train_parser.add_argument(
        "-c",
        "--C",
        type=float,
        default=1.0,
        help="сила регуляризации (обратная зависимость) " "[по умолчанию: 1.0]",
    )
    train_parser.add_argument(
        "-m",
        "--max_iter",
        type=int,
        default=1000,
        help="максимальное число итераций при попытке до"
        "стижения сходимости [по умолчанию: 1000]",
    )

    def __init__(self, ml_app: Any):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)  # type: ignore
    def do_train(self, ns: argparse.Namespace, unknown: list[str]) -> None:
        if ns.penalty in ["l1", "elasticnet"]:
            unknown.append("--solver")
            unknown.append("saga")
            if ns.penalty == "elasticnet":
                unknown.append("--l1_ratio")
                unknown.append("0.5")
        elif ns.penalty == "l2":
            unknown.append("--solver")
            unknown.append("lbfgs")
        finilize(self.app, parse_unknown_args(self.app.config["model"], unknown, ns))


@with_default_category("Обучение и оценка (дерево решений)")
class LoadableTree(CommandSet):  # type: ignore
    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument(
        "-d",
        "--max_depth",
        type=str,
        default="None",
        help="максимальная глубина дерева; целое "
        "положительное число или None [по "
        "умолчанию: None]",
    )
    train_parser.add_argument(
        "-f",
        "--max_features",
        type=str,
        default="auto",
        help="максимальное число признаков, используемое"
        " при разделении; число или одно из "
        "следующих значений: auto, sqrt, log2 "
        "[по умолчанию: auto]",
    )
    train_parser.add_argument(
        "-c",
        "--criterion",
        type=str,
        default="gini",
        choices=["gini", "entropy"],
        help="функция, оценивающая качество разделения",
    )

    def __init__(self, ml_app: Any):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)  # type: ignore
    def do_train(self, ns: argparse.Namespace, unknown: list[str]) -> None:
        try:
            ns.max_depth = int(ns.max_depth)
        except ValueError:
            if str.lower(ns.max_depth) == "none":
                ns.max_depth = None
        try:
            ns.max_features = int(ns.max_features)
        except ValueError:
            pass
        finilize(self.app, parse_unknown_args(self.app.config["model"], unknown, ns))


@with_default_category("Обучение и оценка (случайный лес)")
class LoadableForest(CommandSet):  # type: ignore

    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument(
        "-n",
        "--n_estimators",
        type=int,
        default=100,
        help="количество деревьев в лесу " "[по умолчанию: 100]",
    )
    train_parser.add_argument(
        "-d",
        "--max_depth",
        type=str,
        default="None",
        help="максимальная глубина дерева; "
        "целое положительное число или None "
        "[по умолчанию: None]",
    )
    train_parser.add_argument(
        "-f",
        "--max_features",
        type=str,
        default="auto",
        help="максимальное число признаков, используемое"
        " при разделении; число или одно из следующ"
        "их значений: auto, sqrt, log2 "
        "[по умолчанию: auto]",
    )
    train_parser.add_argument(
        "-c",
        "--criterion",
        type=str,
        default="gini",
        choices=["gini", "entropy"],
        help="функция, оценивающая качество " "разделения [по умолчанию: gini]",
    )

    def __init__(self, ml_app: Any):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)  # type: ignore
    def do_train(self, ns: argparse.Namespace, unknown: list[str]) -> None:
        try:
            ns.max_depth = int(ns.max_depth)
        except ValueError:
            if str.lower(ns.max_depth) == "none":
                ns.max_depth = None
        try:
            ns.max_features = int(ns.max_features)
        except ValueError:
            pass
        finilize(self.app, parse_unknown_args(self.app.config["model"], unknown, ns))


@with_default_category("Обучение и оценка (kNN)")
class LoadablekNN(CommandSet):  # type: ignore

    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument(
        "-k",
        "--n_neighbors",
        type=int,
        default=5,
        help="количество ближаших объектов, оцениваемых "
        "при классификации [по умолчанию: 5]",
    )
    train_parser.add_argument(
        "-w",
        "--weights",
        type=str,
        choices=["uniform", "distance"],
        default="uniform",
        help="функция взвешивания [по умолчанию: " "uniform]",
    )

    def __init__(self, ml_app: Any):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)  # type: ignore
    def do_train(self, ns: argparse.Namespace, unknown: list[str]) -> None:
        finilize(self.app, parse_unknown_args(self.app.config["model"], unknown, ns))


class LoadableHyperSearch(CommandSet):  # type: ignore

    hyper_parser = cmd2.Cmd2ArgumentParser()
    hyper_parser.add_argument(
        "-p",
        "--param_grid",
        type=str,
        default="",
        help="сетка параметров для функции GridSearch, "
        "представленная словарем (dict), ограничен"
        "ным фигурными скобками {...} и не содержа"
        "щим пробелов; при отсутствии аргумента "
        "будет передан стандартный набор значений "
        "для поиска",
    )

    def __init__(self, ml_app: Any):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(hyper_parser)  # type: ignore
    def do_hypersearch(self, ns: argparse.Namespace) -> None:
        if ns.param_grid:
            try:
                parameters = literal_eval(ns.param_grid)
                parameters = append_parameter_profixes(parameters)
                check_params_validity(self.app.config, parameters)
            except ValueError as e:
                print(
                    f"Не удалось расшифровать введенные параметры: они "
                    f"должны быть представлены словарем (dict) без пробелов "
                    f"с валидными для данной модели параметрами, "
                    f"заключенными в кавычки, и значениями: {e}"
                )
                return
        else:
            parameters = ""
        self.app.poutput("Оцениваем алгоритм и гиперпараметры...")
        if self.app.data[0] is None:
            self.app.data = load_data(
                self.app.config["loadpath"], self.app.config["targetcolumn"]
            )
        params, scores = hypersearch(self.app.config, self.app.data, parameters)
        accuracy_mean = float(np.mean(scores["test_" + SCORING[0]]))
        f1_mean = float(np.mean(scores["test_" + SCORING[1]]))
        roc_auc = float(np.mean(scores["test_" + SCORING[2]]))
        model = set_model(self.app.config["model"])
        pipeline = create_pipeline(
            self.app.config["scaler"], self.app.config["dimreduct"], model
        )
        train(pipeline, self.app.data, params, self.app.config, True)
        self.app.poutput(
            f"Метрики оцениваемого алгоритма (метод оценки - nested cross-validation): "
            f"accuracy (balanced): "
            f"{round(accuracy_mean, 4)}, "
            f"F1 (weighted): "
            f"{round(f1_mean, 4)}, "
            f"ROC AUC (ovo): "
            f"{round(roc_auc, 4)}. Модель: "
            f'{self.app.config["model"]}, scaler: '
            f'{self.app.config["scaler"]}, dimreduct: '
            f'{self.app.config["dimreduct"]}, feateng: '
            f'{self.app.config["feateng"]}'
        )
        self.app.poutput(
            f"Лучший набор параметров из исследованных GridSearch: {params}"
        )


@with_default_category("Обучение и оценка (логистическая регрессия)")
class LoadableLogitHyperSearch(LoadableHyperSearch):
    def __init__(self, ml_app: Any):
        super().__init__(ml_app)


@with_default_category("Обучение и оценка (дерево решений)")
class LoadableTreeHyperSearch(LoadableHyperSearch):
    def __init__(self, ml_app: Any):
        super().__init__(ml_app)


@with_default_category("Обучение и оценка (случайный лес)")
class LoadableForestHyperSearch(LoadableHyperSearch):
    def __init__(self, ml_app: Any):
        super().__init__(ml_app)


@with_default_category("Обучение и оценка (kNN)")
class LoadableKnnHyperSearch(LoadableHyperSearch):
    def __init__(self, ml_app: Any):
        super().__init__(ml_app)
