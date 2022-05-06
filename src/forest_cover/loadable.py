import cmd2
from cmd2 import (
    CommandSet,
    with_default_category)
import argparse
import numpy as np
from typing import Any

from .models import set_model, clean_parameters, MODELS
from .pipeline import create_pipeline
from .datahandler import load_data
from .train import train


def finilize(app: Any, parameters: dict) -> None:
    if not app.data:
        app.data = load_data(app.config['loadpath'], app.config['targetcolumn'])
    model = set_model(app.config['model'], parameters)
    app.poutput(f"Строим модель {app.config['model']} c параметрами {parameters} (scaler: {app.config['scaler']}, "
                f"feateng: {app.config['feateng']}, dimreduct: {app.config['dimreduct']})...")
    pipeline = create_pipeline(app.config['scaler'], app.config['dimreduct'], model)
    scores = train(pipeline, app.data, parameters, app.config)
    app.poutput(f"Успешно! Accuracy (balanced): {round(float(np.mean(scores['test_balanced_accuracy'])), 4)}")


def parse_unknown_args(model: str, u_args: list, k_args: argparse.Namespace) -> dict:
    uknown_parameters = {}
    if len(u_args) % 2 != 0:
        raise ValueError('Список дополнительных аругментов имеет неверную длину (заданы аргументы без значений)')
    for name, value in zip(u_args[::2], u_args[1::2]):
        try:
            if name[:2] != '--':
                raise ValueError(f'Неверный аргумент {name}: аргумент должен начинаться с --')
        except ValueError as e:
            raise ValueError(f'Неверный аргумент {name}: аргумент должен начинаться с --') from e
        try:
            value = dict(value)
        except ValueError:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if str.lower(value) == 'none':
                        value = None
                    elif str.lower(value) == 'true':
                        value = True
                    elif str.lower(value) == 'false':
                        value = False
            try:
                uknown_parameters[name[2:]] = value
            except KeyError as e:
                raise KeyError(f'Не удалось разместить параметр {name} (возможно, дубликат?)') from e

    known_parameters = clean_parameters(vars(k_args))
    all_parameters = uknown_parameters | known_parameters

    for key, value in all_parameters.items():
        try:
            MODELS[model].set_params(**{key: value})
        except ValueError as e:
            raise ValueError(f'Не удалось распознать введеный дополнительный параметр {key}') from e

    return all_parameters


@with_default_category('Обучение и оценка (логистическая регрессия)')
class LoadableLogit(CommandSet):

    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument('-p', '--penalty', type=str, choices=['none', 'l1', 'l2', 'elasticnet'], default='l2',
                              help='норма регуляризации [по умолчанию: l2]')
    train_parser.add_argument('-c', '--C', type=float, default=1.0,
                              help='сила регуляризации (обратная зависимость) [по умолчанию: 1.0]')
    train_parser.add_argument('-m', '--max_iter', type=int, default=100,
                              help='максимальное число итераций при попытке достижения сходимости [по умолчанию: 100]')

    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)
    def do_train(self, ns: argparse.Namespace, unknown: list) -> None:
        if ns.penalty in ['l1', 'elasticnet']:
            unknown.append('--solver')
            unknown.append('saga')
            if ns.penalty == 'elasticnet':
                unknown.append('--l1_ratio')
                unknown.append('0.5')
        finilize(self.app, parse_unknown_args(self.app.config['model'], unknown, ns))


@with_default_category('Обучение и оценка (дерево решений)')
class LoadableTree(CommandSet):
    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument('-d', '--max_depth', type=str, default='None',
                              help='максимальная глубина дерева; целое положительное число или None '
                                   '[по умолчанию: None]')
    train_parser.add_argument('-f', '--max_features', type=str, default='auto',
                              help='максимальное число признаков, используемое при разделении; число или одно из '
                                   'следующих значений: auto, sqrt, log2 [по умолчанию: auto]')
    train_parser.add_argument('-c', '--criterion', type=str, default='gini', choices=['gini', 'entropy'],
                              help='функция, оценивающая качество разделения')

    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)
    def do_train(self, ns: argparse.Namespace, unknown: list) -> None:
        try:
            ns.max_depth = int(ns.max_depth)
        except ValueError:
            if str.lower(ns.max_depth) == 'none':
                ns.max_depth = None
        try:
            ns.max_features = int(ns.max_features)
        except ValueError:
            pass
        finilize(self.app, parse_unknown_args(self.app.config['model'], unknown, ns))


@with_default_category('Обучение и оценка (случайный лес)')
class LoadableForest(CommandSet):

    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument('-n', '--n_estimators', type=int, default=100,
                              help='количество деревьев в лесу [по умолчанию: 100]')
    train_parser.add_argument('-d', '--max_depth', type=str, default='None',
                              help='максимальная глубина дерева; целое положительное число или None '
                                   '[по умолчанию: None]')
    train_parser.add_argument('-f', '--max_features', type=str, default='auto',
                              help='максимальное число признаков, используемое при разделении; число или одно из '
                                   'следующих значений: auto, sqrt, log2 [по умолчанию: auto]')
    train_parser.add_argument('-c', '--criterion', type=str, default='gini', choices=['gini', 'entropy'],
                              help='функция, оценивающая качество разделения [по умолчанию: gini]')

    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)
    def do_train(self, ns: argparse.Namespace, unknown: list) -> None:
        try:
            ns.max_depth = int(ns.max_depth)
        except ValueError:
            if str.lower(ns.max_depth) == 'none':
                ns.max_depth = None
        try:
            ns.max_features = int(ns.max_features)
        except ValueError:
            pass
        finilize(self.app, parse_unknown_args(self.app.config['model'], unknown, ns))


@with_default_category('Обучение и оценка (kNN)')
class LoadablekNN(CommandSet):

    train_parser = cmd2.Cmd2ArgumentParser()
    train_parser.add_argument('-k', '--n_neighbors', type=int, default=5,
                              help='количество ближаших объектов, оцениваемых при классификации [по умолчанию: 5]')
    train_parser.add_argument('-w', '--weights', type=str, choices=['uniform', 'distance'], default='uniform',
                              help='функция взвешивания [по умолчанию: uniform]')

    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    @cmd2.with_argparser(train_parser, with_unknown_args=True)
    def do_train(self, ns: argparse.Namespace, unknown: list) -> None:
        finilize(self.app, parse_unknown_args(self.app.config['model'], unknown, ns))
