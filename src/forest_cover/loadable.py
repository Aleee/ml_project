import cmd2
from cmd2 import (
    CommandSet,
    with_default_category)
import argparse

from .models import set_model, clean_parameters
from .pipeline import create_pipeline
from .datahandler import load_data
from .train import train


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


@with_default_category('Обучение (логистическая регрессия)')
class LoadableLogit(CommandSet):
    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    def do_train(self, _: cmd2.Statement):
        self._cmd.poutput('logit trained')


@with_default_category('Обучение (дерево решений)')
class LoadableTree(CommandSet):
    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    def do_train(self, _: cmd2.Statement):
        self._cmd.poutput('decision tree trained')


@with_default_category('Обучение (случайный лес)')
class LoadableForest(CommandSet):
    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    def do_train(self, _: cmd2.Statement):
        self._cmd.poutput('random forest tree trained')


@with_default_category('Обучение (kNN)')
class LoadablekNN(CommandSet):
    def __init__(self, ml_app):
        super().__init__()
        self.app = ml_app

    def do_train(self, _: cmd2.Statement):
        self._cmd.poutput('kNN trained')
