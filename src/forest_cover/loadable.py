import cmd2
from cmd2 import (
    CommandSet,
    with_default_category)
import argparse

from .models import set_model, clean_parameters
from .pipeline import create_pipeline
from .datahandler import load_data
from .train import train


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
