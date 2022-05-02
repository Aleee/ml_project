import cmd2
from cmd2 import (
    CommandSet,
    with_argparser,
    with_category,
    style
)
import argparse

from .loadable import LoadableLogit, LoadableTree, LoadableForest, LoadablekNN

CONFIG_DEFAULTS = {"loadpath": "data/train.csv",
                   "exportpath": "data/submission.csv",
                   "dumppath": "data/model.joblib",
                   "model": "logit",
                   "scaler": "none",
                   "eval": "kfoldcv",
                   "split": 0.3,
                   "targetcolumn": "Cover_Type",
                   "randomstate": 42}


class MLApp(cmd2.Cmd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, auto_load_commands=False, **kwargs)

        self.hidden_commands.extend(['alias', 'edit', 'macro', 'run_pyscript', 'run_script', 'set', 'shell'])
        self.default_category = 'Встроенные команды'

        self.intro = style("Начните работу с выбора алгоритма (\'setmodel\'). "
                           "Справка: \'?\' или \'help\'. Выход: \'quit\'.",
                           bold=True)

        self.config = CONFIG_DEFAULTS
        self.data = None

        self._logit = LoadableLogit(self)
        self._tree = LoadableTree(self)
        self._forest = LoadableForest(self)
        self._knn = LoadablekNN(self)

    # SETMODEL

    setmodel_parser = cmd2.Cmd2ArgumentParser()
    setmodel_parser.add_argument('algorythm', type=str, choices=['logit', 'tree', 'forest', 'knn'],
                                 help='Выберите один из следующих алгоритмов: логистическая регрессия (logit), '
                                      'дерево решений (tree), случайный лес (forest) или k-ближайших соседей (knn)')

    @with_argparser(setmodel_parser)
    @with_category('Выбор алгоритма')
    def do_setmodel(self, ns: argparse.Namespace) -> None:
        self.config["model"] = ns.algorythm

        for command_set in [self._logit, self._tree, self._forest, self._knn]:
            try:
                self.unregister_command_set(command_set)
            except ValueError:
                pass

        try:
            self.register_command_set(getattr(self, f"_{ns.algorythm}"))
            self.poutput(f'Выбрана модель: {ns.algorythm}. Теперь настройте препроцессинг (scaler),'
                         f' методы оценки модели (seteval) или сразу перейдите к обучению (train).')
        except ValueError:
            pass


def start():
    app = MLApp()
    app.cmdloop()


if __name__ == '__main__':
    start()
