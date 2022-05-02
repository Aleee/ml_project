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

    # SETPATH

    setpath_parser = cmd2.Cmd2ArgumentParser()
    setpath_subparsers = setpath_parser.add_subparsers(title='подкоманды', help='справка по подкомандам:')

    parser_load = setpath_subparsers.add_parser('load', help='задать путь к анализируемому датасету')
    parser_load.add_argument('input_file', type=str, help='csv-файл с анализируемыми данными')
    parser_export = setpath_subparsers.add_parser('export', help='задать путь для сохранения csv-файла с предиктом')
    parser_export.add_argument('output_file', type=str,
                               help='csv-файл, который будет создан по результам работы предикта')
    parser_dump = setpath_subparsers.add_parser('dump', help='задать путь для сохранения параметров модели')
    parser_dump.add_argument('dump_file', type=str, help='joblib-файл, в который будут сохраняться параметры модели')

    def setpath_load(self, args: argparse.Namespace) -> None:
        filepath = make_abs_path(args.input_file)
        if check_file_exists(filepath):
            if check_extension(filepath, "csv"):
                self.config["loadpath"] = filepath
                self.poutput(f"Установлен путь к файлу датасета: {filepath}")
            else:
                raise ValueError("Указан путь к файлу с форматом, отличным от csv")
        else:
            raise FileNotFoundError(f"Файл по указанному пути ({filepath}) не найден")
    parser_load.set_defaults(func=setpath_load)

    def setpath_export(self, args: argparse.Namespace) -> None:
        filepath = make_abs_path(args.output_file)
        if check_dir_exists(filepath):
            if check_extension(filepath, "csv"):
                self.config["exportpath"] = filepath
                self.poutput(f"Установлен путь для выгрузки предикта: {filepath}")
            else:
                raise ValueError("Указан путь к файлу с форматом, отличным от csv")
        else:
            raise FileNotFoundError(f"Указанная папка для размещения файла не найдена ({filepath})")
    parser_export.set_defaults(func=setpath_export)

    def setpath_dump(self, args: argparse.Namespace) -> None:
        filepath = make_abs_path(args.dump_file)
        if check_dir_exists(filepath):
            if check_extension(filepath, "joblib"):
                self.config["dumppath"] = filepath
                self.poutput(f"Установлен путь для выгрузки данных модели: {filepath}")
            else:
                raise ValueError("Указан путь к файлу с форматом, отличным от joblib")
        else:
            raise FileNotFoundError(f"Указанная папка для размещения файла не найдена ({filepath})")
    parser_dump.set_defaults(func=setpath_dump)

    @cmd2.with_argparser(setpath_parser)
    @with_category('Управление файлами')
    def do_setpath(self, args: argparse.Namespace):
        func = getattr(args, 'func', None)
        if func is not None:
            func(self, args)
        else:
            self.do_help('setpath')

    # PATHS

    paths_parser = cmd2.Cmd2ArgumentParser()
    paths_parser.add_argument('-r', '--reset', action='store_true', help='сброс на значения по умолчанию')

    @with_category('Управление файлами')
    @cmd2.with_argparser(paths_parser)
    def do_paths(self, args):
        if args.reset:
            for key in ["loadpath", "exportpath", "dumppath"]:
                self.config[key] = CONFIG_DEFAULTS[key]
            self.poutput("Установлены значения путей к файлам по умолчанию")
        else:
            self.poutput(f"Путь к датасету: {make_abs_path(self.config['loadpath'])}\n"
                         f"Путь для выгрузки предикта: {make_abs_path(self.config['exportpath'])}\n"
                         f"Путь для выгрузки данных модели: {make_abs_path(self.config['dumppath'])}\n")


def start():
    app = MLApp()
    app.cmdloop()


if __name__ == '__main__':
    start()
