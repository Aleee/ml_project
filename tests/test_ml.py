import cmd2_ext_test
from cmd2 import CommandResult
import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),  '../src/')))
from forest_cover.ml import MLApp


ERR_ARGUMENT = "Error: argument"
ERR_CHOICE = "Error: invalid choice"
EXC_FILENOTFOUND = "FileNotFoundError"
EXC_VALUEERROR = "ValueError"
EXC_KEYERROR = "KeyError"
TXT_RANDOMSTATE = "Число должно быть положительным"
TXT_FEATENGDONE = "Эта опция доступна только для датасета Forest Cover Type Prediction"


class AppTester(cmd2_ext_test.ExternalTestMixin, MLApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@pytest.fixture
def ml_app():
    app = AppTester()
    app.fixture_setup()
    yield app
    app.fixture_teardown()


def common_asserts(out):
    assert isinstance(out, CommandResult)
    assert str(out.stdout).strip() == ''
    assert out.data is None


def test_wrong_algorythm_name(ml_app):
    out = ml_app.app_cmd("setmodel logi")
    common_asserts(out)
    assert ERR_ARGUMENT in out.stderr


def test_setpath_wrong_subcommand(ml_app):
    out = ml_app.app_cmd("setpath loaddd")
    common_asserts(out)
    assert ERR_CHOICE in out.stderr


def test_setpath_load_nofile(ml_app):
    filename = "inreinre"
    out = ml_app.app_cmd("setpath load " + filename)
    common_asserts(out)
    assert EXC_FILENOTFOUND in out.stderr
    assert filename in out.stderr


def test_randomstate_wrong_input(ml_app):
    out = ml_app.app_cmd("randomstate 0")
    assert TXT_RANDOMSTATE in out.stdout
    out = ml_app.app_cmd("randomstate -45")
    assert TXT_RANDOMSTATE in out.stdout
    out = ml_app.app_cmd("randomstate sometext")
    assert ERR_ARGUMENT in out.stderr


def test_wrong_independetcolumn_name(ml_app):
    out = ml_app.app_cmd("targetcolumn cover_type")
    assert EXC_KEYERROR in out.stderr


def test_dimreduct_wrong_input(ml_app):
    out = ml_app.app_cmd("dimreduct tsne")
    common_asserts(out)
    assert ERR_ARGUMENT in out.stderr


def test_scaler_wrong_input(ml_app):
    out = ml_app.app_cmd("scaler nonimportant")
    common_asserts(out)
    assert ERR_ARGUMENT in out.stderr


def test_feateng_wrong_input(ml_app):
    out = ml_app.app_cmd("feateng complicated")
    common_asserts(out)
    assert ERR_ARGUMENT in out.stderr


def test_feateng_not_forestdataset(ml_app):
    ml_app.app_cmd("feateng auto")
    out = ml_app.app_cmd("feateng auto")
    assert TXT_FEATENGDONE in out.stdout


def test_feateng_duplicate_runs(ml_app):
    out = ml_app.app_cmd("feateng auto")
    assert TXT_FEATENGDONE in out.stdout


def test_wrong_train_parameters_unknown(ml_app):
    ml_app.app_cmd("setmodel logit")
    out = ml_app.app_cmd("train --max_depth 30")
    assert EXC_VALUEERROR in out.stderr


def test_wrong_train_parameters_badsyntax(ml_app):
    ml_app.app_cmd("setmodel logit")
    out = ml_app.app_cmd("train -intercept_scaling 1")
    assert EXC_VALUEERROR in out.stderr
