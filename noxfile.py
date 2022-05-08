import nox


@nox.session
def typecheck(session):
    session.install('mypy')
    session.run('mypy', 'src', '--config-file', 'mypy.ini')


@nox.session
def codeformat(session):
    session.install('black')
    session.run('black', 'SRC')


@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'src')