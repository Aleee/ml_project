import nox


@nox.session
def typecheck(session):
    session.install('mypy')
    session.run('mypy', 'src')


@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'src')