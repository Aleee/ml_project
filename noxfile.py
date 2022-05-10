import tempfile
from typing import Any

import nox

nox.options.sessions = "typecheck", "lint", "test"
locations = "src", "noxfile.py"


def install_with_constraints(session: nox.sessions.Session,
                             *args: str, **kwargs: Any) -> None:
    """Install packages constrained by Poetry's lock file.
    By default newest versions of packages are installed,
    but we use versions from poetry.lock instead to guarantee
    reproducibility of sessions.
    """
    f = tempfile.NamedTemporaryFile(mode='w', delete=False)
    session.run(
        "poetry",
        "export",
        "--dev",
        "--format=requirements.txt",
        "--without-hashes",
        f"--output={f.name}",
        external=True,
    )
    session.install(f"--constraint={f.name}", *args, **kwargs)


@nox.session
def typecheck(session: nox.sessions.Session) -> None:
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run('mypy', *args)


@nox.session
def lint(session: nox.sessions.Session) -> None:
    args = session.posargs or locations
    install_with_constraints(session, "flake8")
    session.run('flake8', *args)


@nox.session
def test(session: nox.sessions.Session) -> None:
    args = session.posargs
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "pytest")
    session.run("pytest", *args)
