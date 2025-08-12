import os
import pathlib
import shutil
import subprocess
import time


class LocalPostgres:
    def __init__(self):
        self._ci_mode = os.environ.get('GITHUB_ACTIONS') == 'true'

        self.dbdir = str(
            (pathlib.Path(__file__).parent.parent / ".pycor" / "pgdata").absolute()
        )
        self.logfile = os.path.join(self.dbdir, "logfile")
        self.port = str(5432)
        self.user = "localuser"
        self.db = "localdb"
        self.password = "none"
        self.host = "localhost"

    def boot(self):
        """Initialize and start the database (local) or connect to service container (CI)"""
        if self._ci_mode:
            print("CI mode - using service container PostgreSQL")
            return

        print("Local mode - booting LocalPostgres")
        self._stop()
        self._remove()
        self._init()
        self._start()

    def cleanup(self):
        """Stop and remove the database (local only)"""
        if self._ci_mode:
            print("CI mode - database cleanup handled by GitHub")
            return

        print("Local mode - cleaning up LocalPostgres")
        self._stop()
        self._remove()

    def _run(self, *args, check=True, capture_output=False):
        if pathlib.Path(self.dbdir).exists():
            print("Running", args)
            return subprocess.run(
                args, check=check, capture_output=capture_output, text=True
            )
        print("Skipped", args)
        return None

    def _init(self):
        os.makedirs(self.dbdir, exist_ok=True)
        self._run("initdb", "-D", self.dbdir)
        self._run(
            "pg_ctl",
            "-D",
            self.dbdir,
            "-o",
            f"-p {self.port}",
            "-l",
            self.logfile,
            "start",
        )
        time.sleep(2)
        self._run("createuser", "-p", self.port, "-s", self.user)
        self._run("createdb", "-p", self.port, "-O", self.user, self.db)
        self._run("pg_ctl", "-D", self.dbdir, "stop")

    def _start(self):
        self._run(
            "pg_ctl",
            "-D",
            self.dbdir,
            "-o",
            f"-p {self.port}",
            "-l",
            self.logfile,
            "start",
        )

    def _stop(self):
        self._run("pg_ctl", "-D", self.dbdir, "stop")

    def _status(self):
        self._run("pg_ctl", "-D", self.dbdir, "status")

    def _remove(self):
        try:
            self._run("pg_ctl", "-D", self.dbdir, "status", capture_output=True)
            self._run("pg_ctl", "-D", self.dbdir, "stop")
        except subprocess.CalledProcessError:
            pass
        shutil.rmtree(self.dbdir, ignore_errors=True)

    @property
    def connection_string(self):
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


def copy_stored_workflows(destination: str | pathlib.Path) -> list[str]:
    source = pathlib.Path(__file__).parent.parent / "stored_workflows"
    json_files = [f.name for f in source.glob("*.json")]
    for f in json_files:
        shutil.copy(source / f, pathlib.Path(destination))
    return json_files
