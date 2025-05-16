import os
import pathlib
import shutil
import subprocess
import time

class LocalPostgres:
    def __init__(self, dbdir="./.pycor/pgdata", logfile=None, port=5432, user="localuser", db="localdb"):
        self.dbdir = str(pathlib.Path(dbdir))
        print(self.dbdir)
        self.logfile = logfile or os.path.join(self.dbdir, "logfile")
        self.port = str(port)
        self.user = user
        self.db = db

    def _run(self, *args, check=True, capture_output=False):
        if pathlib.Path(self.dbdir).exists():
            print("Running", args)
            return subprocess.run(args, check=check, capture_output=capture_output, text=True)
        print("Skipped", args)
        return None

    def init(self):
        os.makedirs(self.dbdir, exist_ok=True)
        self._run("initdb", "-D", self.dbdir)
        self._run("pg_ctl", "-D", self.dbdir, "-o", f"-p {self.port}", "-l", self.logfile, "start")
        time.sleep(2)
        self._run("createuser", "-p", self.port, "-s", self.user)
        self._run("createdb", "-p", self.port, "-O", self.user, self.db)
        self._run("pg_ctl", "-D", self.dbdir, "stop")

    def start(self):
        self._run("pg_ctl", "-D", self.dbdir, "-o", f"-p {self.port}", "-l", self.logfile, "start")

    def stop(self):
        self._run("pg_ctl", "-D", self.dbdir, "stop")

    def status(self):
        self._run("pg_ctl", "-D", self.dbdir, "status")

    def remove(self):
        try:
            self._run("pg_ctl", "-D", self.dbdir, "status", capture_output=True)
            self._run("pg_ctl", "-D", self.dbdir, "stop")
        except subprocess.CalledProcessError:
            pass
        shutil.rmtree(self.dbdir, ignore_errors=True)
