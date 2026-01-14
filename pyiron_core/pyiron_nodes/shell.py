from pyiron_core.pyiron_workflow import Node, as_function_node


@as_function_node
def ShellNode(
    input_writer: Node,
    command: str,
    stdin: str = "",
    environment: str = "",
    execution_directory: str = "~/.cache/pyiron_core",
    conda: str = "conda",
):
    """Execute a command on the shell.

    Args:
        input_writer (Node or None): optional node to create input files; should be able to be called without input
                                     parameters
        command (str): command to run, may contain multiple arguments split by spaces
        stdin (str, optional): optional text to be supplied on stdin
        environment (str, optional): name of conda environment to switch into before running command
        execution_directory (str): base directory where to run commands
        conda (str): conda executable name

    Returns:
        str: directory where command ran
        str: captured stdout
        str: captured stderr
        str: CalledProcessError on error, None otherwise."""

    from contextlib import chdir
    from hashlib import sha256
    from pathlib import Path
    from shlex import split
    from subprocess import CalledProcessError, run

    from pyiron_core.pyiron_database.api import get_hash

    digest = sha256()
    digest.update(command.encode("utf8"))
    if input_writer is not None:
        digest.update(get_hash(input_writer).encode("utf8"))
    if stdin is not None:
        digest.udpate(stdin.encode("utf8"))
    if environment is not None:
        digest.update(environment.encode("utf8"))

    directory = Path(execution_directory) / digest.hexdigest()
    directory = directory.expanduser()
    directory.mkdir(parents=True)

    args = split(command)
    if environment != "":
        args = [conda, "run", "-n", environment] + args

    with chdir(directory):
        if input_writer is not None:
            input_writer()
        ret = run(args, stdin=stdin, capture_output=True)

    stdout = ret.stdout.decode("utf8")
    stderr = ret.stderr.decode("utf8")

    error = None
    try:
        ret.check_returncode()
    except CalledProcessError as e:
        error = e
        pass

    return directory, stdout, stderr, error


@as_function_node
def SingleFileWriter(
    name: str,
    content: str,
):
    """Write a single file to the current directory."""
    with open(name, "w") as f:
        f.write(content)
    return name
