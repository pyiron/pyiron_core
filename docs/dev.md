# Development Guide

## Contributing

1. Fork the repository and clone your fork.
2. Create a new branch for your feature/fix.
3. Install the development dependencies:
   ```bash
   pip install -e .[dev]
   ```
4. Run the test suite:
   ```bash
   pytest -q
   ```
5. Open a Pull Request against `main`.

## Testing

The project uses **pytest** with the following extras:
* `tests/unit/` – fast unit tests for nodes, graph utilities, and helper functions.
* `tests/integration/` – slower tests that may spin up a temporary PostgreSQL instance (see `pyiron_workflow/util.py`).

Run only unit tests with:
```bash
pytest tests/unit -q
```

## Building the Front‑End

The UI assets are built with npm:
```bash
cd pyiron_core
npm install   # installs react‑flow & other deps
npm run build # produces pyiron_core/static/widget.{js,css}
```

The `mkdocs.yml` configuration already points to the generated static files, so after rebuilding the UI you can simply run `mkdocs serve` to see the updated docs.
