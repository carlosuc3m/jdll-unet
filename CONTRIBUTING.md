# Contributing

Use a Python 3.10+ environment with the package installed in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Before opening a change, run:

```bash
python -m pytest -q
python -m compileall -q jdll_unet tests
python -m ruff check .
python -m build --sdist --wheel
```

The package keeps the Appose-facing public API small:

- `jdll_unet.appose_api.train(config, task=None)`
- `jdll_unet.appose_api.infer(config, inputs, task=None)`
- `jdll_unet.appose_api.detect_task(config)`

Keep configuration and output payloads JSON-serializable unless they are explicit image arrays returned from inference.
