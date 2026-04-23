## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for blazing-fast Python dependency and environment management.

### 1. Install `uv`
If you do not have `uv` installed, you can install it globally via pip:
```bash
pip install uv
```
*(Alternatively, use the standalone installer from their official documentation).*

### 2. Sync the Environment
Once cloned, navigate to the project directory and run:
```bash
uv sync
```
This command will automatically create a virtual environment (`.venv`) and install all required dependencies listed in the `pyproject.toml` and `uv.lock` files.
