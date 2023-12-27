# Bare Bones Language Model

A PyTorch starter for doing something actually interesting.

## Development

This project follows these principles:

- _Explicit_, because you shouldn't have to guess
- _Dependency-light_, because dependencies break
- _Concise_, because verbose code is skimmed & ignored

## Setup

```sh
python3 -m venv .venv
# Add to the end of .venv/bin/activate
# export PYTHONPATH="${PYTHONPATH}:$(dirname ${VIRTUAL_ENV})"

source .venv/bin/activate
pip install wheel
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # for CPU
pip install -r requirements-dev.txt

./dev  # run CI checks
```
