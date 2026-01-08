PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: venv install dev test run clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip

install: venv
	$(BIN)/pip install .

dev: venv
	$(BIN)/pip install -e .[dev]

test: dev
	$(BIN)/pytest -q

run: dev
	$(BIN)/avasimrt <ARGS>

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dist build *.egg-info
