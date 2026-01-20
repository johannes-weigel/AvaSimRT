PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
BLENDER_SETUP := scripts/setup_blender.sh

.PHONY: venv install dev test test-regression run clean setup-blender

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip

install: venv
	$(BIN)/pip install .

dev: venv
	$(BIN)/pip install -e .[dev]

setup-blender:
	@echo "Setting up Blender 4.2 + Mitsuba plugin..."
	@bash $(BLENDER_SETUP)

setup: dev setup-blender
	@echo "✅ Full development environment ready!"

test: dev
	$(BIN)/pytest -q

test-regression: dev
	$(BIN)/pytest tests/test_regression.py -v

run: dev
	$(BIN)/avasimrt <ARGS>

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dist build *.egg-info

clean-blender:
	rm -rf $(HOME)/.local/blender-4.2.17
	rm -rf $(HOME)/.config/blender/4.2/scripts/addons/mitsuba-blender
	@echo "✅ Blender installation cleaned"
