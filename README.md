# AvaSimRT

This project is developed in the context of a seminar at [Technical University of Berlin](https://www.tu.berlin/), Group of [Telecommunication Networks](https://www.tkn.tu-berlin.de/).

It combines [PyBullet](https://pybullet.org) with [Sionna RT](https://nvlabs.github.io/sionna/rt/index.html) to perform preparatory simulations for measuring movement in (snow) avalanches using UWB ranging techniques.

## Requirements

The project has been developed and tested with **Python 3.11.7**.

Other Python versions may work, but compatibility is not guaranteed and has not been systematically tested.

## Installation

The recommended way to set up the development environment is via the provided `Makefile`.

```bash
make dev
```

Alternatively, the setup can be performed manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

## Running the Project

After installation, the CLI can be invoked via:

```bash
avasimrt --help
```

Example configurations and demo runs are provided in the `examples/` directory.

## Development

For local development, it is recommended to use the editable install:

```bash
make dev
```

Code changes take effect immediately without reinstalling the package.
