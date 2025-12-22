# ***Jack Skeleton***

## Table of Contents
- Logging
- Hatchling
- UV
- git
- Hydra
- Optuna
- Lightning
- Pathlib

## [Logging](https://docs.python.org/3/howto/logging.html#basic-logging-tutorial)

The logging module is Python’s standard library for tracking events in applications. It allows you to record messages about your program’s execution.

Adding in any module the lines below, the tracking of the workflow will be easier:
```
from src import utils
logger = utils.get_logger(__name__)
```
All settings of the logger are defined in `src/utils.py`: few changes and everything will be shared in all modules.

## [Hatchling](https://pypi.org/project/hatchling/)

Hatchling is a modern build system that replaces setuptools for building Python packages cleanly, supporting both development and distribution workflows.

In the pyproject.toml:
```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```
Then run `uv pip install -e .` to install the package as editable.

> Python links `src` in the virtual environment and it can be imported everywhere when the venv is activated.

New functions/ modules are not auto-loaded. On the other hand, it is fixed restarting the Python session (easy peasy) or with `importlib.reload(module)` in an interactive session. 

## [UV](https://docs.astral.sh/uv/getting-started/installation/)

uv is a lightweight tool for managing Python virtual environments and project dependencies. It allows you to create and activate isolated virtual environments easily. It is realy fast.
```
uv init
uv venv
source .venv/bin/activate
```
Operations of adding a removing packages are super fast:
```
uv add torch
uv remove torch
```

## [Hydra]()
## [Optuna]()
## [OmegaConf]()
## [Pathlib]()




# Backstage
- [Git Installation](#git-installation-)
- [Environment Creation](#-environment-creation)

## 🛠️ Workflow of main.py
> Handled with booleans in `config/default_config.yaml:settings.pipeline`
1. Preprocessing of data: from `catchme_model_train/raw_dataset_preproccessing.py`
  - Unzip data 
  - Remove .Identifier, portion of empty files and too rare classes
  - Check the size of images to match with expected one in `img_standard_size`
2. Dataset creation: from `catchme_model_train/dataset_creation.py`
  - From the global folder containing all data 
  - Test, Val and Train Sets are created balancing the number of instances of each class
  - Distributions are printed for double check
  - Creation of a YAML file: YOLO needs it to check paths, classes and tasks
3. Models training and testing:
  - Init the desired model type and size
  - Train for `settings.epochs` epochs
  - Test the best weights on Val and Test Sets

> Training command (from ROOT/catchme_model): `python main.py --config-name=abacusConf -m settings.epochs=130 settings.patch_size=1500 settings.batch_size=-1`
## 📦 Git Installation

Run `git clone git@gitlab.fbk.eu:dsip/dsip_industry/dsip_di_catchme/catchme_model.git` to clone the git repo.

The folder has to be installed in a Root folder structured like below.
<pre lang="markdown"> <code> Root/
    ├── catchme_model/
    └── data/ </code> </pre>


*Useful git functions*:

To check available branches: `git branch -r`

To switch to remote-only branches creating a local version: `git checkout -b <branch-name> origin/branch-name>`

## 🌿 Environment Creation
The cloned folder contains a file `pyproject.toml` containing all the dependencies.
The creation of the environment is handled by uv: https://docs.astral.sh/uv/getting-started/

After cloning the repo, you can find a git-tracked file called 'requirements.yml'.
It contains all needed pip packages.
- `uv venv .venv --python 3.8`
- `uv init`
- `uv sync`
 
***Updating the env***: uv add and uv remove

***Activating the env***: source .venv/bin/activate

***Remove files***
- `git rm --cached filename`: remove the cached version, but keep it locally
- `git rm filename`: completely removed 
- `git ls-files`: check tracked files
