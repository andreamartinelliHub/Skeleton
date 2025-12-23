# ***Jack Skeleton***

## Table of Contents
- GIT
- UV
- Hatchling
- OmegaConf
- Hydra
- Optuna
- Lightning
- Pathlib
- Logging

## 📦 [GIT](https://git-scm.com/cheat-sheet)

The repo is cloned. You have to link it with your remote git project: `git remote -v` print nothing, right?  
Get the SSH URL: Open the page of the project in gitlab.fbk.eu > Code button (top right corner) > Copy URL  
`$ git remote add origin URL` to link this folder to a remote git repository.  
`$ git push -u origin master`: “Whenever I push or pull from now on, use origin/master as my default target”

> ***Now Jack Skeleton is tracked as starting point.***

## 🪶 [UV](https://docs.astral.sh/uv/getting-started/installation/)

uv is a lightweight tool for managing Python virtual environments and project dependencies. It allows you to create and activate isolated virtual environments easily. It is realy fast.
```
uv init # not really needed since there is already the pyproject.toml file
uv venv
source .venv/bin/activate
```
Operations of adding and removing packages are super fast:
```
uv add torch
uv remove torch
```
Import all packages you need.

## 🛞 [Hatchling](https://pypi.org/project/hatchling/)

Hatchling is a modern build backend that replaces setuptools for building Python packages cleanly, supporting both development and distribution workflows.

In pyproject.toml there are these important lines:
```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```
Means “To build my package, install hatchling and ask it to do the build.”

```
[tool.hatch.build.targets.wheel]
packages = ["src"]
```
Tells Hatchling which folders contain Python packages to include in the wheel (basically a ready-to-install package format for Python).

Then run `uv pip install -e .` to install the package as editable.

> Python adds `src` in the virtual environment and it can be imported everywhere when the venv is activated.

New functions/modules are not auto-loaded. On the other hand, it is fixed restarting the Python session (easy peasy) or with `importlib.reload(module)` in an interactive session. 

**Important**: in case of multiple custom packages reformat the folder from
<pre>project-root/
├─ src/
│  ├─ __init__.py
│  └─ .py files
:
└─ main.py </pre>
to
<pre>project-root/
├─ src/
│   ├─ package1/
│   |  ├─ __init__.py
│   |  └─ .py files
│   └─ package2/
│      ├─ __init__.py
│      └─ .py files
:
└─ main.py </pre>
Remember to change also
```
[tool.hatch.build.targets.wheel]
packages = ["src.package1", "src.package2"]
```
In this way the `src` layout is perfect. Now you can import your different packages with `import package1` (and similar) in all files and notebooks.

## 🔀 [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/)
OmegaConf is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources.

It unlocks variable interpolations, usage of resoovers, merging configurations and so on.

In our case everything is set up via Hydra.

## 📢 [Hydra](https://hydra.cc/docs/intro/)
Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

To import: `$ uv add hydra-core --upgrade`

Looking at the main() function in main.py, you can see the decorator
> @hydra.main(version_base=None, config_path="conf", config_name="config")  
def my_app(cfg : DictConfig):

Here:
- `version_base=None` = you want to use default hydra settings
- `config_path="conf"` = the configuration folder is ./conf/
- `config_name="config"` = hydra will create a DictConfig variable loading config_path/config.yaml

The DictConfig variable is imported from OmegaConf and allows easier access to elements in the dict, e.g via dot notation access. It is possible to override elements via CLI config entries.

It’s useful to:
- centralize config for data paths, models, hyperparameters (look at config/config.yaml)
- Support hierarchical configs and overrides (look at config/clusterConf.yaml)
- Work well with experiment tracking (logs, checkpoints)
- Encourage reproducible experiments 

## 🕹️ [Optuna]()

## 🗲 [Lightning]()

## 🛣️ [Pathlib]()

## 🪵 [Logging](https://docs.python.org/3/howto/logging.html#basic-logging-tutorial)

The logging module is Python’s standard library for tracking events in applications. It allows you to record messages about your program’s execution.

Adding in any module the lines below, the tracking of the workflow will be easier:
```
from src import utils
logger = utils.get_logger(__name__)
```
All settings of the logger are defined in `src/utils.py`: few changes and everything will be shared in all modules.




# Backstage
- [Git Installation](#git-installation-)
- [Environment Creation](#-environment-creation)

## 🛠️ Workflow of main.py
> Handled with booleans in `config/default_config.yaml:settings.pipeline`
1. Preprocessing of data: from `catchme_model_train/raw_dataset_preproccessing.py`
2. Dataset creation: from `catchme_model_train/dataset_creation.py`
3. Models training and testing

> Training command (from ROOT/catchme_model): `python main.py --config-name=abacusConf -m settings.epochs=130 settings.patch_size=1500 settings.batch_size=-1`


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
