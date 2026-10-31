# Installation details

The reference environment targets Linux x86-64. Windows can be used for development and
smoke testing, but a Windows result is not considered reference-compatible.

## Requirements

- conda with access to conda-forge;
- Git;
- network access during installation;
- a C/C++ compiler, because PyRadiomics 3.0.1 is built from source for Python 3.10.

## Reference installation with conda

```bash
git clone https://github.com/levijb/RadiomicsViz.git
cd RadiomicsViz
conda env create --file environment.yml
conda activate radiomicviz
python -m pip install "pyradiomics==3.0.1" --no-build-isolation --no-deps
python -m pip install --editable . --no-build-isolation --no-deps
python -m pip check
radiomicviz --version
```

If `radiomicviz` is already the name of a conda environment, choose a different name:

```bash
conda env create --file environment.yml --name radiomicviz-validation
conda activate radiomicviz-validation
```

The order is intentional:

1. `environment.yml` installs Python 3.10.10, SimpleITK 2.5.5, the build tools, and all
   other reference pins.
2. `--no-build-isolation` lets the PyRadiomics build use the pinned NumPy and build tools.
3. `--no-deps` prevents pip from replacing the reference environment's versions.
4. The editable package install lets a later `git pull --ff-only` update the source. Repeat
   the install after a change to `pyproject.toml`.

The environment also pins Flask for the optional `radiomicviz view` browser viewer.

## Verify the active installation

```bash
python -c "import platform, numpy, radiomics, SimpleITK, radiomicviz; print('Python:', platform.python_version()); print('RadiomicsViz:', radiomicviz.__version__); print('PyRadiomics:', radiomics.__version__); print('SimpleITK:', SimpleITK.Version_VersionString()); print('NumPy:', numpy.__version__)"
python -m pip check
command -v radiomicviz
radiomicviz --version
```

The reference versions are Python 3.10.10, PyRadiomics 3.0.1, SimpleITK 2.5.5,
NumPy 1.26.4, and PyWavelets 1.8.0. Habitat clustering additionally checks SciPy 1.11.4
and scikit-learn 1.1.3. `command -v radiomicviz` should resolve inside
`$CONDA_PREFIX/bin`.

Extraction stops when checked versions or upstream reference state differ. Use
`--allow-nonreference` only when that difference is intentional. The command records the
deviation and stamps the result `non-reference`. The older flag name
`--allow-version-mismatch` is an equivalent alias.

## Pip-only development install

This route is not the reference environment. Use Python 3.10 and a platform compiler
(MSVC on Windows, GCC or Clang on Linux):

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip "setuptools==82.0.1" "wheel==0.46.3"
python -m pip install "numpy==1.26.4"
python -m pip install "pyradiomics==3.0.1" --no-build-isolation --no-deps
python -m pip install --editable ".[test,viewer]"
```

The `viewer` extra adds Flask; omit it if you do not need `radiomicviz view`.

Pip may select patch versions that differ from the reference pins. Use this environment
for development, or explicitly use `--allow-nonreference` for a scientific run.

## Run the tests

From the repository root:

```bash
python -m pytest -q
```

The suite uses synthetic images and fake extractors; it does not need study data or
network access. On Windows, use a short temporary path to avoid the legacy 260-character
path limit:

```powershell
python -m pytest -q --basetemp C:\rvt
```

## Troubleshooting

### Environment creation was interrupted

Do not repair a partially removed environment. Confirm its name, remove it, then recreate
it:

```bash
conda deactivate
conda env remove --name radiomicviz --yes
git pull --ff-only
conda env create --file environment.yml
```

Then repeat the two explicit `pip install` commands from the reference procedure.

### The shell finds another radiomicviz launcher

If `python -m radiomicviz --version` works but `radiomicviz --version` fails or reports a
different version:

```bash
type -a radiomicviz
command -v radiomicviz
```

A stale user launcher is commonly under `~/.local/bin`. Move or uninstall that old
launcher, run `hash -r`, and verify that the command now resolves under
`$CONDA_PREFIX/bin`.

### PyRadiomics cannot import NumPy or its build tools

```bash
python -m pip install "setuptools==82.0.1" "wheel==0.46.3" "numpy==1.26.4"
python -m pip install "pyradiomics==3.0.1" --no-build-isolation --no-deps
```

### Editable installation cannot import editables

```bash
python -m pip install "editables==0.6"
python -m pip install --editable . --no-build-isolation --no-deps
hash -r
```

### SimpleITK changed after installation

Do not add a separate `pip install SimpleITK` to the conda procedure. It can replace the
pinned conda build and will make the run non-reference. Recreate the environment if the
reference SimpleITK version has been replaced.
