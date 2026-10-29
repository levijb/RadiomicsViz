# Installation details

The [README](README.md#installation) lists the recommended installation commands. This
document explains the supported platform, why the commands run in that order, how to check
the result and how to recover from common problems.

## Supported platform and requirements

The reference workflow is validated for Linux x86-64. The recommended route needs:

- Git, to clone the repository;
- conda (for example Miniforge, Miniconda or Anaconda) with access to the conda-forge
  channel, which is the only channel [`environment.yml`](environment.yml) uses;
- network access to conda-forge and PyPI during installation; and
- a C compiler such as GCC or Clang. PyRadiomics 3.0.1 is built from its source
  distribution for Python 3.10, including its compiled extensions.

Conda installs Python 3.10.10 and SimpleITK 2.5.5. The pip section of `environment.yml`
installs the remaining pinned packages, including NumPy 1.26.4, PyWavelets 1.8.0,
SciPy 1.11.4, scikit-learn 1.1.3, pytest and the build tools setuptools, wheel,
Hatchling and editables.

### Pip-only installation

Pip-only installation is useful for development or diagnostic runs, but it is not the
frozen reference environment. Use Python 3.10 and ensure a C compiler is available
(GCC/Clang on Linux or macOS, or MSVC on Windows):

```bash
git clone https://github.com/levijb/RadiomicsViz.git
cd RadiomicsViz
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.26,<2"
python -m pip install "pyradiomics==3.0.1" --no-build-isolation --no-deps
python -m pip install --editable .
```

The last command lets pip choose the remaining dependency versions, so they may differ
from the reference versions. `radiomicviz whole` then stops unless the run is explicitly
marked non-reference with `--allow-version-mismatch`.

RadiomicsViz is not currently distributed through PyPI, and the planned `viewer`,
`analysis` and `all` optional dependency groups are not available in this release.

## Why the install steps are ordered as they are

1. **The conda environment comes first.** Conda resolves Python and SimpleITK in one
   transaction, then pip installs the packages pinned in the file's pip section.
   SimpleITK is a conda dependency rather than a pip requirement because some conda
   configurations add an older SimpleITK, installed with distutils, to every new
   environment. Pip cannot safely uninstall such a package; conda can replace it.
2. **PyRadiomics comes second.** Its build compiles extensions against NumPy.
   `--no-build-isolation` makes the build use the environment's pinned NumPy, setuptools
   and wheel instead of downloading newer build tools into a temporary environment.
   `--no-deps` stops pip from installing or upgrading PyRadiomics' own requirements,
   which `environment.yml` already provides at the pinned versions.
3. **RadiomicsViz comes third, as an editable install.** Its build backend (Hatchling)
   and the `editables` helper are already pinned in the environment, so build isolation
   is not needed, and `--no-deps` again keeps every pinned version. Because the install
   is editable, `git pull` updates the code in place. Repeat the install command after an
   update that changes `pyproject.toml`, for example a new command entry point.
4. **The checks come last.** `pip check` confirms that the installed packages satisfy
   each other's requirements, and `radiomicviz --version` confirms that the shell finds
   the command.

## How to verify the installation

With the environment active, run:

```bash
python -m pip check
radiomicviz --version
command -v radiomicviz
python -c "import platform, numpy, radiomics, SimpleITK; print('Python', platform.python_version()); print('PyRadiomics', radiomics.__version__); print('SimpleITK', SimpleITK.Version_VersionString()); print('NumPy', numpy.__version__)"
```

Expected results:

- `pip check` prints `No broken requirements found.`
- `radiomicviz --version` prints `radiomicviz` followed by the version of your checkout.
- `command -v radiomicviz` prints a path inside the active environment,
  `$CONDA_PREFIX/bin/radiomicviz`. If it prints another location, see
  [Another radiomicviz command runs](#another-radiomicviz-command-runs).
- The last command prints Python 3.10.10, PyRadiomics v3.0.1 (PyRadiomics reports its
  version with a leading `v`), SimpleITK 2.5.5 and NumPy 1.26.4.

`radiomicviz whole` checks these versions, and PyWavelets 1.8.0, before every run. A
mismatch stops the run unless `--allow-version-mismatch` is given, in which case every
output is stamped non-reference.

### Running the test suite

The conda environment already includes pytest. From the repository root, run:

```bash
python -m pytest -q
```

For a pip-only installation, install the test dependency first:

```bash
python -m pip install --editable ".[test]"
python -m pytest -q
```

## Troubleshooting

### Environment creation failed or was interrupted

A failed or interrupted `conda env create`, or an interrupted `conda env remove`, leaves
an incomplete environment that should not be used or repaired in place. Let any running
conda command finish without interrupting or suspending it. Then remove the environment
completely, pull the current environment file and recreate it:

```bash
conda deactivate
conda env remove --name radiomicviz --yes
git pull --ff-only
conda env create --file environment.yml
conda activate radiomicviz
```

`conda env list` should not show `radiomicviz` between the removal and the new creation.
After recreation, continue with step 3 of the README installation.

Typical symptoms of an incomplete environment:

- `Cannot uninstall SimpleITK 2.2.1` during the pip phase of environment creation. The
  current `environment.yml` installs SimpleITK 2.5.5 through conda, which avoids asking
  pip to overwrite a legacy distutils installation.
- Errors about missing package metadata, such as a missing `METADATA` file inside a
  `.dist-info` folder, after an interrupted removal.

### No module named 'editables'

If editable package installation reports `No module named 'editables'`, install the
pinned editable-build helper and retry. Current versions of `environment.yml` install it
automatically:

```bash
python -m pip install "editables==0.6"
python -m pip install --editable . --no-build-isolation --no-deps
hash -r
```

`hash -r` makes Bash forget previously found command locations, so the next
`radiomicviz` call finds the newly installed command.

### Another radiomicviz command runs

If `radiomicviz --version` fails with an import error or reports an unexpected version,
or `command -v radiomicviz` prints a path outside `$CONDA_PREFIX/bin`, an older
installation is probably earlier on `PATH`. A common case is a launcher left in
`~/.local/bin` by a previous user-level installation.

1. Check the environment's own package directly. `python -m radiomicviz --version`
   always uses the active environment's Python.
2. List every `radiomicviz` the shell can find with `type -a radiomicviz`.
3. Move the stale launcher aside rather than deleting it, for example
   `mv ~/.local/bin/radiomicviz ~/.local/bin/radiomicviz.old`, or uninstall the old
   package with the Python that installed it.
4. Run `hash -r` so Bash forgets the old location, then check `command -v radiomicviz`
   again.

### PyRadiomics build cannot import NumPy or build tools

This project intentionally pins PyRadiomics 3.0.1. If its build cannot import NumPy or
other build tools, confirm the environment is active and run the commands in this order:

```bash
python -m pip install --upgrade setuptools wheel
python -m pip install "numpy==1.26.4"
python -m pip install "pyradiomics==3.0.1" --no-build-isolation --no-deps
```

### Do not install SimpleITK separately

Do not add a separate `pip install SimpleITK` step to the conda procedure. Installing
another SimpleITK version, such as 2.2.1, would make runs non-reference relative to this
package's protocol.
