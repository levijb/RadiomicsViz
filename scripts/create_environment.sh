#!/usr/bin/env bash

# Create the validated RadiomicsViz environment and install this checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_DIR}/environment.yml"
ENV_NAME="radiomicviz-validation-v1"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available on PATH." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "ERROR: conda environment ${ENV_NAME} already exists." >&2
    echo "Remove it explicitly or use the existing environment; this script never replaces one." >&2
    exit 1
fi

conda env create --file "${ENV_FILE}"

# PyRadiomics 3.0.1 needs NumPy and build tools present before its extension is built.
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install pyradiomics==3.0.1 --no-build-isolation --no-deps

# Dependencies are already fixed by environment.yml; do not let package installation
# resolve or replace them.
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install --editable "${REPO_DIR}" --no-build-isolation --no-deps

conda run --no-capture-output -n "${ENV_NAME}" python -m pip check
conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import platform

import numpy
import pywt
import radiomics
import scipy
import SimpleITK
import trimesh

versions = {
    "python": platform.python_version(),
    "pyradiomics": str(radiomics.__version__).lstrip("v"),
    "numpy": numpy.__version__,
    "simpleitk": SimpleITK.Version_VersionString(),
    "pywavelets": pywt.__version__,
    "scipy": scipy.__version__,
    "trimesh": trimesh.__version__,
}
expected = {
    "python": "3.10.10",
    "pyradiomics": "3.0.1",
    "numpy": "1.26.4",
    "simpleitk": "2.5.5",
    "pywavelets": "1.8.0",
    "scipy": "1.11.4",
    "trimesh": "5.2.0",
}
for name, expected_version in expected.items():
    actual = versions[name]
    if actual != expected_version:
        raise SystemExit(f"{name}: expected {expected_version}, found {actual}")
    print(f"{name}: {actual}")
PY

echo
echo "Environment created successfully. Activate it with:"
echo "  conda activate ${ENV_NAME}"
