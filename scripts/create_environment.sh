#!/usr/bin/env bash

# Optional convenience wrapper for the documented conda installation commands.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_DIR}/environment.yml"
ENV_NAME="radiomicviz"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available on PATH." >&2
    exit 1
fi

echo "Creating conda environment '${ENV_NAME}' from ${ENV_FILE}"
conda env create --file "${ENV_FILE}"

echo "Installing PyRadiomics 3.0.1 with the environment's pinned build dependencies"
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install "pyradiomics==3.0.1" --no-build-isolation --no-deps

echo "Installing RadiomicsViz from ${REPO_DIR}"
conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install --editable "${REPO_DIR}" --no-build-isolation --no-deps

echo "Checking the installed environment"
conda run --no-capture-output -n "${ENV_NAME}" python -m pip check
conda run --no-capture-output -n "${ENV_NAME}" \
    python -c 'import radiomicviz, radiomics; print("RadiomicsViz", radiomicviz.__version__); print("PyRadiomics", str(radiomics.__version__).lstrip("v"))'

echo
echo "Environment created successfully. Activate it with:"
echo "  conda activate ${ENV_NAME}"
