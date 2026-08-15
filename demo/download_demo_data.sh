#!/usr/bin/env bash
# Download the ESA public PlanetScope sample and crop the demo tile used by
# demo/configs/inf_demo.py.
#
# NOTE: The original satellite image belongs to PlanetScope (ESA sample data)
# and is NOT redistributed by this repository. See:
# https://earth.esa.int/eogateway/missions/planetscope/sample-data
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${REPO_ROOT}/demo"
DATA_DIR="${DEMO_DIR}/data"
TMP_DIR="${DEMO_DIR}/tmp_download"

ESA_ZIP_URL="https://earth.esa.int/eogateway/ftp/missions/sample-data/third-party-missions/planetscope/PSScene_Visual.zip"
TILE_NAME="saint_martin_de_crau_town.tif"

mkdir -p "${DATA_DIR}" "${TMP_DIR}"

if [ -f "${DATA_DIR}/${TILE_NAME}" ]; then
  echo "Demo tile already exists: ${DATA_DIR}/${TILE_NAME}"
  exit 0
fi

echo "Downloading ESA PlanetScope Visual sample (~280 MB)..."
curl -L "${ESA_ZIP_URL}" -o "${TMP_DIR}/PSScene_Visual.zip"

echo "Extracting..."
unzip -o -q "${TMP_DIR}/PSScene_Visual.zip" -d "${TMP_DIR}"

SCENE="$(find "${TMP_DIR}" -name '*_3B_Visual.tif' | head -n1)"
if [ -z "${SCENE}" ]; then
  echo "ERROR: could not locate the PlanetScope visual tif in the download." >&2
  exit 1
fi

echo "Cropping 2048x2048 tile around Saint-Martin-de-Crau..."
python "${DEMO_DIR}/crop_town.py" "${SCENE}" "${DATA_DIR}/${TILE_NAME}"

rm -rf "${TMP_DIR}"
echo "Done: ${DATA_DIR}/${TILE_NAME}"
