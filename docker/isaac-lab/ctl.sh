#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DOCKER_DIR="${SCRIPT_DIR}/upstream/docker"

if [[ ! -f "${UPSTREAM_DOCKER_DIR}/container.py" ]]; then
  echo "[error] Missing upstream docker tool: ${UPSTREAM_DOCKER_DIR}/container.py" >&2
  exit 1
fi

cd "${UPSTREAM_DOCKER_DIR}"
python3 ./container.py "$@"
