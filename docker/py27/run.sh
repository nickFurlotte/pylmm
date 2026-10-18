#!/usr/bin/env bash
# Run a command inside the pylmm Python 2.7 baseline container with the repo
# mounted at /repo (the working directory).
#
#   docker/py27/run.sh                          # print python/numpy/scipy versions
#   docker/py27/run.sh python scripts/example.py
#   docker/py27/run.sh --build                  # (re)build the image first
#
# Env: PYLMM_PY27_IMAGE overrides the image tag (default pylmm-py27:baseline).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
IMAGE="${PYLMM_PY27_IMAGE:-pylmm-py27:baseline}"

if [[ "${1:-}" == "--build" ]]; then
  shift
  docker build --platform linux/amd64 -t "${IMAGE}" "${HERE}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker build --platform linux/amd64 -t "${IMAGE}" "${HERE}"
fi

TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then TTY_FLAGS=(-it); fi

exec docker run --rm ${TTY_FLAGS[@]+"${TTY_FLAGS[@]}"} --platform linux/amd64 \
  -v "${REPO}:/repo" -w /repo "${IMAGE}" "$@"
