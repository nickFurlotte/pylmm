# Python 2.7 baseline environment for the original (2015) pylmm code.
#
# Pinned to the final CPython 2.7 release and the last numpy/scipy releases
# that support Python 2. No aarch64 cp27 wheels exist for these versions, so
# the image is built and run as linux/amd64 (emulated on Apple Silicon).
#
# NUMPY_VERSION / SCIPY_VERSION are build args so the same file can build a
# "legacy" image (older numpy) for cross-checking; defaults are the baseline.
FROM --platform=linux/amd64 python:2.7.18-slim-buster@sha256:6c1ffdff499e29ea663e6e67c9b6b9a3b401d554d2c9f061f9a45344e3992363

ARG NUMPY_VERSION=1.16.6
ARG SCIPY_VERSION=1.2.3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Single-threaded BLAS for reproducible floating point results.
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN pip install --only-binary=:all: "numpy==${NUMPY_VERSION}" "scipy==${SCIPY_VERSION}"

WORKDIR /repo
# The repo is bind-mounted at /repo by run.sh; make `import pylmm` work.
ENV PYTHONPATH=/repo
CMD ["python", "-c", "import sys, numpy, scipy; print(sys.version); print('numpy', numpy.__version__); print('scipy', scipy.__version__)"]
