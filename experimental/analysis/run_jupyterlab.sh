#!/bin/bash

PROJECT_DIR=$(dirname $(realpath $0))
set -a; source "${PROJECT_DIR}/../.env"; set +a;
uv run --isolated --locked --project ${PROJECT_DIR}/pyproject.toml jupyter lab