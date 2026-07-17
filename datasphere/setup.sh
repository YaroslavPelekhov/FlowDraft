#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements-datasphere.txt

if [ ! -d upstream_orthrus/.git ]; then
  git clone https://github.com/chiennv2000/orthrus upstream_orthrus
fi

python -m pip install -e .
