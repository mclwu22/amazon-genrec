#!/usr/bin/env bash
# Overnight setup on exx (M2) via user-space Miniconda (no sudo needed). Logs everything.
set -euo pipefail

BASE=/data/yizhou
MC=$BASE/miniconda
ENVDIR=$BASE/envs/spark

echo "[$(date '+%F %T')] === creating dirs ==="
mkdir -p "$BASE"/{raw,logs,spark-tmp,output,repo,envs}

echo "[$(date '+%F %T')] === installing Miniconda (user-space) at $MC ==="
if [ ! -x "$MC/bin/conda" ]; then
  cd "$BASE"
  curl -fsSL -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash miniconda.sh -b -p "$MC"
  rm -f miniconda.sh
fi
export PATH="$MC/bin:$PATH"
conda --version

echo "[$(date '+%F %T')] === creating env (python 3.10, conda-forge) at $ENVDIR ==="
if [ ! -x "$ENVDIR/bin/python" ]; then
  conda create -y -p "$ENVDIR" --override-channels -c conda-forge python=3.10
fi
# shellcheck disable=SC1091
source "$MC/etc/profile.d/conda.sh"
conda activate "$ENVDIR"

echo "[$(date '+%F %T')] === installing pinned packages ==="
python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  "pyspark==3.5.*" \
  "huggingface_hub[hf_transfer]" \
  "pyarrow>=15" \
  "pandas>=2"

echo "[$(date '+%F %T')] === versions ==="
python -c "import sys,pyspark,huggingface_hub,pyarrow,pandas; print('python',sys.version.split()[0]); print('pyspark',pyspark.__version__); print('huggingface_hub',huggingface_hub.__version__); print('pyarrow',pyarrow.__version__); print('pandas',pandas.__version__)"
java -version 2>&1 | head -1

echo "[$(date '+%F %T')] === SETUP DONE ==="
