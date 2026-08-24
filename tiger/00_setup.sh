#!/usr/bin/env bash
# TIGER env setup on exx (GPU). User-space conda env on the /data disk. Logs everything.
set -euo pipefail

BASE=/data/yizhou
MC=$BASE/miniconda
ENVDIR=$BASE/envs/tiger

echo "[$(date '+%F %T')] === dirs ==="
mkdir -p "$BASE"/tiger/{data,checkpoints,semantic_ids,logs} "$BASE"/tiger

export PATH="$MC/bin:$PATH"
source "$MC/etc/profile.d/conda.sh"

echo "[$(date '+%F %T')] === create env (python 3.10, conda-forge) ==="
if [ ! -x "$ENVDIR/bin/python" ]; then
  conda create -y -p "$ENVDIR" --override-channels -c conda-forge python=3.10
fi
conda activate "$ENVDIR"

echo "[$(date '+%F %T')] === pip install torch (cu121) + libs ==="
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  "transformers>=4.44" \
  "sentence-transformers>=3.0" \
  "pyarrow>=15" "pandas>=2" "numpy<2" \
  "scikit-learn>=1.3" \
  tqdm

echo "[$(date '+%F %T')] === versions / GPU visible? ==="
python - <<'PY'
import torch, transformers, sentence_transformers, sklearn, numpy
print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("transformers", transformers.__version__, "| sentence-transformers", sentence_transformers.__version__)
print("numpy", numpy.__version__, "| sklearn", sklearn.__version__)
PY

echo "[$(date '+%F %T')] === SETUP DONE ==="
