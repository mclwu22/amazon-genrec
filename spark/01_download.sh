#!/usr/bin/env bash
# Download 3 categories (reviews + meta) from Amazon-Reviews-2023 to /data/yizhou/raw
set -euo pipefail
source /data/yizhou/miniconda/etc/profile.d/conda.sh
conda activate /data/yizhou/envs/spark

export HF_HUB_ENABLE_HF_TRANSFER=1
REPO=McAuley-Lab/Amazon-Reviews-2023
DEST=/data/yizhou/raw

CATS=(Beauty_and_Personal_Care Sports_and_Outdoors Toys_and_Games)

echo "[$(date '+%F %T')] === START download to $DEST ==="
for c in "${CATS[@]}"; do
  echo "[$(date '+%F %T')] --- $c reviews ---"
  hf download "$REPO" --repo-type dataset \
    --include "raw/review_categories/${c}.jsonl" \
    --local-dir "$DEST"
  echo "[$(date '+%F %T')] --- $c meta ---"
  hf download "$REPO" --repo-type dataset \
    --include "raw/meta_categories/meta_${c}.jsonl" \
    --local-dir "$DEST"
done

echo "[$(date '+%F %T')] === download sizes ==="
du -sh "$DEST"/raw/review_categories/*.jsonl "$DEST"/raw/meta_categories/*.jsonl 2>/dev/null
echo "[$(date '+%F %T')] === total ==="
du -sh "$DEST"
echo "[$(date '+%F %T')] === DOWNLOAD DONE ==="
