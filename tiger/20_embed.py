"""TIGER Stage-1 prerequisite: precompute frozen Sentence-T5 embeddings for all items.

Encodes each item's text (title+cats+store+desc) -> a 768-d semantic vector with a FROZEN
Sentence-T5 encoder. These vectors x are the input to RQ-VAE (which produces Semantic IDs).

  conda activate /data/yizhou/envs/tiger
  python tiger_20_embed.py --model sentence-transformers/sentence-t5-base --batch 256

Output: /data/yizhou/tiger/data/item_emb.npy  (float32 [N,768])
        /data/yizhou/tiger/data/item_ids.txt  (parent_asin per row, aligned)
"""
import argparse, time, os
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

DATA = "/data/yizhou/tiger/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/sentence-t5-base")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-chars", type=int, default=1000, help="truncate long texts")
    args = ap.parse_args()

    # load item texts (few parquet files) in stable order
    tbl = pq.read_table(f"{DATA}/item_text.parquet")
    asins = tbl.column("parent_asin").to_pylist()
    texts = tbl.column("text").to_pylist()
    texts = [(t or "")[:args.max_chars] for t in texts]
    n = len(texts)
    print(f"[embed] {n:,} items | model={args.model} | batch={args.batch}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=dev)
    dim = model.get_sentence_embedding_dimension()
    print(f"[embed] device={dev} dim={dim}", flush=True)

    emb = np.zeros((n, dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, args.batch):
        chunk = texts[i:i + args.batch]
        v = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=False,
                         show_progress_bar=False)
        emb[i:i + len(chunk)] = v
        if (i // args.batch) % 50 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(f"  {done:>7,}/{n:,}  {rate:6.0f} items/s  eta {eta/60:5.1f} min", flush=True)

    np.save(f"{DATA}/item_emb.npy", emb)
    with open(f"{DATA}/item_ids.txt", "w") as f:
        f.write("\n".join(asins))
    print(f"[embed] DONE {n:,} x {dim} -> {DATA}/item_emb.npy  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
