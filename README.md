# Amazon-Reviews Generative Sequential Recommendation

An end-to-end pipeline that turns raw **Amazon Reviews 2023** (McAuley Lab) into a
**generative sequential-recommendation** model, in two connected stages:

1. **PySpark data engineering** — build training-ready user behavior sequences (dedup →
   iterative *k*-core → sessionize → leave-one-out split → joined item text), with the
   distributed-systems behavior *measured* as a first-class deliverable.
2. **TIGER reproduction** — [RQ-VAE Semantic IDs](https://arxiv.org/abs/2305.05065) +
   a seq2seq Transformer that **generates** the next item's Semantic ID (generative retrieval).

Everything runs on a **single workstation** (32-core CPU / 251 GB RAM / one RTX 6000 Ada) —
no cluster, no cloud. Sizing the 40 GB working set against 251 GB RAM, a multi-node cluster
would only add shuffle overhead, so Spark runs in **local mode** on purpose.

> ⚠️ **Paths are hardcoded to the author's environment** (`/data/yizhou/...`). This repo is
> the real, reproducible artifact that produced the numbers below; to re-run, edit the paths
> at the top of each script (or the constants in `spark/spark_common.py`) for your machine.

---

## Results

### Part B — TIGER generative retrieval (leave-one-out test, 20,000 users)

| metric | model | popularity baseline | improvement |
|---|---|---|---|
| Recall@5  | **0.0070** | 0.0023 | **3.0×** |
| NDCG@5    | **0.0046** | 0.0015 | **3.1×** |
| Recall@10 | **0.0100** | 0.0036 | **2.8×** |
| NDCG@10   | **0.0055** | 0.0019 | **2.9×** |

Test Recall@10 rose monotonically 0.0066 → 0.0087 → 0.0100 → 0.0101 across epochs 5/10/15/20
(full curve in [`docs/tiger_training_curve.log`](docs/tiger_training_curve.log)). Absolute
values are modest by design (large/hard 2023 dataset, 3 categories mixed, a compact
reproduction with `sentence-t5-base` + a 17.6M-param model) — the point is a full
generative-retrieval pipeline that works end to end and beats a non-trivial baseline ~3×.

### Part A — PySpark pipeline (measured)

| stage | result |
|---|---|
| Ingest (explicit schema, no inference) | 59.8M raw reviews → 59.1M after dedup; **26 GB → 3.6 GB** Parquet |
| Iterative 5-core, **cached vs uncached** | cached converges in 8 rounds / **237 s**; uncached explodes (round 6 alone **188 s = 15×**, round 7 killed >250 s — lazy DAG re-derivation) |
| 5-core output | 22.0M → 2.39M users, 3.5M → 715K items, avg interactions/user 2.69 → **9.55** |
| Data skew | items power-law (max 13,970 vs median 12 ≈ **1160×**), yet 256-partition row imbalance only **1.05×** (by user) / **1.18×** (by item) → skew depends on the partition key |
| Broadcast vs shuffle join | broadcasting full 3.5M-row metadata **fails** (>1 GB > driver maxResultSize); filter to 715K used items → broadcast **1.37×** faster than SortMergeJoin (confirmed via physical plan) |
| AQE off vs on | 7.0 s vs 8.2 s — no gain (partitions already balanced) |

Full numbers: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Repository layout

```
spark/                  PySpark data-engineering pipeline (Part A)
  spark_common.py       session builder (AQE flag) + explicit schemas
  10_ingest.py          read reviews (explicit schema) + dedup + normalize timestamp
  20_kcore.py           iterative 5-core (--cache toggles the cached-vs-uncached benchmark)
  30_sessionize_skew.py per-user sequences (Window) + skew measurement
  50_join.py            reviews ⋈ item metadata, broadcast vs shuffle, with .explain()
  60_split.py           leave-one-out split + final Parquet + item-metadata table
  00_setup_env.sh, 01_download.sh   env + dataset download
  tutorial/             10-lesson hands-on Spark tutorial (notebook + markdown)
tiger/                  TIGER reproduction (Part B)
  10_prep.py            export item text + user sequences from the Spark output
  20_embed.py           frozen Sentence-T5 item embeddings
  30_rqvae.py           RQ-VAE → per-item Semantic IDs (kmeans init, STE, dead-code revival)
  40_make_tokens.py     map user sequences → Semantic-ID token sequences
  50_train.py           seq2seq (T5) generative retrieval + constrained beam search eval
  60_baseline.py        popularity baseline
  70_eval.py            Recall@K / NDCG@K from a checkpoint
docs/                   BENCHMARKS.md, environment audit, training curve
```

## Reproduce (high level)

```bash
# 1. environment (user-space conda; pyspark 3.5 needs JDK 11, NOT Spark 4.x)
bash spark/00_setup_env.sh
bash spark/01_download.sh

# 2. Spark pipeline (Part A)
python spark/10_ingest.py --out .../reviews_clean.parquet
python spark/20_kcore.py  --input .../reviews_clean.parquet --k 5 --cache --out .../reviews_5core.parquet
python spark/30_sessionize_skew.py --input .../reviews_5core.parquet
python spark/50_join.py   --reviews .../reviews_5core.parquet
python spark/60_split.py  --reviews .../reviews_5core.parquet --out .../interactions_split.parquet --meta-out .../item_meta.parquet

# 3. TIGER (Part B) — GPU env (torch + transformers + sentence-transformers)
python tiger/10_prep.py
python tiger/20_embed.py --model sentence-transformers/sentence-t5-base
python tiger/30_rqvae.py --num-codebooks 3 --codebook-size 256
python tiger/40_make_tokens.py
python tiger/60_baseline.py --topk 10
python tiger/50_train.py --bf16 --epochs 20 --d-model 256 --layers 6 --test-every 5
python tiger/70_eval.py --ckpt tiger/checkpoints/seq2seq_best.pt --ks 5,10
```

Data files (raw JSONL, Parquet, `.npy` embeddings, checkpoints) are **not** committed — see
`.gitignore`. See `docs/spark_env_audit.md` for the exact environment.

## References
- Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER), NeurIPS 2023.
- Hou et al., *Bridging Language and Items for Retrieval and Recommendation* (Amazon Reviews 2023).
