# Amazon-Reviews-2023 → Sequential-Recommendation Dataset (PySpark)

A PySpark pipeline that turns raw Amazon Reviews 2023 (McAuley Lab) into training-ready
data for **sequential recommendation** — per-user chronological behavior sequences with
leave-one-out splits and joined item text metadata. The output is the direct input for a
follow-on **TIGER** (RQ-VAE Semantic IDs + generative retrieval) reproduction.

The point of this repo is not "a cleaned dataset." It is a **measured** pipeline: the
shuffle behavior, data skew, and caching/join trade-offs are instrumented and recorded in
[`BENCHMARKS.md`](BENCHMARKS.md). See that file for all numbers.

## Why local mode (the key design decision)

The three candidate categories are ~26 GB of review JSONL / 59.8M rows. The machine (`exx`)
has **251 GB RAM**. The working set fits in memory many times over.

> I sized the data against available memory, concluded a multi-node cluster would add
> cross-machine shuffle overhead for no benefit, and ran **local mode** instead — then spent
> the time on the things that actually bind: caching an iterative loop, join strategy, and
> skew. A cluster here would be cargo-culting.

Concretely, a distributed cluster was rejected because: (a) 40 GB of working data fits in
250 GB RAM, so any cross-node shuffle is a net loss; (b) both candidate boxes have `ufw`
active, no inter-host name resolution, and no sudo; (c) the interesting engineering
(partition tuning, skew, cache) is identical in local mode and is where the real bottleneck
was. Choosing local mode *for a stated, quantified reason* is the better engineering story
than standing up a cluster that isn't warranted.

## Environment

| | |
|---|---|
| Machine | `exx` — Threadripper 5955WX, 32 vCPU, 251 GB RAM |
| Storage | local 19 TB ext4 `/data` (separate partition; never write to `/`) |
| Java | OpenJDK 11 → **pyspark 3.5.9** (NOT Spark 4.x, which needs JDK 17/21) |
| Python | 3.10 (user-space Miniconda env at `/data/yizhou/envs/spark`) |
| Spark | `local[24]` (8 cores reserved), `driver.memory=96g`, `SPARK_LOCAL_DIRS=/data/yizhou/spark-tmp` |

## Pipeline stages

| # | script | what |
|---|---|---|
| 1–2 | `10_ingest.py` | Read reviews with an **explicit schema** (no inference; drops review text/images → 26 GB→3.6 GB), dedup on (user, item, timestamp), normalize timestamp (ms). |
| 3 | `20_kcore.py` | Iterative **5-core** filtering. `--cache` flag toggles the cached-vs-uncached comparison. |
| 4 | `30_sessionize_skew.py` | Per-user chronological sequences (Window) + **skew measurement** (per-user/item distributions, per-partition imbalance). |
| 5 | `50_join.py` | Join reviews ⋈ item metadata, **broadcast vs shuffle**, with `.explain()`. |
| 6–7 | `60_split.py` | **Leave-one-out** split + write final Parquet + item-metadata table. `--aqe` flag for the AQE comparison. |

`spark_common.py` holds the session builder (AQE is a flag) and the explicit schemas.

## Reproduce

```bash
source /data/yizhou/miniconda/etc/profile.d/conda.sh && conda activate /data/yizhou/envs/spark
cd /data/yizhou/repo
python -u 10_ingest.py --out /data/yizhou/output/reviews_clean.parquet
python -u 20_kcore.py  --input /data/yizhou/output/reviews_clean.parquet --k 5 --cache \
                       --out /data/yizhou/output/reviews_5core.parquet
python -u 30_sessionize_skew.py --input /data/yizhou/output/reviews_5core.parquet
python -u 50_join.py  --reviews /data/yizhou/output/reviews_5core.parquet
python -u 60_split.py --reviews /data/yizhou/output/reviews_5core.parquet \
                      --out /data/yizhou/output/interactions_split.parquet \
                      --meta-out /data/yizhou/output/item_meta.parquet
```

Everything is parameterized: `--categories`, `--k`, `--cache`, `--aqe`, `--sample`, `--cores`.

## Headline results (full numbers in BENCHMARKS.md)

- **Ingest**: 59.77M raw → 59.14M after dedup (−1.06%); explicit schema shrinks 26 GB → **3.6 GB**.
- **k-core cached vs uncached**: cached converges in 8 rounds, **237 s**; uncached explodes
  (round 6 alone 188 s = 15× cached, round 7 killed >250 s) because a lazy iterative loop
  re-derives the whole DAG each round. **Biggest wall-time lever in the pipeline.**
- **5-core**: 22.82M interactions kept; users 22.0M→2.39M (−89% long tail), avg
  interactions/user 2.69 → **9.55**.
- **Skew**: items are power-law (max 13,970 vs median 12 ≈ 1160×), but at 256 partitions the
  *row-level* partition imbalance is mild (user 1.05×, item 1.18×) — skew depends on the
  partition key, and no single key is bigger than a partition.
- **Join**: broadcasting the full 3.5M-row metadata **fails** (>1 GB > maxResultSize — it's
  not a small table); filtering to the 715K used items first makes broadcast viable and
  **1.37× faster** than SortMergeJoin.
- **AQE**: off 7.0 s vs on 8.2 s — no win, because partitions are already balanced.
