# BENCHMARKS — PySpark Amazon-Reviews-2023 Sequential-Rec Pipeline

Machine: **exx** (Threadripper 5955WX, 32 vCPU / 251G RAM), OpenJDK 11, pyspark 3.5.9,
local mode `local[24]`, `spark.driver.memory=96g`, `spark.sql.shuffle.partitions=256`,
all I/O on the local 19T `/data` ext4 volume. Categories: Beauty_and_Personal_Care,
Sports_and_Outdoors, Toys_and_Games.

## 0. Ingest (explicit schema, no inference)

| metric | value |
|---|---|
| raw JSONL (3 categories, reviews) | ~26 GB |
| raw rows | 59,766,966 |
| after dedup on (user_id, parent_asin, timestamp) | 59,135,240 (−1.06%) |
| distinct users | 22,014,396 |
| distinct items | 3,506,617 |
| avg interactions/user | 2.69 |
| clean Parquet size | **3.6 GB** (26 GB → 3.6 GB by dropping review text/title/images) |
| wall time (parse + dedup + write, page-cache warm) | 67 s |

Explicit schemas mean fields not needed for sequences (review `text`, `title`, `images`)
are never materialized — this is the 26 GB → 3.6 GB reduction and it also skips the extra
full pass that schema inference would cost on tens of GB.

## 1. k-core (5-core) — cached vs uncached  ← headline result

Iterative 5-core: repeatedly drop users <5 interactions, then items <5, until the row
count stabilizes. Same input (59,135,240 rows), same k=5, `local[24]`, AQE off.
**Per round we count rows only** (distinct users/items computed once at the end), so the
timing difference reflects *purely* the cost of re-deriving the growing DAG.

Both runs produce identical per-round row counts (correctness is the same) — only the
timing differs:

| round | rows | **cached** count time | **uncached** count time |
|---|---|---|---|
| 1 | 24,633,891 | 58.5 s (first materialize) | 32.0 s |
| 2 | 22,940,366 | 12.4 s | 20.6 s |
| 3 | 22,829,476 | 11.0 s | 27.5 s |
| 4 | 22,821,246 | 11.1 s | 35.7 s |
| 5 | 22,820,668 | 11.4 s | 46.9 s |
| 6 | 22,820,553 | 12.3 s | **187.8 s** |
| 7 | 22,820,549 | 17.4 s | **killed (>250 s, incomplete)** |
| 8 | 22,820,549 (converged) | 29.6 s | — |

- **Cached: converged in 8 rounds, TOTAL 237 s**, per-round roughly flat (~11–30 s).
- **Uncached: exploding.** By round 6 a single count takes 187.8 s (vs 12.3 s cached =
  **15×**); cumulative through only 6 rounds is **354 s — already more than the cached
  run's full 8-round total**. Round 7 was killed after >250 s (it re-derives 6 prior
  rounds of groupBy+join from the Parquet source every time).
- Takeaway: because Spark is lazy, an uncached iterative loop re-computes the entire
  lineage each round (round *r* redoes rounds 1..*r*). `.cache()` after each round cuts
  this from super-linear to flat. This is the single biggest wall-time lever in the pipeline.

### 5-core output (the retained "core")

| metric | before (ingest) | after 5-core | change |
|---|---|---|---|
| interactions | 59,135,240 | 22,820,549 | 38.6% kept |
| users | 22,014,396 | 2,390,683 | −89% (1–2 interaction long tail dropped) |
| items | 3,506,617 | 715,729 | −80% |
| avg interactions/user | 2.69 | **9.55** | sequences now meaningful |

## 2. Skew measurement (per-user / per-item, partition imbalance)

Measured on the 22.82M-row 5-core dataset, AQE off, `local[24]`, 256 shuffle partitions.

### Interaction distributions (the power-law)

| distribution | p50 | p95 | p99 | max | avg | min |
|---|---|---|---|---|---|---|
| interactions / **user** | 7 | 21 | 42 | **3,772** | 9.55 | 5 |
| interactions / **item** | 12 | 107 | 345 | **13,970** | 31.88 | 5 |

Hottest items: `B0BVGHXZJ1` 13,970 · `B01LSUQSB0` 11,762 · `B0B6QVGZ4X` 9,763 · … .
The item side is far heavier-tailed: max/median ≈ **13,970 / 12 ≈ 1160×**, vs the user
side's 3,772 / 7 ≈ 540×.

### Per-partition row imbalance — skew depends on the partition key

| repartition key | p50 | max | min | stddev | **imbalance (max/avg)** |
|---|---|---|---|---|---|
| `user_id` (sessionize key) | 89,041 | 93,625 | 85,704 | 1,545 | **1.05×** |
| `parent_asin` (item/join key) | 88,840 | 104,981 | 73,387 | 6,173 | **1.18×** |

**Finding (more precise than "e-commerce data is skewed"):**
- Partitioning by **user_id** (what sessionize does) is essentially balanced — no user is
  a whale relative to 22.8M rows (biggest is 3,772 = 0.017%). Sessionize will NOT straggle.
- Partitioning by **parent_asin** is measurably worse (stddev 4× higher, 1.18×) but still
  mild, because even the hottest item (13,970 rows) is smaller than one ~89K-row partition.
- Catastrophic stragglers (10×+) require a single key **larger than a partition**, or a
  **skew-amplifying join** (hot key × many matches). At 256 partitions over 23M rows,
  neither occurs — so the honest conclusion is that AQE/salting buys little *here*, and the
  win would appear at higher scale or lower partition counts. (This is exactly the kind of
  measured, quantified claim the project is meant to earn.)

## 3. Broadcast vs shuffle join (reviews ⋈ item metadata)

Join reviews (22.82M rows) ⋈ item metadata on `parent_asin`. Timing uses the `noop` sink
to materialize the **full wide rows** (all text columns) — a `.count()` here is misleading
because the optimizer prunes a left-join count down to just the key column (that gave a
bogus 1.2s "SortMergeJoin"). AQE off, auto-broadcast disabled so only the explicit hint decides.

| join | physical plan | result |
|---|---|---|
| shuffle, full meta (3,507,209 rows) | `SortMergeJoin ... LeftOuter` | **4.4 s** |
| **broadcast, full meta** (3,507,209 rows) | `BroadcastHashJoin ... BuildRight` | **FAILED** |
| broadcast, filtered meta (715,729 rows) | `BroadcastHashJoin ... BuildRight` | **3.2 s** |

**Finding (contradicts the "metadata is a small table, just broadcast it" assumption):**
- Broadcasting the full metadata table **fails**: `Total size of serialized results
  (>1 GiB) is bigger than spark.driver.maxResultSize (1024 MiB)`. 3.5M items with
  `title` + `description` (array) + `categories` (array) is **not** a small table — the
  build side must be collected to the driver, and it blows the 1 GB result cap.
- The correct fix is a pipeline step, not a config bump: **filter meta to the 715,729 items
  that actually survive 5-core** (an inner join against the distinct review items). That build
  side is small enough to broadcast, and broadcast then beats SortMergeJoin **1.37×**
  (3.2 s vs 4.4 s) by avoiding the shuffle+sort of the 22.8M-row left side.
- `.explain()` confirms the plan flips from `SortMergeJoin` → `BroadcastHashJoin (BuildRight)`.

## 4. AQE off vs on

Same job — leave-one-out split (two `Window.partitionBy(user_id)` shuffles + per-split
counts over 22.82M rows) — run with `spark.sql.adaptive.enabled` false then true.

| AQE | compute wall (window + counts) |
|---|---|
| off | **7.0 s** |
| on  | **8.2 s** |

**Finding:** AQE is slightly *slower* here, not faster. This is the expected result given
§2: our partitions are already balanced (user_id imbalance 1.05×), so AQE's two main levers
— coalescing post-shuffle partitions and splitting skewed join partitions — have nothing to
fix, and the extra runtime planning is pure overhead. AQE earns its keep when there IS skew
(a hot join key, wildly uneven partition sizes) or when the static `shuffle.partitions=256`
is badly mis-sized for the stage; neither holds at this scale. The honest takeaway is that
turning AQE on is not a free win — it paid off nowhere our data wasn't already balanced.

## 5. Final artifacts (TIGER handoff)

Leave-one-out split (last→test, 2nd-last→valid, rest→train), validated:

| split | rows |
|---|---|
| train | 18,039,183 |
| valid | 2,390,683 (= #users ✓) |
| test | 2,390,683 (= #users ✓) |

Written under `/data/yizhou/output/`:
- `interactions_split.parquet` (partitioned by split) — user_id, parent_asin, rating, ts_ms,
  event_ts, category, pos, rev_pos, split.
- `item_meta.parquet` — 715,729 items × {title, description, categories, store, main_category,
  price, average_rating, rating_number} — the text side for RQ-VAE Semantic IDs.
- `reviews_clean.parquet` (59.1M) and `reviews_5core.parquet` (22.82M) — intermediate reuse.
