"""Stage 4 + brief §4: Sessionize (per-user chronological sequences) + SKEW measurement.

Skew is a first-class deliverable (brief §4):
  1. Quantify: interactions-per-user and per-item distributions (p50/p95/p99/max)
  2. Feel it: per-partition row counts after the sessionize shuffle (imbalance)

  python 30_sessionize_skew.py --input /data/yizhou/output/reviews_5core.parquet
  python 30_sessionize_skew.py --input ... --aqe        # compare partition skew with AQE
"""
import argparse
from pyspark.sql import functions as F
from pyspark.sql import Window
from spark_common import build_spark


def pct(df, col, qs=(0.5, 0.95, 0.99)):
    row = df.select(
        F.expr(f"percentile_approx({col}, array({','.join(str(q) for q in qs)}), 10000)").alias("p"),
        F.max(col).alias("mx"), F.min(col).alias("mn"), F.avg(col).alias("avg"),
    ).collect()[0]
    return row["p"], row["mn"], row["avg"], row["mx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--aqe", action="store_true")
    ap.add_argument("--shuffle-partitions", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spark = build_spark("sessionize_skew", cores=args.cores, aqe=args.aqe,
                        shuffle_partitions=args.shuffle_partitions)
    df = spark.read.parquet(args.input).cache()
    n = df.count()
    print(f"\n================ SKEW MEASUREMENT (aqe={args.aqe}) ================", flush=True)
    print(f"rows: {n:,}", flush=True)

    # --- 1. interactions per USER ---
    uc = df.groupBy("user_id").count()
    (up, umn, uavg, umx) = pct(uc, "count")
    print(f"\ninteractions/USER : p50={up[0]:.0f}  p95={up[1]:.0f}  p99={up[2]:.0f}  "
          f"max={umx:,}  min={umn:.0f}  avg={uavg:.2f}", flush=True)

    # --- 2. interactions per ITEM (this is where the real power-law bites) ---
    ic = df.groupBy("parent_asin").count()
    (ip, imn, iavg, imx) = pct(ic, "count")
    print(f"interactions/ITEM : p50={ip[0]:.0f}  p95={ip[1]:.0f}  p99={ip[2]:.0f}  "
          f"max={imx:,}  min={imn:.0f}  avg={iavg:.2f}", flush=True)
    # top items = the skew culprits
    top_items = ic.orderBy(F.desc("count")).limit(5).collect()
    print("top-5 hottest items:", [(r['parent_asin'], f"{r['count']:,}") for r in top_items], flush=True)

    # --- 3. Sessionize: per-user chronological sequence via Window (canonical wide shuffle) ---
    w = Window.partitionBy("user_id").orderBy("ts_ms")
    seq = (df.withColumn("pos", F.row_number().over(w))
             .withColumn("seq_len", F.count("*").over(Window.partitionBy("user_id"))))
    seq = seq.cache()
    _ = seq.count()

    # --- 4. Per-partition row skew AFTER a shuffle. The KEY point: skew depends on the
    #        partition key. user_id (sessionize) is balanced (no whale users); parent_asin
    #        (metadata join / item aggregation) is where the power-law bites. Measure both. ---
    def part_skew(key):
        shuffled = df.repartition(args.shuffle_partitions, key)
        part = (shuffled.withColumn("pid", F.spark_partition_id()).groupBy("pid").count())
        (pp, pmn, pavg, pmx) = pct(part, "count", qs=(0.5, 0.95, 0.99))
        sd = part.select(F.stddev("count").alias("sd")).collect()[0]["sd"]
        print(f"\nper-partition rows (repartition by {key} into {args.shuffle_partitions}):", flush=True)
        print(f"  p50={pp[0]:.0f}  p95={pp[1]:.0f}  p99={pp[2]:.0f}  max={pmx:,.0f}  min={pmn:.0f}  "
              f"avg={pavg:.0f}  stddev={sd:.0f}", flush=True)
        print(f"  imbalance (max/avg) = {pmx/max(pavg,1):.2f}x", flush=True)

    part_skew("user_id")       # sessionize key — expect balanced
    part_skew("parent_asin")   # item key — expect skewed (whale items)
    print("================================================================\n", flush=True)

    if args.out:
        seq.write.mode("overwrite").parquet(args.out)
        print(f"[write] wrote sessionized -> {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()
