"""Stage 1-2: Ingest reviews (explicit schema) + dedup + normalize timestamp.

Run on a 1% sample first to validate parsing & get the data shape, then full scale.

  python 10_ingest.py --categories Beauty_and_Personal_Care,Sports_and_Outdoors,Toys_and_Games \
      --sample 0.01               # validate on 1%
  python 10_ingest.py --categories ... --out /data/yizhou/output/reviews_clean.parquet
"""
import argparse
import time
from pyspark.sql import functions as F
from spark_common import build_spark, REVIEW_SCHEMA, review_path

DEFAULT_CATS = "Beauty_and_Personal_Care,Sports_and_Outdoors,Toys_and_Games"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=DEFAULT_CATS)
    ap.add_argument("--sample", type=float, default=1.0, help="row fraction (0-1)")
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--aqe", action="store_true")
    ap.add_argument("--out", default=None, help="parquet output path; if unset, only report")
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    spark = build_spark("ingest", cores=args.cores, aqe=args.aqe)
    t0 = time.time()

    # Read each category with the EXPLICIT schema, tag with category column, union.
    dfs = []
    for c in cats:
        d = spark.read.schema(REVIEW_SCHEMA).json(review_path(c)).withColumn("category", F.lit(c))
        dfs.append(d)
    df = dfs[0]
    for d in dfs[1:]:
        df = df.unionByName(d)

    if args.sample < 1.0:
        df = df.sample(withReplacement=False, fraction=args.sample, seed=42)

    # Keep only rows with the fields we truly need for sequences.
    df = df.filter(F.col("user_id").isNotNull()
                   & F.col("parent_asin").isNotNull()
                   & F.col("timestamp").isNotNull())

    df = df.cache()
    n_raw = df.count()

    # --- Dedup on (user_id, parent_asin, timestamp) (brief stage 2) ---
    df_dedup = df.dropDuplicates(["user_id", "parent_asin", "timestamp"])
    df_dedup = df_dedup.cache()
    n_dedup = df_dedup.count()

    # --- Normalize timestamp: data is 13-digit epoch MILLIS. Verify & keep ms canonical,
    #     also derive event_ts (seconds) for readability. ---
    dgt = df_dedup.select(F.length(F.col("timestamp").cast("string")).alias("d"))
    digit_dist = dgt.groupBy("d").count().orderBy("d").collect()
    df_dedup = df_dedup.withColumn("ts_ms", F.col("timestamp")) \
                       .withColumn("event_ts", (F.col("timestamp") / 1000).cast("timestamp"))

    # --- Metrics ---
    n_users = df_dedup.select("user_id").distinct().count()
    n_items = df_dedup.select("parent_asin").distinct().count()
    ts_range = df_dedup.select(F.min("event_ts").alias("min"), F.max("event_ts").alias("max")).collect()[0]
    per_cat = df_dedup.groupBy("category").count().orderBy("category").collect()

    dt = time.time() - t0
    print("\n================ INGEST REPORT ================")
    print(f"categories        : {cats}")
    print(f"sample fraction   : {args.sample}")
    print(f"raw rows          : {n_raw:,}")
    print(f"after dedup       : {n_dedup:,}  (removed {n_raw - n_dedup:,}, "
          f"{100*(n_raw-n_dedup)/max(n_raw,1):.2f}%)")
    print(f"distinct users    : {n_users:,}")
    print(f"distinct items    : {n_items:,}")
    print(f"avg interactions/user : {n_dedup/max(n_users,1):.2f}")
    print(f"timestamp digits  : {[(r['d'], r['count']) for r in digit_dist]}  (expect all 13 = millis)")
    print(f"time range        : {ts_range['min']}  ->  {ts_range['max']}")
    print("per-category rows :")
    for r in per_cat:
        print(f"    {r['category']:32s} {r['count']:,}")
    print(f"schema            :")
    for f in df_dedup.schema.fields:
        print(f"    {f.name:20s} {f.dataType.simpleString()}")
    print(f"wall time         : {dt:.1f}s")
    print("================================================\n")
    df_dedup.show(5, truncate=40)

    if args.out:
        (df_dedup.select("user_id", "parent_asin", "asin", "rating",
                         "verified_purchase", "helpful_vote", "ts_ms", "event_ts", "category")
         .write.mode("overwrite").partitionBy("category").parquet(args.out))
        print(f"[write] wrote clean parquet -> {args.out}")

    spark.stop()


if __name__ == "__main__":
    main()
