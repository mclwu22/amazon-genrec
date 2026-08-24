"""Stage 6-7: Leave-one-out split + write final Parquet (the TIGER handoff artifact).

Per user, chronological: last interaction -> test, second-to-last -> valid, rest -> train.
Also writes the filtered item-metadata table (text for RQ-VAE). Timed + AQE-flag so the
same job doubles as the AQE off-vs-on comparison (the Window shuffle is by user_id).

  python 60_split.py --reviews .../reviews_5core.parquet --meta-out .../item_meta.parquet \
      --out /data/yizhou/output/interactions_split.parquet
  python 60_split.py --reviews ... --aqe        # AQE-on timing
"""
import argparse
import time
from pyspark.sql import functions as F
from pyspark.sql import Window
from spark_common import build_spark, META_SCHEMA, meta_path

DEFAULT_CATS = "Beauty_and_Personal_Care,Sports_and_Outdoors,Toys_and_Games"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--categories", default=DEFAULT_CATS)
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--aqe", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta-out", default=None)
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    spark = build_spark("split", cores=args.cores, aqe=args.aqe)
    df = spark.read.parquet(args.reviews)

    # chronological rank per user; tie-break by parent_asin for determinism.
    w_asc = Window.partitionBy("user_id").orderBy(F.col("ts_ms").asc(), F.col("parent_asin").asc())
    w_desc = Window.partitionBy("user_id").orderBy(F.col("ts_ms").desc(), F.col("parent_asin").desc())
    labeled = (df.withColumn("pos", F.row_number().over(w_asc))
                 .withColumn("rev_pos", F.row_number().over(w_desc))
                 .withColumn("split", F.when(F.col("rev_pos") == 1, "test")
                                       .when(F.col("rev_pos") == 2, "valid")
                                       .otherwise("train")))
    labeled = labeled.select("user_id", "parent_asin", "rating", "ts_ms", "event_ts",
                             "category", "pos", "rev_pos", "split")

    print(f"\n================ LOO SPLIT (aqe={args.aqe}) ================", flush=True)
    t0 = time.time()
    counts = labeled.groupBy("split").count().collect()
    cmap = {r["split"]: r["count"] for r in counts}
    n_users = labeled.select("user_id").distinct().count()
    t_compute = time.time() - t0
    print(f"train rows : {cmap.get('train',0):,}", flush=True)
    print(f"valid rows : {cmap.get('valid',0):,}  (== #users? {n_users:,})", flush=True)
    print(f"test  rows : {cmap.get('test',0):,}  (== #users? {n_users:,})", flush=True)
    print(f"users      : {n_users:,}", flush=True)
    print(f"compute (window+counts) wall={t_compute:.1f}s  (aqe={args.aqe})", flush=True)

    if args.out:
        tw = time.time()
        labeled.write.mode("overwrite").partitionBy("split").parquet(args.out)
        print(f"[write] interactions_split -> {args.out}  ({time.time()-tw:.1f}s)", flush=True)

    if args.meta_out:
        used = df.select("parent_asin").distinct()
        mdfs = [spark.read.schema(META_SCHEMA).json(meta_path(c)) for c in cats]
        meta = mdfs[0]
        for d in mdfs[1:]:
            meta = meta.unionByName(d)
        meta = (meta.filter(F.col("parent_asin").isNotNull())
                    .dropDuplicates(["parent_asin"])
                    .join(used, "parent_asin", "inner")
                    .select("parent_asin", "title", "description", "categories", "store",
                            "main_category", "price", "average_rating", "rating_number"))
        tw = time.time()
        meta.write.mode("overwrite").parquet(args.meta_out)
        print(f"[write] item_meta -> {args.meta_out}  ({time.time()-tw:.1f}s)", flush=True)

    print("==========================================================\n", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
