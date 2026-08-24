"""Stage 5: Join reviews (large) with item metadata (small) — broadcast vs shuffle.

Brief §3 stage 5: measure WITH and WITHOUT the broadcast hint, and inspect the physical
plan to confirm BroadcastHashJoin vs SortMergeJoin.

  python 50_join.py --reviews /data/yizhou/output/reviews_5core.parquet \
      --out /data/yizhou/output/reviews_joined.parquet
"""
import argparse
import time
from pyspark.sql import functions as F
from spark_common import build_spark, META_SCHEMA, meta_path

DEFAULT_CATS = "Beauty_and_Personal_Care,Sports_and_Outdoors,Toys_and_Games"


def plan_join_type(df):
    plan = df._jdf.queryExecution().executedPlan().toString()
    for line in plan.splitlines():
        if "Join" in line:
            return line.strip()[:100]
    return "(no join node found)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--categories", default=DEFAULT_CATS)
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--aqe", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    spark = build_spark("join", cores=args.cores, aqe=args.aqe)
    reviews = spark.read.parquet(args.reviews).cache()
    n_rev = reviews.count()

    # --- ingest meta (small table): explicit schema, dedup to one row per item ---
    mdfs = []
    for c in cats:
        mdfs.append(spark.read.schema(META_SCHEMA).json(meta_path(c)))
    meta = mdfs[0]
    for d in mdfs[1:]:
        meta = meta.unionByName(d)
    meta = (meta.filter(F.col("parent_asin").isNotNull())
                .dropDuplicates(["parent_asin"])
                .select("parent_asin", "title", "description", "categories", "store",
                        "main_category", "price", "average_rating", "rating_number"))
    meta = meta.cache()
    n_meta = meta.count()
    print(f"\n================ JOIN (aqe={args.aqe}) ================", flush=True)
    print(f"reviews rows : {n_rev:,}")
    print(f"meta rows    : {n_meta:,}  (one per item)", flush=True)

    # Materialize the FULL joined wide rows (all text columns) via the noop sink, so timing
    # reflects the real join cost — not a count that the optimizer prunes to just the key.
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")  # only our explicit hint decides

    def timed_materialize(j, label):
        try:
            print(f"\n[{label}] join type:", plan_join_type(j), flush=True)
            t = time.time()
            j.write.format("noop").mode("overwrite").save()
            dt = time.time() - t
            print(f"[{label}] materialized full rows  wall={dt:.1f}s", flush=True)
            return dt
        except Exception as e:
            print(f"[{label}] FAILED: {str(e)[:160]}", flush=True)
            return None

    # (A) SHUFFLE join on FULL meta -> SortMergeJoin
    t_shuffle = timed_materialize(reviews.join(meta, "parent_asin", "left"), "shuffle-fullmeta")
    # (B) BROADCAST FULL meta -> expected to FAIL (meta is not actually small: >1GB serialized)
    t_bcast_full = timed_materialize(reviews.join(F.broadcast(meta), "parent_asin", "left"),
                                     "broadcast-fullmeta")

    # (C) Filter meta to only the items present in the 5-core reviews (the realistic step),
    #     which makes the build side genuinely small -> broadcast becomes viable.
    used = reviews.select("parent_asin").distinct()
    meta_small = meta.join(used, "parent_asin", "inner").cache()
    n_small = meta_small.count()
    print(f"\nfiltered meta rows (only 5-core items): {n_small:,}  (from {n_meta:,})", flush=True)
    t_bcast_small = timed_materialize(reviews.join(F.broadcast(meta_small), "parent_asin", "left"),
                                      "broadcast-smallmeta")

    def fmt(x):
        return f"{x:.1f}s" if x is not None else "FAILED"
    print(f"\nSUMMARY:", flush=True)
    print(f"  shuffle (SortMergeJoin, full meta)   : {fmt(t_shuffle)}", flush=True)
    print(f"  broadcast full meta ({n_meta:,} rows): {fmt(t_bcast_full)}", flush=True)
    print(f"  broadcast small meta ({n_small:,} rows): {fmt(t_bcast_small)}", flush=True)
    if t_shuffle and t_bcast_small:
        print(f"  broadcast(small) vs shuffle speedup  : {t_shuffle/max(t_bcast_small,0.1):.2f}x", flush=True)
    print("======================================================\n", flush=True)

    if args.out:
        (reviews.join(F.broadcast(meta), "parent_asin", "left")
         .write.mode("overwrite").partitionBy("category").parquet(args.out))
        print(f"[write] wrote joined -> {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()
