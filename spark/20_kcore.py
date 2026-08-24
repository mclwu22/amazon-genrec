"""Stage 3: Iterative k-core filtering (brief §3 stage 3 — the highest-risk stage).

Repeatedly drop users with <k interactions, then items with <k interactions, until
the row count stabilizes. Because Spark is lazy, an UNCACHED loop re-derives the whole
DAG from source every round (round 6 recomputes 1-5) — this is the thing we measure.

  python 20_kcore.py --input .../reviews_clean_1pct.parquet --k 5            # uncached
  python 20_kcore.py --input .../reviews_clean_1pct.parquet --k 5 --cache    # cached

Reports per-round row counts, #rounds to converge, and total wall time.
"""
import argparse
import time
from pyspark.sql import functions as F
from spark_common import build_spark

T0 = 0.0  # run start time, set in main(); used for cumulative per-round timing


def kcore(df, k, max_rounds, do_cache):
    # Per round we ONLY count rows — keeping rounds cheap so the uncached-vs-cached
    # comparison reflects purely the cost of re-deriving the growing DAG, not extra
    # distinct-count shuffles. Final distinct users/items are computed once at the end.
    rounds = []
    prev = -1
    for r in range(1, max_rounds + 1):
        # drop users with < k interactions
        keep_u = df.groupBy("user_id").count().filter(F.col("count") >= k).select("user_id")
        df = df.join(keep_u, "user_id", "inner")
        # drop items with < k interactions (this can push some users back below k)
        keep_i = df.groupBy("parent_asin").count().filter(F.col("count") >= k).select("parent_asin")
        df = df.join(keep_i, "parent_asin", "inner")

        if do_cache:
            df = df.cache()

        t = time.time()
        n = df.count()
        rt = time.time() - t
        rounds.append((r, n, rt))
        print(f"  round {r:2d}: rows={n:>12,}  (count {rt:6.1f}s, cumulative {time.time()-T0:6.1f}s)",
              flush=True)
        if n == prev:
            print(f"  --> converged at round {r} (row count stable)", flush=True)
            break
        prev = n
    return df, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument("--cache", action="store_true", help="cache each round (vs re-derive DAG)")
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--aqe", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spark = build_spark("kcore", cores=args.cores, aqe=args.aqe)
    df = spark.read.parquet(args.input)

    global T0
    n0 = df.count()
    print(f"\n================ K-CORE (k={args.k}, cache={args.cache}, aqe={args.aqe}) ================", flush=True)
    print(f"input rows: {n0:,}", flush=True)
    T0 = time.time()
    df_out, rounds = kcore(df, args.k, args.max_rounds, args.cache)
    total = time.time() - T0

    # final distinct users/items computed once (materialize df_out first if cached)
    df_out = df_out.cache()
    final_rows = df_out.count()
    final_users = df_out.select("user_id").distinct().count()
    final_items = df_out.select("parent_asin").distinct().count()

    print("--------------------------------------------------------------", flush=True)
    print(f"rounds to converge : {rounds[-1][0]}")
    print(f"final rows         : {final_rows:,}  ({100*final_rows/max(n0,1):.1f}% of input)")
    print(f"final users        : {final_users:,}")
    print(f"final items        : {final_items:,}")
    print(f"avg interactions/user (post k-core): {final_rows/max(final_users,1):.2f}")
    print(f"per-round timings  : {[(r, f'{rt:.1f}s') for r, _, rt in rounds]}")
    print(f"TOTAL wall time    : {total:.1f}s   (cache={args.cache})")
    print("==============================================================\n", flush=True)

    if args.out:
        df_out.write.mode("overwrite").parquet(args.out)
        print(f"[write] wrote k-core output -> {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()
