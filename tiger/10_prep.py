"""TIGER data prep: export item text corpus + user sequences from the Spark parquet.

Reads the handoff artifacts written by the Spark project and produces PyTorch-friendly files:
  - item_text.parquet   : (parent_asin, text)   one row per item; text = title+desc+cats+store
  - sequences.jsonl     : one line per user: {"user_id","items":[...ordered parent_asin...],
                          "n_train","n":total}  (train = all but last 2; valid=last-1; test=last)

Run in the SPARK env (it reads parquet with pyspark):
  conda activate /data/yizhou/envs/spark
  python tiger_10_prep.py
"""
import os, json, time
os.environ["SPARK_LOCAL_DIRS"] = "/data/yizhou/spark-tmp"
from pyspark.sql import SparkSession, functions as F, Window

OUT = "/data/yizhou/tiger/data"
META = "/data/yizhou/output/item_meta.parquet"
INTER = "/data/yizhou/output/interactions_split.parquet"

spark = (SparkSession.builder.master("local[16]").appName("tiger_prep")
         .config("spark.driver.memory", "64g")
         .config("spark.local.dir", "/data/yizhou/spark-tmp")
         .config("spark.sql.shuffle.partitions", "128")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
t0 = time.time()

# ---------- 1. item text corpus ----------
meta = spark.read.parquet(META)
# build a single text field: title. description(join). categories(join). store
def arr_join(col):
    return F.concat_ws(" ", F.col(col))
text = (meta
    .withColumn("desc_s", arr_join("description"))
    .withColumn("cats_s", arr_join("categories"))
    .withColumn("text", F.concat_ws(". ",
        F.coalesce(F.col("title"), F.lit("")),
        F.coalesce(F.col("cats_s"), F.lit("")),
        F.coalesce(F.col("store"), F.lit("")),
        F.coalesce(F.col("desc_s"), F.lit("")),
    ))
    .select("parent_asin", "text"))
# coalesce to few files for fast sequential read in torch
text.coalesce(8).write.mode("overwrite").parquet(f"{OUT}/item_text.parquet")
n_items = text.count()
print(f"[items] wrote {n_items:,} item texts -> {OUT}/item_text.parquet", flush=True)
text.show(3, truncate=90)

# ---------- 2. user sequences ----------
inter = spark.read.parquet(INTER)
w_asc = Window.partitionBy("user_id").orderBy(F.col("ts_ms").asc(), F.col("parent_asin").asc())
ordered = (inter.withColumn("pos", F.row_number().over(w_asc))
                 .select("user_id", "parent_asin", "pos", "ts_ms"))
# collect ordered item list per user
seqs = (ordered.groupBy("user_id")
        .agg(F.sort_array(F.collect_list(F.struct("pos", "parent_asin"))).alias("s"))
        .withColumn("items", F.expr("transform(s, x -> x.parent_asin)"))
        .withColumn("n", F.size("items"))
        .withColumn("n_train", F.col("n") - 2)   # last2 = valid,test
        .select("user_id", "items", "n", "n_train"))

# write as jsonl (one user per line) via toLocalIterator to avoid a giant driver collect
outpath = f"{OUT}/sequences.jsonl"
cnt = 0
with open(outpath, "w") as f:
    for r in seqs.toLocalIterator():
        f.write(json.dumps({"user_id": r["user_id"], "items": list(r["items"]),
                            "n": r["n"], "n_train": r["n_train"]}) + "\n")
        cnt += 1
print(f"[seqs] wrote {cnt:,} user sequences -> {outpath}", flush=True)
print(f"[done] wall={time.time()-t0:.1f}s", flush=True)
spark.stop()
