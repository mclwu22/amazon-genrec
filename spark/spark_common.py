"""Shared Spark session builder + explicit schemas for the Amazon-Reviews pipeline.

Baseline config follows the project brief §5:
  - local[24]  (leave 8 cores free for the MIND dataloader)
  - driver.memory = 96g (driver == executor in local mode)
  - SPARK_LOCAL_DIRS under /data (never write to /)
  - AQE is a FLAG we flip between runs (default OFF for run 1)
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, BooleanType, ArrayType,
)

DATA_ROOT = "/data/yizhou"
SPARK_TMP = f"{DATA_ROOT}/spark-tmp"


def build_spark(app_name, cores=24, driver_mem="96g", aqe=False, shuffle_partitions=256):
    os.environ["SPARK_LOCAL_DIRS"] = SPARK_TMP
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.memory", driver_mem)
        .config("spark.local.dir", SPARK_TMP)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true" if aqe else "false")
        .config("spark.sql.adaptive.skewJoin.enabled", "true" if aqe else "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"[spark] app={app_name} master=local[{cores}] driver={driver_mem} "
          f"aqe={aqe} shuffle_parts={shuffle_partitions} version={spark.version}")
    return spark


# ---- Explicit schemas (NO inference — brief §3 stage 1). Fields omitted here are
# ---- dropped by Spark at read time, so we never materialize review text/images. ----

# Reviews: keep only what sequential-rec needs. item identity = parent_asin.
REVIEW_SCHEMA = StructType([
    StructField("rating", DoubleType(), True),
    StructField("asin", StringType(), True),
    StructField("parent_asin", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("timestamp", LongType(), True),        # 13-digit epoch MILLIS
    StructField("helpful_vote", LongType(), True),
    StructField("verified_purchase", BooleanType(), True),
])

# Meta: text only (brief §2) — title, description, categories, store + a few numerics.
META_SCHEMA = StructType([
    StructField("parent_asin", StringType(), True),
    StructField("title", StringType(), True),
    StructField("description", ArrayType(StringType()), True),
    StructField("categories", ArrayType(StringType()), True),
    StructField("store", StringType(), True),
    StructField("main_category", StringType(), True),
    StructField("price", StringType(), True),          # messy: numeric-or-null-or-str
    StructField("average_rating", DoubleType(), True),
    StructField("rating_number", LongType(), True),
])

RAW = f"{DATA_ROOT}/raw/raw"


def review_path(category):
    return f"{RAW}/review_categories/{category}.jsonl"


def meta_path(category):
    return f"{RAW}/meta_categories/meta_{category}.jsonl"
