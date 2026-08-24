# Spark 上手教程 (hands-on, 基于本项目真实数据)

目标: 你亲手敲完这 10 课, 能独立读写 Spark、理解惰性求值/shuffle/cache/分区/join/window,
并能对着 Spark UI 讲清楚发生了什么。全部用我们 `/data/yizhou/output/` 的真实数据。

## 怎么上手 (两种方式)

**方式 A — 交互式 shell (推荐, 状态保持, 最适合学)。** 在你自己的终端里:
```bash
ssh exx-server
source /data/yizhou/miniconda/etc/profile.d/conda.sh && conda activate /data/yizhou/envs/spark
pyspark --master 'local[4]' --driver-memory 8g
```
> 学习用 `local[4]` + 8g 就够, 起得快、不占满机器。shell 会自动给你两个对象:
> `spark` (SparkSession) 和 `sc` (SparkContext)。UI 在 exx 的 `http://localhost:4040`
> (要在自己浏览器看就另开一个: `ssh -L 4040:localhost:4040 exx-server`)。

**方式 B — 每课存成脚本跑** `python lessonN.py` (每次重启 SparkSession, 慢一点但可复现)。

下面每课结构: **概念 → 敲这个 → 看什么 → 为什么重要 / 面试怎么说 / 对应我们项目哪一步**。

---

## Lesson 0 — 心智模型: driver / executor / partition / task

Spark 把一份大数据切成很多 **partition (分区)**; 每个 partition 上的一段计算叫一个
**task**; **executor** 是干活的进程 (跑 task), **driver** 是指挥的进程 (你的程序)。

```python
print(sc.master)                 # local[4]  → 本地模式, 4 个线程当 executor
print(sc.defaultParallelism)     # 4
df = spark.read.parquet("/data/yizhou/output/reviews_clean.parquet")
print(df.rdd.getNumPartitions()) # 这份数据被切成多少块
```
**看什么**: 分区数 = Spark 能同时跑多少个 task。
**面试**: "本地模式下 driver 就是 executor, 同一个 JVM; 所以我把 driver.memory 设成 96g,
因为它既当指挥又当干活的。" ← 这正是我们 §5 的配置理由。

---

## Lesson 1 — DataFrame 基础: 读、看 schema、看数据

```python
df = spark.read.parquet("/data/yizhou/output/reviews_clean.parquet")
df.printSchema()          # 列名+类型
df.show(5, truncate=40)   # 看前 5 行
len(df.columns), df.columns
```
**概念**: DataFrame = 带 schema 的分布式表 (像 pandas 但分布式 + 惰性)。它背后有
**Catalyst 优化器**帮你重写执行计划 —— 这是我们用 DataFrame 而不是 RDD 的原因。
**对应项目**: `10_ingest.py` 里我们用**显式 schema** 读 JSON, 避免 Spark 多扫一遍去推断。

---

## Lesson 2 — 惰性求值: transformation vs action (最重要的一课)

```python
import time
# --- transformation: 定义, 不执行 (瞬间返回) ---
t = time.time()
beauty = df.filter(df.category == "Beauty_and_Personal_Care") \
           .filter(df.rating >= 4.0)
print("定义 transformation 用时:", round(time.time()-t, 4), "秒")   # ≈ 0 秒!

# --- action: 触发真正计算 ---
t = time.time()
n = beauty.count()          # count() 是 action → 现在才真跑
print("count =", n, " 用时:", round(time.time()-t, 2), "秒")
```
**看什么**: 定义 filter 几乎 0 秒 (什么都没算); `.count()` 才花时间 (真正扫数据)。
**概念**: **transformation (filter/select/join/groupBy...) 是懒的, 只搭计划**;
**action (count/show/collect/write...) 才触发执行**。
**面试金句**: "Spark 惰性求值 —— 一连串 transformation 只是搭 DAG, 直到遇到 action 才跑。
这正是我们 k-core 循环不加 cache 会爆炸的原因: 每轮的 count 都从头重算整个 DAG。"
**对应项目**: k-core 的 uncached vs cached 就是这个概念的直接后果。

---

## Lesson 3 — 执行计划: .explain() 读 DAG, narrow vs wide

```python
beauty.explain()                          # 简版物理计划
df.groupBy("category").count().explain()  # 注意 Exchange (=shuffle) 出现了
```
**看什么**: `filter/select` 的计划里**没有 Exchange**; `groupBy` 的计划里**有 Exchange
hashpartitioning** —— 那就是 **shuffle**。
**概念**:
- **narrow transformation** (filter/select/map): 每个输出分区只依赖一个输入分区, **不用搬数据**。
- **wide transformation** (groupBy/join/distinct/window): 需要**跨分区搬数据 = shuffle**, 很贵。
**面试**: "我会先 `.explain()` 看有没有 Exchange, shuffle 是最贵的操作, 数据量大时它决定成败。"

---

## Lesson 4 — Shuffle 与分区数

```python
spark.conf.set("spark.sql.shuffle.partitions", "8")   # 故意设小看效果
g = df.groupBy("category").count()
g.show()
# groupBy 之后默认会产生 spark.sql.shuffle.partitions 个输出分区
print("shuffle 后分区数:", df.groupBy("parent_asin").count().rdd.getNumPartitions())
spark.conf.set("spark.sql.shuffle.partitions", "200")  # 调回
```
**概念**: 每次 shuffle 后的分区数由 `spark.sql.shuffle.partitions` 决定 (默认 200)。
太少→每个 task 太大、并行度不够; 太多→小文件多、调度开销大。
**对应项目**: 我们设 256, 并在 BENCHMARKS 里量了 shuffle 后每分区的行数是否均衡。

---

## Lesson 5 — Cache: 复现我们的 k-core 教训 (缩小版)

```python
import time
# 一个"稍微贵"的中间结果
mid = df.filter(df.verified_purchase == True).groupBy("user_id").count()

# --- 不 cache: 每个 action 都重算 ---
t = time.time(); a = mid.count();               print("count1 (no cache):", round(time.time()-t,2))
t = time.time(); b = mid.filter("count>=5").count(); print("count2 (no cache):", round(time.time()-t,2))

# --- cache: 第一次物化, 之后复用 ---
mid.cache()
t = time.time(); a = mid.count();               print("count1 (cache, 首次物化):", round(time.time()-t,2))
t = time.time(); b = mid.filter("count>=5").count(); print("count2 (cache, 复用):", round(time.time()-t,2))
```
**看什么**: 不 cache 时两次 count 都慢 (都重算); cache 后第一次慢 (物化), 第二次快很多 (读内存)。
**面试金句 (我们的头号 deliverable)**: "迭代算法里, 惰性 + 不 cache = 每轮重算整个血缘。
我实测 k-core uncached 第 6 轮单轮 188s (cached 只 12s, 15×), 第 7 轮直接跑爆; 加 `.cache()`
后每轮拉平到 ~11s。这是全 pipeline 最大的 wall-time 杠杆。"
> 记得 `mid.unpersist()` 释放内存。

---

## Lesson 6 — 分区与数据倾斜 (skew)

```python
from pyspark.sql import functions as F
five = spark.read.parquet("/data/yizhou/output/reviews_5core.parquet")

# 每个 item 的交互数分布 (幂律)
ic = five.groupBy("parent_asin").count()
ic.selectExpr("percentile_approx(count, array(0.5,0.95,0.99), 10000) as p", "max(count) as mx").show(truncate=False)

# 按 key 重分区后, 每个物理分区多少行 (倾斜?)
def part_rows(dfx, key, nparts=64):
    return (dfx.repartition(nparts, key)
              .withColumn("pid", F.spark_partition_id())
              .groupBy("pid").count()
              .selectExpr("min(count) mn","avg(count) avg","max(count) mx","stddev(count) sd"))
part_rows(five, "user_id").show()      # 均衡?
part_rows(five, "parent_asin").show()  # 更倾斜?
```
**看什么**: item 的交互 max 远大于中位数 (幂律); 但按 key 重分区后 max/avg 只略大于 1
—— 因为最热的 key 还没大过一个分区。
**面试金句 (精确版)**: "倾斜取决于分区键。按 user 分区很均衡 (没有鲸鱼用户); 真正的幂律在
item 侧, 但在这个规模最热 key 仍没大过一个分区, 所以行级倾斜只有 1.18×。灾难性 straggler
需要单 key 大过分区或倾斜 join 放大。" ← 这也解释了下面 AQE 为什么没用。

---

## Lesson 7 — Join: broadcast vs sort-merge

```python
from pyspark.sql import functions as F
from spark_common import META_SCHEMA, meta_path   # 需在 /data/yizhou/repo 下启动
meta = (spark.read.schema(META_SCHEMA).json(meta_path("Toys_and_Games"))
        .dropDuplicates(["parent_asin"]))
five = spark.read.parquet("/data/yizhou/output/reviews_5core.parquet")

# 关掉自动 broadcast, 强制 SortMergeJoin
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
five.join(meta, "parent_asin", "left").explain()          # 看到 SortMergeJoin + 两个 Exchange

# 显式 broadcast → BroadcastHashJoin, 小表被复制到每个 task, 大表不用 shuffle
five.join(F.broadcast(meta), "parent_asin", "left").explain()  # 看到 BroadcastHashJoin
```
**概念**: 大表 ⋈ 小表时, 把小表 broadcast 到每个 executor, 就免了对大表的 shuffle+sort。
但小表必须真的小 (要 collect 到 driver)。
**面试金句 (我们的意外)**: "我天真地 broadcast 整张 metadata (3.5M 行带文本) 直接失败 ——
>1GB 超过 driver maxResultSize, 说明它根本不是小表; 先过滤到 5-core 用到的 715K 行才行,
之后 broadcast 比 SortMergeJoin 快 1.37×。`.explain()` 能看到计划从 SortMergeJoin 翻成
BroadcastHashJoin。"

---

## Lesson 8 — Window 函数: sessionize 与 leave-one-out

```python
from pyspark.sql import functions as F, Window
five = spark.read.parquet("/data/yizhou/output/reviews_5core.parquet")

w_asc  = Window.partitionBy("user_id").orderBy(F.col("ts_ms").asc())
w_desc = Window.partitionBy("user_id").orderBy(F.col("ts_ms").desc())

labeled = (five.withColumn("pos", F.row_number().over(w_asc))       # 时间正序位置
                .withColumn("rev", F.row_number().over(w_desc))      # 倒序位置
                .withColumn("split", F.when(F.col("rev")==1,"test")
                                      .when(F.col("rev")==2,"valid")
                                      .otherwise("train")))
labeled.filter("user_id = (select user_id from (select user_id from reviews limit 1))")  # 或挑一个用户看
labeled.groupBy("split").count().show()   # train/valid/test 行数
```
**概念**: `Window.partitionBy(user).orderBy(ts)` 就是"每个用户内部按时间排序"—— 序列推荐的
序列就是这么来的。它是 wide (按 user shuffle)。
**对应项目**: `30_sessionize_skew.py` (排序) 和 `60_split.py` (leave-one-out) 就是这个。
**面试**: "leave-one-out = 每用户最后一次做 test、倒数第二做 valid, 用 Window row_number 实现;
我还验证了 valid/test 行数正好等于用户数。"

---

## Lesson 9 — 读 Spark UI (面试常让你"讲一下你怎么调优的")

浏览器打开 `http://localhost:4040` (或 port-forward 后), 重点看四个页:
- **Jobs**: 一个 action = 一个 job; 看它拆成几个 stage。
- **Stages**: stage 边界 = shuffle 边界。看 **shuffle read/write** 大小 (数据搬了多少)。
- **Stage 内 Tasks**: 看 **task duration 分布** —— 如果少数 task 特别慢 = **straggler = 倾斜**。
- **SQL**: 看物理计划图 (Exchange / BroadcastHashJoin / SortMergeJoin)。
**练习**: 在 shell 里跑一次 `five.groupBy("parent_asin").count().count()`, 然后去 UI 的 Stages
看这次 shuffle read/write 多大、task 时间是否均匀。
**面试金句**: "我看 Stages 页的 task-duration 分布判断有没有 straggler, 看 shuffle read/write
判断哪一步在搬数据。"

---

## Lesson 10 — 写 Parquet (分区输出)

```python
(labeled.select("user_id","parent_asin","ts_ms","split")
        .write.mode("overwrite").partitionBy("split")
        .parquet("/data/yizhou/output/_tutorial_out.parquet"))
# 看目录结构: 按 split 分了子目录
import subprocess; print(subprocess.run(["ls","/data/yizhou/output/_tutorial_out.parquet"],capture_output=True,text=True).stdout)
```
**概念**: Parquet = 列式存储 (只读需要的列, 压缩好); `partitionBy` 把不同值写到不同子目录,
之后按该列过滤能"分区裁剪"跳过无关文件。
**对应项目**: 最终产物就是这样按 split / category 分区写的。

---

## 收尾: 一句话把整条线串成面试叙事

> "我把 5.9 千万条亚马逊评论用 PySpark 做成序列推荐训练集。因为 40GB 数据能装进 250G 内存,
> 我判断上集群只会增加 shuffle 开销, 所以选了 local 模式 —— 然后把时间花在真正的瓶颈上:
> 我实测了不 cache 的迭代 k-core 会因惰性求值每轮重算整个 DAG 而爆炸 (15×), 加 cache 后拉平;
> 我量化了数据倾斜, 发现倾斜取决于分区键、且这个规模下行级倾斜温和, 这也解释了 AQE 为何无收益;
> 我发现所谓'元数据小表'其实 >1GB broadcast 会失败, 过滤后才 1.37× 提速。全部有数字、有物理计划。"

练完这 10 课, 上面每一句你都能自己动手复现。
