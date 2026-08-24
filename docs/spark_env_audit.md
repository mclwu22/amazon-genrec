# Spark Environment Audit — exx (M2)

- Host: sn4622122600, user exx
- CPU: 32 vCPU (AMD Threadripper PRO 5955WX)
- RAM: 251G total (~214G available)
- Java: OpenJDK 11.0.25 (no javac; not needed)
- Python: 3.10 via user-space Miniconda at /data/yizhou/miniconda, env /data/yizhou/envs/spark
- Spark: pyspark 3.5.9 (pinned 3.5.*; Spark 4.x needs JDK17/21 — avoided)
- Libs: pyarrow 25, pandas 2.3, huggingface_hub 1.27
- Storage: /data = 19T ext4, local disk (NOT NFS), separate partition; ~3T free
- SPARK_LOCAL_DIRS = /data/yizhou/spark-tmp (never write to /)
- Config: local[24], driver.memory=96g, shuffle.partitions=256
- No sudo; everything user-space
- GPU: RTX 6000 Ada — MIND experiment finished during this work (100%→idle), so the
  8-core reservation constraint effectively lifted, but local[24] was kept per brief.
