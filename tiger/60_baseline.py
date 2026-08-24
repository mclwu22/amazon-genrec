"""Popularity baseline for TIGER eval — the honest comparison point.

"Just recommend the globally most-popular items" — if the trained model can't beat this,
it hasn't learned sequential structure. Computes Recall@K / NDCG@K of the top-K most
popular items (by train-portion interaction count) on the SAME leave-one-out test set.

  conda activate /data/yizhou/envs/tiger
  python tiger_60_baseline.py --topk 10
"""
import argparse, json, math, collections

DATA = "/data/yizhou/tiger/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--eval-users", type=int, default=20000)
    args = ap.parse_args()

    # popularity from TRAIN portion only (avoid leakage): count items in each user's
    # sequence except the last 2 (valid/test).
    pop = collections.Counter()
    with open(f"{DATA}/sequences.jsonl") as f:
        for line in f:
            u = json.loads(line)
            for it in u["items"][:-2]:
                pop[it] += 1
    topk_items = [i for i, _ in pop.most_common(args.topk)]
    topk_set = set(topk_items)
    rank_of = {it: r for r, it in enumerate(topk_items)}
    print(f"[baseline] top-{args.topk} popular items computed from train portion", flush=True)

    # eval on test target = last item of each user
    hit = 0.0; ndcg = 0.0; n = 0
    with open(f"{DATA}/sequences.jsonl") as f:
        for line in f:
            if n >= args.eval_users:
                break
            u = json.loads(line)
            if len(u["items"]) < 3:
                continue
            true = u["items"][-1]
            n += 1
            if true in topk_set:
                hit += 1
                ndcg += 1.0 / math.log2(rank_of[true] + 2)
    print(f"[baseline] popularity  Recall@{args.topk}={hit/n:.4f}  NDCG@{args.topk}={ndcg/n:.4f}  "
          f"(on {n:,} users)", flush=True)


if __name__ == "__main__":
    main()
