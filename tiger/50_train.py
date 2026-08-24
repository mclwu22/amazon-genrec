"""TIGER Stage 2, step 2+3: train seq2seq (T5) generative retrieval + eval Recall@K/NDCG@K.

Input tokens from 40_make_tokens.py. Model = small T5 encoder-decoder; target = next item's
4 Semantic-ID tokens + EOS. Eval = constrained beam search (only valid Semantic IDs) on a
sample of valid users; final test eval at the end.

  conda activate /data/yizhou/envs/tiger
  python tiger_50_train.py --epochs 10 --max-train-pairs 6000000 --d-model 128
"""
import argparse, json, time, math, os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Config, T5ForConditionalGeneration

DATA = "/data/yizhou/tiger/data"
CKPT = "/data/yizhou/tiger/checkpoints"


def load_pairs(path, max_n=None, in_len=81, tgt_len=5, EOS=1026, PAD=1024):
    inps, tgts = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_n and i >= max_n:
                break
            d = json.loads(line)
            a = d["inp"][:in_len]
            a = a + [PAD] * (in_len - len(a))
            t = d["tgt"] + [EOS]                       # 4 + EOS = 5
            inps.append(a); tgts.append(t)
    X = np.asarray(inps, dtype=np.int64)
    Y = np.asarray(tgts, dtype=np.int64)
    return X, Y


def build_trie(path, EOS=1026):
    """nested dict: prefix tuple -> set of allowed next tokens; leaf -> {EOS}."""
    root = {}
    item_of = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            t = tuple(d["t"])
            item_of[t] = d["pa"]
            node = root
            for tok in t:
                node = node.setdefault(tok, {})
            node[EOS] = True
    return root, item_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-train-pairs", type=int, default=6_000_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--eval-users", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--test-every", type=int, default=5, help="run test eval every N epochs (overnight safety)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bf16", action="store_true", help="use bfloat16 autocast (faster on Ada)")
    args = ap.parse_args()
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.bf16 else \
              (lambda: torch.autocast("cuda", enabled=False))
    os.makedirs(CKPT, exist_ok=True)
    dev = "cuda"
    VOCAB, EOS, PAD = 1027, 1026, 1024

    print(f"[train] loading data ...", flush=True)
    t0 = time.time()
    Xtr, Ytr = load_pairs(f"{DATA}/tokens_train.jsonl", args.max_train_pairs)
    Xva, Yva = load_pairs(f"{DATA}/tokens_valid.jsonl", args.eval_users)
    print(f"[train] train={len(Xtr):,} valid_eval={len(Xva):,} ({time.time()-t0:.0f}s)", flush=True)
    trie, item_of = build_trie(f"{DATA}/valid_item_tokens.jsonl")
    print(f"[train] trie built: {len(item_of):,} valid semantic ids", flush=True)

    cfg = T5Config(vocab_size=VOCAB, d_model=args.d_model, d_ff=args.d_ff,
                   num_layers=args.layers, num_decoder_layers=args.layers,
                   num_heads=args.heads, d_kv=args.d_model // args.heads,
                   decoder_start_token_id=PAD, pad_token_id=PAD, eos_token_id=EOS,
                   dropout_rate=0.1)
    model = T5ForConditionalGeneration(cfg).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={nparams/1e6:.1f}M d_model={args.d_model} layers={args.layers}", flush=True)

    # Keep ALL training data resident on the GPU as int16 (token ids < 1027 fit in int16),
    # and batch by slicing a GPU permutation. This removes DataLoader/worker/H2D overhead,
    # which dominates for this small model + short sequences (was ~3 steps/s DataLoader-bound).
    Xtr_g = torch.from_numpy(Xtr.astype(np.int16)).to(dev)
    Ytr_g = torch.from_numpy(Ytr.astype(np.int16)).to(dev)
    n_tr = Xtr_g.shape[0]
    steps_per_epoch = n_tr // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    # --- constrained decoding: allowed next tokens given generated prefix ---
    def allowed_fn(batch_id, input_ids):
        # input_ids starts with decoder_start_token (PAD); strip it to index the trie
        gen = input_ids.tolist()
        if gen and gen[0] == PAD:
            gen = gen[1:]
        node = trie
        for tok in gen:
            if tok in node:
                node = node[tok]
            else:
                return [EOS]
        return list(node.keys())

    @torch.no_grad()
    def evaluate(X, Y, k):
        model.eval()
        hit = 0.0; ndcg = 0.0; n = len(X)
        bs = 128
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i+bs]).to(dev)
            am = (xb != PAD).long()
            with amp_ctx():
                out = model.generate(input_ids=xb, attention_mask=am, max_new_tokens=6,
                                      num_beams=k, num_return_sequences=k,
                                      prefix_allowed_tokens_fn=allowed_fn,
                                      early_stopping=True)
            out = out.view(len(xb), k, -1)
            for j in range(len(xb)):
                true = tuple(int(t) for t in Y[i+j][:4])
                found_rank = None
                for r in range(k):
                    seq = [int(t) for t in out[j, r].tolist() if t not in (PAD, EOS)]
                    if tuple(seq[:4]) == true:
                        found_rank = r; break
                if found_rank is not None:
                    hit += 1
                    ndcg += 1.0 / math.log2(found_rank + 2)
        return hit / n, ndcg / n

    Xte, Yte = load_pairs(f"{DATA}/tokens_test.jsonl", max(args.eval_users, 20000))
    print(f"[train] test_eval={len(Xte):,}", flush=True)

    print(f"[train] START epochs={args.epochs} steps/epoch={steps_per_epoch}", flush=True)
    best_r = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0; nb = 0; te = time.time()
        perm = torch.randperm(n_tr, device=dev)
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            xb = Xtr_g[idx].long(); yb = Ytr_g[idx].long()
            am = (xb != PAD).long()
            labels = yb.clone()                          # 4 tokens + EOS, no pad -> no -100 needed
            with amp_ctx():
                loss = model(input_ids=xb, attention_mask=am, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item(); nb += 1
        msg = f"  ep{ep:3d} loss={tl/nb:.4f} ({time.time()-te:.0f}s)"
        if ep % args.eval_every == 0:
            r, nd = evaluate(Xva, Yva, args.topk)
            msg += f"  | valid R@{args.topk}={r:.4f} N@{args.topk}={nd:.4f}"
            if r > best_r:
                best_r = r
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep,
                            "valid_recall": r}, f"{CKPT}/seq2seq_best.pt")
                msg += " *best*"
        if ep % args.test_every == 0:
            rt, ndt = evaluate(Xte, Yte, args.topk)
            msg += f"  || TEST R@{args.topk}={rt:.4f} N@{args.topk}={ndt:.4f}"
        print(msg, flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, f"{CKPT}/seq2seq.pt")

    # --- final TEST eval on a larger sample ---
    r, nd = evaluate(Xte, Yte, args.topk)
    print(f"[train] FINAL TEST  Recall@{args.topk}={r:.4f}  NDCG@{args.topk}={nd:.4f}  "
          f"(on {len(Xte):,} users)  best_valid_R={best_r:.4f}", flush=True)
    print(f"[train] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
