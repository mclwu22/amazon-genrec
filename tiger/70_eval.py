"""Standalone eval of a trained seq2seq checkpoint at multiple K (Recall@K / NDCG@K).

  conda activate /data/yizhou/envs/tiger
  python tiger_70_eval.py --ckpt checkpoints/seq2seq_best.pt --ks 5,10 --eval-users 20000
"""
import argparse, json, math
import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration

DATA = "/data/yizhou/tiger/data"
PAD, BOS, EOS, VOCAB = 1024, 1025, 1026, 1027


def load_pairs(path, max_n, in_len=81):
    inps, tgts = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            d = json.loads(line)
            a = d["inp"][:in_len]; a = a + [PAD] * (in_len - len(a))
            inps.append(a); tgts.append(d["tgt"] + [EOS])
    return np.asarray(inps, np.int64), np.asarray(tgts, np.int64)


def build_trie(path):
    root = {}
    with open(path) as f:
        for line in f:
            t = tuple(json.loads(line)["t"]); node = root
            for tok in t:
                node = node.setdefault(tok, {})
            node[EOS] = True
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/yizhou/tiger/checkpoints/seq2seq_best.pt")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--eval-users", type=int, default=20000)
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    cfg = T5Config(vocab_size=VOCAB, d_model=a["d_model"], d_ff=a["d_ff"],
                   num_layers=a["layers"], num_decoder_layers=a["layers"], num_heads=a["heads"],
                   d_kv=a["d_model"] // a["heads"], decoder_start_token_id=PAD,
                   pad_token_id=PAD, eos_token_id=EOS)
    model = T5ForConditionalGeneration(cfg).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[eval] loaded {args.ckpt} (epoch {ck.get('epoch')}, valid_R {ck.get('valid_recall')})", flush=True)

    trie = build_trie(f"{DATA}/valid_item_tokens.jsonl")
    Xte, Yte = load_pairs(f"{DATA}/tokens_test.jsonl", args.eval_users)
    print(f"[eval] test users={len(Xte):,}", flush=True)

    def allowed_fn(bid, ids):
        gen = ids.tolist()
        if gen and gen[0] == PAD:
            gen = gen[1:]
        node = trie
        for tok in gen:
            node = node.get(tok) if isinstance(node, dict) else None
            if node is None:
                return [EOS]
        return list(node.keys())

    kmax = max(ks)
    hits = {k: 0.0 for k in ks}; ndcg = {k: 0.0 for k in ks}
    n = len(Xte); bs = 128
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(Xte[i:i+bs]).to(dev); am = (xb != PAD).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(input_ids=xb, attention_mask=am, max_new_tokens=6,
                                     num_beams=kmax, num_return_sequences=kmax,
                                     prefix_allowed_tokens_fn=allowed_fn, early_stopping=True)
            out = out.view(len(xb), kmax, -1)
            for j in range(len(xb)):
                true = tuple(int(t) for t in Yte[i+j][:4])
                rank = None
                for r in range(kmax):
                    seq = [int(t) for t in out[j, r].tolist() if t not in (PAD, EOS)]
                    if tuple(seq[:4]) == true:
                        rank = r; break
                if rank is not None:
                    for k in ks:
                        if rank < k:
                            hits[k] += 1; ndcg[k] += 1.0 / math.log2(rank + 2)
    print("\n=== TEST metrics ===", flush=True)
    for k in ks:
        print(f"  Recall@{k}={hits[k]/n:.4f}  NDCG@{k}={ndcg[k]/n:.4f}", flush=True)


if __name__ == "__main__":
    main()
