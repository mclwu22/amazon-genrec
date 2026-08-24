"""TIGER Stage 2, step 1: translate user sequences -> Semantic-ID token sequences.

Vocab layout (each item = 4 tokens):
  layer1 code c1 in 0..255   -> token        c1          (0..255)
  layer2 code c2 in 0..255   -> token  256 + c2          (256..511)
  layer3 code c3 in 0..255   -> token  512 + c3          (512..767)
  collision  c4 in 0..255    -> token  768 + c4          (768..1023)
  specials: PAD=1024, BOS=1025, EOS=1026                 vocab size = 1027

Produces token id lists per user with leave-one-out structure, saved as a compact .npz-ish
jsonl the trainer reads:
  data/tokens_train.jsonl : {"inp":[...hist tokens...], "tgt":[4 tokens]}   (many per user)
  data/tokens_valid.jsonl : predict 2nd-last item from its prefix
  data/tokens_test.jsonl  : predict last item from its prefix
  data/semantic_vocab.json: sizes + the item<->4tokens maps for constrained decoding
"""
import json, time
import pyarrow.parquet as pq

DATA = "/data/yizhou/tiger/data"
SIDS = "/data/yizhou/tiger/semantic_ids/item_semantic_ids.parquet"
MAX_HIST_ITEMS = 20          # cap history length (most recent 20 items = 80 tokens)

L1, L2, L3, COL = 0, 256, 512, 768
PAD, BOS, EOS = 1024, 1025, 1026
VOCAB = 1027


def item_tokens(row):
    return [L1 + row[0], L2 + row[1], L3 + row[2], COL + row[3]]


def main():
    t0 = time.time()
    # load semantic ids -> map parent_asin -> 4 tokens
    tbl = pq.read_table(SIDS).to_pydict()
    sid = {}
    for pa, c1, c2, c3, c4 in zip(tbl["parent_asin"], tbl["c1"], tbl["c2"], tbl["c3"], tbl["c_collision"]):
        sid[pa] = item_tokens((c1, c2, c3, c4))
    print(f"[tokens] loaded {len(sid):,} item semantic ids ({time.time()-t0:.0f}s)", flush=True)

    ftr = open(f"{DATA}/tokens_train.jsonl", "w")
    fva = open(f"{DATA}/tokens_valid.jsonl", "w")
    fte = open(f"{DATA}/tokens_test.jsonl", "w")
    n_users = n_train = n_val = n_test = skipped = 0

    with open(f"{DATA}/sequences.jsonl") as f:
        for line in f:
            u = json.loads(line)
            items = u["items"]
            # map to tokens; skip items missing a semantic id (shouldn't happen)
            toks = [sid[i] for i in items if i in sid]
            n = len(toks)
            if n < 3:
                skipped += 1
                continue
            n_users += 1
            # flatten helper for a history window
            def flat(seq):
                out = [BOS]
                for it in seq[-MAX_HIST_ITEMS:]:
                    out.extend(it)
                return out
            # TEST: predict last from all before it
            fte.write(json.dumps({"inp": flat(toks[:n-1]), "tgt": toks[n-1]}) + "\n"); n_test += 1
            # VALID: predict 2nd-last from all before it
            fva.write(json.dumps({"inp": flat(toks[:n-2]), "tgt": toks[n-2]}) + "\n"); n_val += 1
            # TRAIN: predict each position t in [1 .. n-3] from its prefix (train portion only)
            for t in range(1, n - 2):
                ftr.write(json.dumps({"inp": flat(toks[:t]), "tgt": toks[t]}) + "\n"); n_train += 1

    for fh in (ftr, fva, fte):
        fh.close()

    vocab = {"vocab_size": VOCAB, "PAD": PAD, "BOS": BOS, "EOS": EOS,
             "layer_offsets": [L1, L2, L3, COL], "max_hist_items": MAX_HIST_ITEMS,
             "n_items": len(sid)}
    json.dump(vocab, open(f"{DATA}/semantic_vocab.json", "w"))
    # also dump the set of all valid item token-tuples for constrained decoding / dedup
    with open(f"{DATA}/valid_item_tokens.jsonl", "w") as f:
        for pa, toks in sid.items():
            f.write(json.dumps({"pa": pa, "t": toks}) + "\n")

    print(f"[tokens] users={n_users:,} skipped={skipped:,}", flush=True)
    print(f"[tokens] train pairs={n_train:,}  valid={n_val:,}  test={n_test:,}", flush=True)
    print(f"[tokens] vocab_size={VOCAB}  max_hist_items={MAX_HIST_ITEMS}", flush=True)
    print(f"[tokens] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
