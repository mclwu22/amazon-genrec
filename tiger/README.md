# TIGER Reproduction (Amazon Reviews 2023)

Generative retrieval for sequential recommendation — RQ-VAE Semantic IDs + a seq2seq
Transformer that *generates* the next item's Semantic ID. Input = the Spark project's
output (`interactions_split.parquet`, `item_meta.parquet`). Runs on **exx** (RTX 6000 Ada).

Env: `/data/yizhou/envs/tiger` (torch 2.5.1+cu121, transformers, sentence-transformers).
Activate: `source /data/yizhou/miniconda/etc/profile.d/conda.sh && conda activate /data/yizhou/envs/tiger`

## Pipeline

| # | script | env | what | status |
|---|---|---|---|---|
| 1 | `10_prep.py` | spark | export item_text.parquet (715K) + sequences.jsonl (2.39M users) | ✅ |
| 2 | `20_embed.py` | tiger | frozen Sentence-T5-base → item_emb.npy [715729,768] | ✅ |
| 3 | `30_rqvae.py` | tiger | **RQ-VAE** → item_semantic_ids.parquet (c1,c2,c3,c_collision) | ✅ |
| 4 | `40_seq2seq.py` | tiger | seq2seq Transformer generates next item's Semantic ID | ⬜ NEXT |

## Stage 1 results (RQ-VAE)

- 100 epochs, ~200s. Reconstruction loss 0.67 → 0.42 (standardized space).
- **Codebook utilization 100% on all 3 layers** — no collapse (k-means init + dead-code revival).
- 484,129 unique 3-code combos / 715,729 items; 231,600 items collide on the first 3 codes
  and are separated by a 4th `c_collision` token (max 150 items per combo).
- Each item → Semantic ID `(c1, c2, c3, c_collision)`, e.g. `0735355673 → (207,131,151,0)`.

## Data locations (all under /data/yizhou/tiger/)

```
data/item_text.parquet          715K (parent_asin, text=title+cats+store+desc)
data/sequences.jsonl            2.39M users {user_id, items[ordered], n, n_train}
data/item_emb.npy               [715729,768] float32  (frozen Sentence-T5)
data/item_ids.txt               parent_asin per emb row (aligned)
semantic_ids/item_semantic_ids.parquet   parent_asin, c1, c2, c3, c_collision
checkpoints/rqvae.pt            model + mu/sd standardization + args
```

## Stage 2 plan (next session)

1. Build token vocab: layer1 codes → tokens 0..255, layer2 → 256..511, layer3 → 512..767,
   collision token, + PAD/BOS/EOS. Each item = 4 tokens.
2. Map each user sequence's parent_asin → its 4 semantic tokens. Train = all but last 2 items;
   valid = predict 2nd-last; test = predict last (leave-one-out, already labeled by Spark).
3. Train a small T5-style encoder-decoder: input = history tokens, target = next item's 4 tokens.
4. Eval: constrained beam search (only emit valid existing Semantic IDs) → Recall@K / NDCG@K
   on the test set.
