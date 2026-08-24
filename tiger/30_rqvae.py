"""TIGER Stage 1: RQ-VAE — learn Semantic IDs from frozen item embeddings.

Pipeline recap: item text -> (frozen Sentence-T5) -> x[768] -> RQ-VAE encoder -> z[32]
-> residual quantize with L codebooks -> Semantic ID (c1,c2,c3) -> decoder -> x_hat.
Loss = reconstruction + sum_l (codebook + beta*commitment). k-means init + dead-code revive.

  conda activate /data/yizhou/envs/tiger
  python tiger_30_rqvae.py --epochs 50 --num-codebooks 3 --codebook-size 256

Outputs: checkpoints/rqvae.pt, semantic_ids/item_semantic_ids.parquet (parent_asin, c1,c2,c3,c4)
"""
import argparse, time, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fnn

DATA = "/data/yizhou/tiger/data"
CKPT = "/data/yizhou/tiger/checkpoints"
SIDS = "/data/yizhou/tiger/semantic_ids"


def kmeans_init(x, k, iters=10):
    """Simple k-means to initialize a codebook from data x[N,d]."""
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device)[:k]
    c = x[idx].clone()
    for _ in range(iters):
        d = torch.cdist(x, c)                    # [N,k]
        a = d.argmin(1)                          # assignment
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


class ResidualQuantizer(nn.Module):
    def __init__(self, num_cb, cb_size, dim, beta=0.25):
        super().__init__()
        self.num_cb, self.cb_size, self.beta = num_cb, cb_size, beta
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(cb_size, dim) * 0.1) for _ in range(num_cb)])
        self.register_buffer("usage", torch.zeros(num_cb, cb_size))
        self.inited = False

    @torch.no_grad()
    def init_codebooks(self, z):
        r = z
        for l in range(self.num_cb):
            self.codebooks[l].data = kmeans_init(r, self.cb_size)
            d = torch.cdist(r, self.codebooks[l])
            idx = d.argmin(1)
            r = r - self.codebooks[l][idx]
        self.inited = True

    def forward(self, z):
        r = z
        codes, cb_loss, commit_loss = [], 0.0, 0.0
        z_q_total = torch.zeros_like(z)
        for l in range(self.num_cb):
            d = torch.cdist(r, self.codebooks[l])        # [B,K]
            idx = d.argmin(1)                            # [B]
            e = self.codebooks[l][idx]                   # [B,dim]
            cb_loss = cb_loss + Fnn.mse_loss(e, r.detach())      # move codebook -> residual
            commit_loss = commit_loss + Fnn.mse_loss(r, e.detach())  # move encoder -> code
            z_q_total = z_q_total + e
            r = r - e                                    # residual for next layer
            codes.append(idx)
            if self.training:
                self.usage[l].scatter_add_(0, idx, torch.ones_like(idx, dtype=self.usage.dtype))
        z_q = z + (z_q_total - z).detach()               # STE
        return z_q, torch.stack(codes, 1), cb_loss, self.beta * commit_loss

    @torch.no_grad()
    def revive_dead(self, z, min_usage=1):
        """Reset codes never used this epoch to random data points."""
        revived = 0
        for l in range(self.num_cb):
            dead = (self.usage[l] < min_usage).nonzero().flatten()
            if len(dead):
                pts = z[torch.randperm(z.shape[0], device=z.device)[:len(dead)]]
                self.codebooks[l].data[dead] = pts
                revived += len(dead)
        self.usage.zero_()
        return revived


class RQVAE(nn.Module):
    def __init__(self, in_dim=768, latent=32, num_cb=3, cb_size=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,512), nn.ReLU(), nn.Linear(512,256),
                                 nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,latent))
        self.dec = nn.Sequential(nn.Linear(latent,128), nn.ReLU(), nn.Linear(128,256),
                                 nn.ReLU(), nn.Linear(256,512), nn.ReLU(), nn.Linear(512,in_dim))
        self.rq = ResidualQuantizer(num_cb, cb_size, latent)

    def forward(self, x):
        z = self.enc(x)
        z_q, codes, cb, commit = self.rq(z)
        x_hat = self.dec(z_q)
        recon = Fnn.mse_loss(x_hat, x)
        return recon, cb, commit, codes, recon + cb + commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--num-codebooks", type=int, default=3)
    ap.add_argument("--codebook-size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True); os.makedirs(SIDS, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(np.load(f"{DATA}/item_emb.npy")).float()
    ids = open(f"{DATA}/item_ids.txt").read().splitlines()
    assert x.shape[0] == len(ids)
    # standardize (helps quantization)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x = (x - mu) / sd
    n = x.shape[0]
    print(f"[rqvae] {n:,} items dim={x.shape[1]} | L={args.num_codebooks} K={args.codebook_size} dev={dev}", flush=True)

    model = RQVAE(x.shape[1], args.latent, args.num_codebooks, args.codebook_size).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # k-means init codebooks from a sample of encoder outputs
    with torch.no_grad():
        samp = x[torch.randperm(n)[:20000]].to(dev)
        model.rq.init_codebooks(model.enc(samp))
    print("[rqvae] codebooks k-means initialized", flush=True)

    xg = x.to(dev)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = {"recon":0.0, "cb":0.0, "commit":0.0}; nb = 0
        for i in range(0, n, args.batch):
            b = xg[perm[i:i+args.batch]]
            recon, cb, commit, _, loss = model(b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot["recon"]+=recon.item(); tot["cb"]+=cb.item(); tot["commit"]+=commit.item(); nb+=1
        revived = model.rq.revive_dead(model.enc(xg[torch.randperm(n, device=dev)[:20000]]))
        # codebook utilization (this epoch, before reset it was accumulated; recompute quick)
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                _,_,_,codes,_ = model(xg[:50000])
                util = [len(codes[:,l].unique())/args.codebook_size for l in range(args.num_codebooks)]
            print(f"  ep{ep:3d} recon={tot['recon']/nb:.4f} cb={tot['cb']/nb:.4f} "
                  f"commit={tot['commit']/nb:.4f} util={['%.2f'%u for u in util]} revived={revived} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # assign semantic ids to ALL items + resolve collisions with a 4th token
    model.eval()
    with torch.no_grad():
        all_codes = []
        for i in range(0, n, 8192):
            _,_,_,codes,_ = model(xg[i:i+8192])
            all_codes.append(codes.cpu())
        codes = torch.cat(all_codes).numpy()   # [N, L]
    print(f"[rqvae] assigning semantic ids + collision 4th token", flush=True)
    seen, c4 = {}, np.zeros(n, dtype=np.int64)
    for i in range(n):
        key = tuple(codes[i].tolist())
        c4[i] = seen.get(key, 0); seen[key] = c4[i] + 1
    n_collide = sum(v-1 for v in seen.values() if v > 1)
    print(f"[rqvae] unique 3-code combos={len(seen):,} / {n:,} items; collisions needing c4>0: {n_collide:,}", flush=True)

    import pyarrow as pa, pyarrow.parquet as pq
    cols = {"parent_asin": ids}
    for l in range(args.num_codebooks):
        cols[f"c{l+1}"] = codes[:, l].tolist()
    cols["c_collision"] = c4.tolist()
    pq.write_table(pa.table(cols), f"{SIDS}/item_semantic_ids.parquet")
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "args": vars(args)}, f"{CKPT}/rqvae.pt")
    print(f"[rqvae] DONE -> {SIDS}/item_semantic_ids.parquet + {CKPT}/rqvae.pt ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
