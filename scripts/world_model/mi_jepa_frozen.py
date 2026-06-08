#!/usr/bin/env python3
"""
JEPA scaling lever #1 — frozen strong pretrained encoder.

Instead of training a distilbert encoder from scratch on 4k AnnoMI examples, use
a sentence encoder pretrained on ~1B pairs (all-mpnet-base-v2), FROZEN, as both
the context and target encoder. This injects "scale" via the pretrained weights
(the JEPA recommendation: init from pretrained, don't train from scratch) and
removes EMA drift / collapse entirely — only the action-conditioned predictor
trains, regressing context+action -> target embedding.

Validation = same linear-probe readout, compared to distilbert-JEPA and tabular.

  python -m scripts.world_model.mi_jepa_frozen --backbone sentence-transformers/all-mpnet-base-v2 --steps 4000
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.world_model.mi_jepa import load, train_linear_probe, macro_f1, TALKS
from scripts.world_model.transition_model import ACTIONS


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


@torch.no_grad()
def embed(texts, tok, bert, dev, max_len, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", truncation=True,
                  max_length=max_len, padding=True)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(dev) for k, v in enc.items()}
        h = bert(**enc).last_hidden_state
        out.append(F.normalize(mean_pool(h, enc["attention_mask"]), dim=-1).cpu())
    return torch.cat(out, 0)


class Predictor(nn.Module):
    def __init__(self, d, n_act, d_act=64, hid=512):
        super().__init__()
        self.act = nn.Embedding(n_act, d_act)
        self.net = nn.Sequential(nn.Linear(d + d_act, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, d))

    def forward(self, ctx, a):
        return self.net(torch.cat([ctx, self.act(a)], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_model/transitions.jsonl")
    ap.add_argument("--backbone", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--tgt-len", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.backbone)
    bert = AutoModel.from_pretrained(args.backbone).eval().to(dev)
    for p in bert.parameters():
        p.requires_grad_(False)

    train_rows, val_rows = load(args.data)
    d = bert.config.hidden_size
    print(f"[jepa-frozen] backbone={args.backbone} dim={d} train={len(train_rows)} val={len(val_rows)}")

    # precompute frozen embeddings (context + target)
    Xc_tr = embed([r["ctx"] for r in train_rows], tok, bert, dev, args.max_len)
    Xt_tr = embed([r["tgt"] for r in train_rows], tok, bert, dev, args.tgt_len)
    Xc_va = embed([r["ctx"] for r in val_rows], tok, bert, dev, args.max_len)
    Xt_va = embed([r["tgt"] for r in val_rows], tok, bert, dev, args.tgt_len)
    a_tr = torch.tensor([r["action"] for r in train_rows])
    y_tr = [r["next_talk"] for r in train_rows]
    y_va = [r["next_talk"] for r in val_rows]

    # train predictor: context+action -> target embedding (cosine + mse)
    pred = Predictor(d, len(ACTIONS)).to(dev)
    opt = torch.optim.AdamW(pred.parameters(), lr=args.lr, weight_decay=1e-4)
    Xc_tr_d, Xt_tr_d, a_tr_d = Xc_tr.to(dev), Xt_tr.to(dev), a_tr.to(dev)
    n = len(train_rows)
    g = torch.Generator().manual_seed(0)
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.bs,), generator=g)
        c, t, ac = Xc_tr_d[idx], Xt_tr_d[idx], a_tr_d[idx]
        p = pred(c, ac)
        loss = F.mse_loss(p, t) + (1 - F.cosine_similarity(p, t, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"[jepa-frozen] step {step+1}/{args.steps} loss={loss.item():.4f}")

    # predicted latents
    pred.eval()
    with torch.no_grad():
        P_tr = pred(Xc_tr.to(dev), a_tr.to(dev)).cpu()
        P_va = pred(Xc_va.to(dev), torch.tensor([r["action"] for r in val_rows]).to(dev)).cpu()

    def fit_eval(Xtr, Xva):
        clf = train_linear_probe(Xtr, y_tr, len(TALKS), dev)
        with torch.no_grad():
            pv = clf(Xva.to(dev)).argmax(-1).cpu().tolist()
        return (round(sum(p == g for p, g in zip(pv, y_va)) / len(y_va), 4),
                round(macro_f1(pv, y_va, len(TALKS)), 4))

    tgt = fit_eval(Xt_tr, Xt_va)
    dyn = fit_eval(P_tr, P_va)
    ctx = fit_eval(Xc_tr, Xc_va)
    report = {
        "backbone": args.backbone, "n_val": len(val_rows),
        "target_probe (sees reply, upper bound)": {"acc": tgt[0], "macro_f1": tgt[1]},
        "jepa_dynamics_readout (predicted latents)": {"acc": dyn[0], "macro_f1": dyn[1]},
        "context_readout": {"acc": ctx[0], "macro_f1": ctx[1]},
        "distilbert_jepa_dynamics": {"acc": 0.4458, "macro_f1": 0.3815},
        "tabular_transition_baseline": {"acc": 0.6714, "macro_f1": 0.5606},
    }
    json.dump(report, open("reports/mi_jepa_frozen_eval.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
