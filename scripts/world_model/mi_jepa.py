#!/usr/bin/env python3
"""
Tier 3 — Action-conditioned MI-JEPA (V-JEPA 2-AC style) for dialogue dynamics.

Idea (the purest form of "don't predict the text, predict its representation"):
    context encoder  E_ctx : dialogue history     -> s_t      (trainable)
    target  encoder  E_tgt : the client's reply   -> s_{t+1}* (EMA of E_ctx, stop-grad)
    predictor        P     : (s_t, action)         -> s^_{t+1}   <- the latent world model
    loss = VICReg(invariance s^ vs s*, + variance + covariance to prevent collapse)

Self-supervised: learns the transition DYNAMICS with NO talk-type labels.
Labels are used only to VALIDATE, via a linear probe:
    probe trained on E_tgt(reply) -> next_talk      (how decodable talk-type is)
    apply that probe to P(context, action)          -> predicted next_talk WITHOUT seeing the reply
    => compare to the tabular transition model (acc 0.67 / macro-F1 0.56).

Backbone = distilbert (init pretrained; do NOT train from scratch on 4k examples).

Usage:
  python -m scripts.world_model.mi_jepa --epochs 8 --out runs/mi_jepa
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.world_model.transition_model import ACTIONS

ACT2ID = {a: i for i, a in enumerate(ACTIONS)}
TALKS = ["change", "sustain", "neutral"]
TALK2ID = {t: i for i, t in enumerate(TALKS)}


# ----------------------------- data -----------------------------
def history_to_text(history, max_turns=6):
    turns = history[-max_turns:]
    return "\n".join(f"{t['role']}: {t['text']}" for t in turns) or "(start)"


def load(path):
    train, val = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rec = {
            "ctx": history_to_text(r["history"]),
            "tgt": r["future_text"],
            "action": ACT2ID[r["action"]],
            "next_talk": TALK2ID[r["next_talk"]],
        }
        (val if r["split"] == "val" else train).append(rec)
    return train, val


# ----------------------------- model -----------------------------
class Encoder(nn.Module):
    """distilbert backbone + mean pooling + projection to dim d."""
    def __init__(self, backbone, d=256):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(backbone)
        h = self.bert.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.proj(pooled)


class Predictor(nn.Module):
    def __init__(self, d=256, n_actions=len(ACTIONS), d_act=64):
        super().__init__()
        self.act = nn.Embedding(n_actions, d_act)
        self.net = nn.Sequential(
            nn.Linear(d + d_act, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, ctx_emb, action_ids):
        return self.net(torch.cat([ctx_emb, self.act(action_ids)], dim=-1))


# ----------------------------- VICReg -----------------------------
def vicreg(pred, target, sim=25.0, var=25.0, cov=1.0):
    inv = F.mse_loss(pred, target)

    def var_term(x):
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    def cov_term(x):
        x = x - x.mean(dim=0)
        n, d = x.shape
        c = (x.T @ x) / (n - 1)
        off = c - torch.diag(torch.diag(c))
        return (off ** 2).sum() / d

    v = var_term(pred) + var_term(target)
    cv = cov_term(pred) + cov_term(target)
    return sim * inv + var * v + cov * cv, {"inv": inv.item(), "var": v.item(), "cov": cv.item()}


# ----------------------------- training -----------------------------
def batched(rows, bs):
    for i in range(0, len(rows), bs):
        yield rows[i:i + bs]


def encode_batch(tok, texts, dev, max_len):
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
    enc.pop("token_type_ids", None)
    return {k: v.to(dev) for k, v in enc.items()}


@torch.no_grad()
def ema_update(target, online, m):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1 - m)


def train(args):
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.backbone)
    train_rows, val_rows = load(args.data)
    print(f"[jepa] train={len(train_rows)} val={len(val_rows)} dev={dev}")

    ctx_enc = Encoder(args.backbone, args.dim).to(dev)
    tgt_enc = Encoder(args.backbone, args.dim).to(dev)
    tgt_enc.load_state_dict(ctx_enc.state_dict())
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    predictor = Predictor(args.dim).to(dev)

    params = list(ctx_enc.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    step = 0
    for ep in range(args.epochs):
        import random
        random.Random(ep).shuffle(train_rows)
        ctx_enc.train(); predictor.train()
        agg = {"inv": 0, "var": 0, "cov": 0, "n": 0}
        for batch in batched(train_rows, args.bs):
            ci = encode_batch(tok, [b["ctx"] for b in batch], dev, args.max_len)
            ti = encode_batch(tok, [b["tgt"] for b in batch], dev, args.tgt_len)
            acts = torch.tensor([b["action"] for b in batch], device=dev)
            ctx_emb = ctx_enc(**ci)
            with torch.no_grad():
                tgt_emb = tgt_enc(**ti)
            pred = predictor(ctx_emb, acts)
            loss, logs = vicreg(pred, tgt_emb, args.sim, args.var, args.cov)
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(tgt_enc, ctx_enc, args.ema)
            step += 1
            for k in logs: agg[k] += logs[k] * len(batch)
            agg["n"] += len(batch)
        n = agg["n"]
        print(f"[jepa] epoch {ep+1}/{args.epochs}  inv={agg['inv']/n:.4f}  "
              f"var={agg['var']/n:.4f}  cov={agg['cov']/n:.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"ctx": ctx_enc.state_dict(), "tgt": tgt_enc.state_dict(),
                "pred": predictor.state_dict(), "args": vars(args)},
               os.path.join(args.out, "mi_jepa.pt"))
    tok.save_pretrained(args.out)
    print(f"[jepa] saved -> {args.out}")
    probe_eval(args, tok, ctx_enc, tgt_enc, predictor, train_rows, val_rows, dev)


# ----------------------------- probe eval -----------------------------
@torch.no_grad()
def embed_all(enc, tok, rows, key, dev, max_len, bs=64):
    enc.eval()
    out = []
    for batch in batched(rows, bs):
        bi = encode_batch(tok, [b[key] for b in batch], dev, max_len)
        out.append(enc(**bi).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def predict_all(predictor, ctx_emb, rows, dev, bs=256):
    predictor.eval()
    out = []
    for i in range(0, len(rows), bs):
        ce = ctx_emb[i:i + bs].to(dev)
        acts = torch.tensor([r["action"] for r in rows[i:i + bs]], device=dev)
        out.append(predictor(ce, acts).cpu())
    return torch.cat(out, 0)


def train_linear_probe(X, y, n_cls, dev, epochs=300, lr=1e-2):
    X = X.to(dev); y = torch.tensor(y, device=dev)
    clf = nn.Linear(X.shape[1], n_cls).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-3)
    # class weights for imbalance
    import collections
    freq = collections.Counter(y.tolist()); tot = len(y)
    w = torch.tensor([tot / (n_cls * freq.get(c, 1)) for c in range(n_cls)], device=dev)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(X), y, weight=w)
        loss.backward(); opt.step()
    return clf


def macro_f1(preds, golds, n):
    f1 = []
    for c in range(n):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return sum(f1) / len(f1)


def probe_eval(args, tok, ctx_enc, tgt_enc, predictor, train_rows, val_rows, dev):
    print("\n[jepa] === linear-probe evaluation ===")
    # Probe trained on TARGET embeddings -> next_talk (how decodable talk-type is)
    Xt_tr = embed_all(tgt_enc, tok, train_rows, "tgt", dev, args.tgt_len)
    Xt_va = embed_all(tgt_enc, tok, val_rows, "tgt", dev, args.tgt_len)
    y_tr = [r["next_talk"] for r in train_rows]
    y_va = [r["next_talk"] for r in val_rows]
    probe = train_linear_probe(Xt_tr, y_tr, len(TALKS), dev)
    with torch.no_grad():
        pt = probe(Xt_va.to(dev)).argmax(-1).cpu().tolist()
    target_acc = sum(p == g for p, g in zip(pt, y_va)) / len(y_va)
    target_f1 = macro_f1(pt, y_va, len(TALKS))

    # JEPA DYNAMICS: predict target emb from (context, action), apply SAME probe
    Xc_va = embed_all(ctx_enc, tok, val_rows, "ctx", dev, args.max_len)
    pred_va = predict_all(predictor, Xc_va, val_rows, dev)
    with torch.no_grad():
        pp = probe(pred_va.to(dev)).argmax(-1).cpu().tolist()
    dyn_acc = sum(p == g for p, g in zip(pp, y_va)) / len(y_va)
    dyn_f1 = macro_f1(pp, y_va, len(TALKS))

    report = {
        "n_val": len(val_rows),
        "target_probe (upper bound, sees reply)": {"acc": round(target_acc, 4), "macro_f1": round(target_f1, 4)},
        "jepa_dynamics (predicts next_talk, no reply)": {"acc": round(dyn_acc, 4), "macro_f1": round(dyn_f1, 4)},
        "tabular_transition_baseline": {"acc": 0.6714, "macro_f1": 0.5606},
    }
    Path("reports").mkdir(exist_ok=True)
    json.dump(report, open("reports/mi_jepa_eval.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("[jepa] wrote -> reports/mi_jepa_eval.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_model/transitions.jsonl")
    ap.add_argument("--out", default="runs/mi_jepa")
    ap.add_argument("--backbone", default="distilbert-base-uncased")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--sim", type=float, default=25.0)
    ap.add_argument("--var", type=float, default=25.0)
    ap.add_argument("--cov", type=float, default=1.0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
