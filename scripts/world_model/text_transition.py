#!/usr/bin/env python3
"""
Tier 3 (context-aware dynamics): P(next_talk | client_text, coach_text).

The tabular world model conditions only on (prev_talk, action) — 27 coarse
cells — which the rerank eval showed is too blunt to steer replies per-turn,
and it needs a noisy action tagger to map a reply to one of 9 actions.

This model instead reads the ACTUAL TEXT: it encodes the client's last
utterance and the coach's reply with a frozen all-mpnet-base-v2 encoder and
classifies the resulting client talk-type with a small head. At rerank time it
can score a candidate reply's TEXT directly — no action tagger, full
granularity.

Trains/evals on the AnnoMI gold transitions (same split as the tabular model),
so macro-F1 is directly comparable to the tabular 0.56 (gold) baseline.

  python -m scripts.world_model.text_transition --head logreg
"""
import argparse
import json
import os
import numpy as np

TALKS = ["change", "sustain", "neutral"]
EMB_CACHE = "data/world_model/text_transition_emb.npz"


def coach_utt(r):
    for t in reversed(r["history"]):
        if t["role"] == "therapist" and t.get("text", "").strip():
            return t["text"]
    return ""


def client_prev(r):
    seen = False
    for t in reversed(r["history"]):
        if t["role"] == "therapist":
            seen = True
            continue
        if seen and t["role"] == "client" and t.get("text", "").strip():
            return t["text"]
    return ""


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def embed_all(rows, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Return (E_client, E_coach) mpnet embeddings, cached to disk."""
    if os.path.exists(EMB_CACHE):
        d = np.load(EMB_CACHE)
        if d["n"] == len(rows):
            print(f"[text-trans] loaded cached embeddings ({len(rows)} rows)")
            return d["client"], d["coach"]
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to("cuda").eval()

    def embed(texts):
        out = []
        bs = 64
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            t = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = enc(**t).last_hidden_state
                mask = t["attention_mask"].unsqueeze(-1).float()
                emb = (o * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pool
                emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.cpu().numpy())
            if (i // bs) % 10 == 0:
                print(f"  embed {i}/{len(texts)}")
        return np.vstack(out)

    cl = embed([client_prev(r) for r in rows])
    co = embed([coach_utt(r) for r in rows])
    np.savez(EMB_CACHE, client=cl, coach=co, n=len(rows))
    print(f"[text-trans] cached embeddings -> {EMB_CACHE}")
    return cl, co


def features(E_cl, E_co, rows, use_prev_talk):
    """[client emb | coach emb | |client-coach| ]  (+ prev_talk one-hot)."""
    diff = np.abs(E_cl - E_co)
    X = np.hstack([E_cl, E_co, diff])
    if use_prev_talk:
        pt = np.zeros((len(rows), len(TALKS)), dtype=np.float32)
        for i, r in enumerate(rows):
            pt[i, TALKS.index(r["prev_talk"])] = 1.0
        X = np.hstack([X, pt])
    return X.astype(np.float32)


def macro_f1(y_true, y_pred):
    per = {}
    for c in range(len(TALKS)):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[TALKS[c]] = {
            "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        }
    f1 = sum(per[c]["f1"] for c in TALKS) / len(TALKS)
    acc = float((y_pred == y_true).mean())
    return round(acc, 4), round(f1, 4), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_model/transitions.jsonl")
    ap.add_argument("--head", choices=["logreg", "mlp"], default="logreg")
    ap.add_argument("--no-prev-talk", action="store_true")
    ap.add_argument("--out", default="reports/text_transition_eval.json")
    args = ap.parse_args()

    rows = load_rows(args.data)
    E_cl, E_co = embed_all(rows)
    X = features(E_cl, E_co, rows, use_prev_talk=not args.no_prev_talk)
    y = np.array([TALKS.index(r["next_talk"]) for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    va = ~tr
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"[text-trans] train={tr.sum()} val={va.sum()} feat_dim={X.shape[1]} head={args.head}")

    if args.head == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 multi_class="multinomial")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
    else:
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        dev = "cuda"
        Xt = torch.tensor(Xtr).to(dev); yt = torch.tensor(ytr).to(dev)
        Xv = torch.tensor(Xva).to(dev)
        # class weights for imbalance
        cw = torch.tensor([1.0 / max((ytr == c).sum(), 1) for c in range(3)], dtype=torch.float32)
        cw = (cw / cw.sum() * 3).to(dev)
        net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(256, 3)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=cw)
        net.train()
        for ep in range(80):
            opt.zero_grad()
            loss = lossf(net(Xt), yt)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xv).argmax(-1).cpu().numpy()

    acc, f1, per = macro_f1(yva, pred)
    res = {"head": args.head, "use_prev_talk": not args.no_prev_talk,
           "feat_dim": int(X.shape[1]), "val_acc": acc, "val_macro_f1": f1,
           "per_class": per, "tabular_gold_baseline_macro_f1": 0.5606}
    os.makedirs("reports", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("\n[text-trans] " + json.dumps(res, indent=2))
    print(f"[text-trans] vs tabular gold baseline 0.5606 -> delta {f1 - 0.5606:+.4f}")


if __name__ == "__main__":
    main()
