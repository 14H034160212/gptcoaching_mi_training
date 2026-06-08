#!/usr/bin/env python3
"""
Re-probe a trained MI-JEPA checkpoint with the standard protocol:
fit the linear probe on the model's OWN predicted latents (train) and evaluate
on predicted latents (val) — the honest "linear readout of the world model".

  python -m scripts.world_model.mi_jepa_probe --ckpt runs/mi_jepa
"""
import argparse
import json

import torch

from scripts.world_model.mi_jepa import (
    Encoder, Predictor, load, embed_all, predict_all, train_linear_probe,
    macro_f1, TALKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mi_jepa")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(f"{args.ckpt}/mi_jepa.pt", map_location=dev)
    a = blob["args"]
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    ctx = Encoder(a["backbone"], a["dim"]).to(dev); ctx.load_state_dict(blob["ctx"])
    tgt = Encoder(a["backbone"], a["dim"]).to(dev); tgt.load_state_dict(blob["tgt"])
    pred = Predictor(a["dim"]).to(dev); pred.load_state_dict(blob["pred"])

    train_rows, val_rows = load(a["data"])
    y_tr = [r["next_talk"] for r in train_rows]
    y_va = [r["next_talk"] for r in val_rows]

    # predicted latents from (context, action)
    Xc_tr = embed_all(ctx, tok, train_rows, "ctx", dev, a["max_len"])
    Xc_va = embed_all(ctx, tok, val_rows, "ctx", dev, a["max_len"])
    P_tr = predict_all(pred, Xc_tr, train_rows, dev)
    P_va = predict_all(pred, Xc_va, val_rows, dev)

    def fit_eval(Xtr, Xva):
        clf = train_linear_probe(Xtr, y_tr, len(TALKS), dev)
        with torch.no_grad():
            pv = clf(Xva.to(dev)).argmax(-1).cpu().tolist()
        acc = sum(p == g for p, g in zip(pv, y_va)) / len(y_va)
        return round(acc, 4), round(macro_f1(pv, y_va, len(TALKS)), 4)

    # standard readout: probe fit on predicted latents
    dyn_acc, dyn_f1 = fit_eval(P_tr, P_va)
    # also probe on context latents directly -> next_talk (context+no explicit action)
    ctx_acc, ctx_f1 = fit_eval(Xc_tr, Xc_va)

    report = {
        "n_val": len(val_rows),
        "jepa_dynamics_readout (probe fit on PREDICTED latents)": {"acc": dyn_acc, "macro_f1": dyn_f1},
        "context_readout (probe fit on CONTEXT latents)": {"acc": ctx_acc, "macro_f1": ctx_f1},
        "tabular_transition_baseline": {"acc": 0.6714, "macro_f1": 0.5606},
    }
    json.dump(report, open("reports/mi_jepa_readout.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
