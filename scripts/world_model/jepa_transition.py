#!/usr/bin/env python3
"""
JEPA last mile — a JEPA-backed transition model that is a DROP-IN for the tabular
TransitionModel (same `predict_dist(prev_talk, action)` interface), so the MPC
planner / counterfactual can use it unchanged once JEPA wins the real-val gate.

Bridge between the planner's discrete state (prev_talk) and JEPA's text-embedding
state: for each prev_talk we hold a set of real AnnoMI client utterances of that
talk-type as *prototype contexts*. predict_dist(prev_talk, action) = average over
those prototypes of  probe( predictor( encode(context), action ) ).

There are only 3 talk-types x 9 actions = 27 cells, so we precompute all of them
at init -> predict_dist is an O(1) dict lookup (fast inside the planner loop).

Champion selection: `runs/world_model_champion.txt` ("tabular" | "jepa") decides
which model production serves; continuous_improve.py writes it from the real-val gate.
"""
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from scripts.world_model.mi_jepa import Encoder, Predictor, load, TALKS, ACT2ID
from scripts.world_model.transition_model import ACTIONS


class JepaTransition:
    def __init__(self, ckpt="runs/mi_jepa", data="data/world_model/transitions.jsonl",
                 n_proto=32, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(f"{ckpt}/mi_jepa.pt", map_location=self.dev)
        a = blob["args"]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.ctx = Encoder(a["backbone"], a["dim"]).to(self.dev); self.ctx.load_state_dict(blob["ctx"]); self.ctx.eval()
        self.pred = Predictor(a["dim"]).to(self.dev); self.pred.load_state_dict(blob["pred"]); self.pred.eval()
        self.max_len = a["max_len"]
        train, _ = load(data)
        self._fit_probe(train)
        self._build_cells(train, n_proto)

    def _emb(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = self.tok(texts[i:i+64], return_tensors="pt", truncation=True,
                           max_length=self.max_len, padding=True)
            enc.pop("token_type_ids", None)
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            with torch.no_grad():
                out.append(self.ctx(**enc).cpu())
        return torch.cat(out, 0)

    def _fit_probe(self, train):
        # probe on PREDICTED latents -> next_talk (standard JEPA readout)
        ctx_emb = self._emb([r["ctx"] for r in train]).to(self.dev)
        acts = torch.tensor([r["action"] for r in train], device=self.dev)
        with torch.no_grad():
            pred = self.pred(ctx_emb, acts)
        y = torch.tensor([r["next_talk"] for r in train], device=self.dev)
        self.probe = torch.nn.Linear(pred.shape[1], len(TALKS)).to(self.dev)
        opt = torch.optim.Adam(self.probe.parameters(), lr=1e-2, weight_decay=1e-3)
        import collections
        freq = collections.Counter(y.tolist())
        w = torch.tensor([len(y)/(len(TALKS)*freq.get(c,1)) for c in range(len(TALKS))], device=self.dev)
        for _ in range(300):
            opt.zero_grad(); F.cross_entropy(self.probe(pred), y, weight=w).backward(); opt.step()

    def _build_cells(self, train, n_proto):
        # prototype context embeddings per prev_talk (real client utterances of that type)
        by_talk = defaultdict(list)
        for r in train:
            # prev_talk proxy: use rows whose NEXT talk equals the type, taking their ctx
            by_talk[TALKS[r["next_talk"]]].append(r["ctx"])
        proto_emb = {}
        for t in TALKS:
            ctxs = by_talk.get(t, [])[:n_proto] or ["(start)"]
            proto_emb[t] = self._emb(ctxs).to(self.dev)
        self.cells = {}
        for pt in TALKS:
            for a in ACTIONS:
                aid = torch.full((proto_emb[pt].shape[0],), ACT2ID[a], device=self.dev)
                with torch.no_grad():
                    pred = self.pred(proto_emb[pt], aid)
                    probs = F.softmax(self.probe(pred), -1).mean(0).cpu().tolist()
                self.cells[(pt, a)] = {TALKS[i]: probs[i] for i in range(len(TALKS))}

    def predict_dist(self, prev_talk, action):
        return self.cells.get((prev_talk, action),
                              self.cells.get((prev_talk, "other"), {t: 1/3 for t in TALKS}))

    def base_change_rate(self):
        return sum(self.cells[(pt, "open_question")]["change"] for pt in TALKS) / 3


def champion(path="runs/world_model_champion.txt"):
    if os.path.exists(path):
        return open(path).read().strip()
    return "tabular"


if __name__ == "__main__":
    jt = JepaTransition()
    print("JEPA predict_dist samples:")
    for pt in TALKS:
        for a in ["open_question", "advice"]:
            d = jt.predict_dist(pt, a)
            print(f"  {pt:8} +{a:14} -> change={d['change']:.2f} sustain={d['sustain']:.2f} neutral={d['neutral']:.2f}")
