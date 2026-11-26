"""Shared neural network, one for all patients (guidelines §4.1).

    60 s window (6 ch x 60 s @1 Hz  [+ 36 features])
          │  WindowEncoder: small 1D CNN
          ▼
    window embedding ──masked mean over 5 windows──► 5-minute token (+ missing flag, time of day)
          │  bidirectional GRU over the last ``context_h`` hours (48 tokens for 4 h)
          ▼
    hour embedding (32-d) at the forecast time t ──► heads: logit P(event in 1 h), in 24 h

Training:
  1. ``pretrain``: self-supervised masked-token reconstruction on every worn hour of every
     patient (no labels): 30 % of the tokens are hidden and the network predicts their summary
     (per-channel mean/std of the raw window + mean features).
  2. ``finetune``: light supervised step on the pre-seizure labels of the training periods,
     with balanced sampling, keeping a small reconstruction term.
The hour embedding is then the input of each patient's personal layer.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SharedNetConfig
from ..dataset import PatientData

N_RAW = 6
N_FEAT = 36
TARGET_DIM = 2 * N_RAW + N_FEAT


class WindowEncoder(nn.Module):
    def __init__(self, emb: int, use_features: bool, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_RAW, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 32, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.use_features = use_features
        d_in = 32 + (N_FEAT if use_features else 0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, emb), nn.GELU())

    def forward(self, raw: torch.Tensor, feats: torch.Tensor | None) -> torch.Tensor:
        h = self.conv(raw).squeeze(-1)
        if self.use_features:
            h = torch.cat([h, feats], dim=-1)
        return self.out(h)


class SharedNet(nn.Module):
    def __init__(self, cfg: SharedNetConfig):
        super().__init__()
        self.cfg = cfg
        E, H = cfg.window_emb, cfg.hour_emb
        self.encoder = WindowEncoder(E, cfg.use_features, cfg.dropout)
        self.missing = nn.Parameter(torch.zeros(E))
        self.mask_tok = nn.Parameter(torch.zeros(E))
        self.gru = nn.GRU(E + 3, H, batch_first=True, bidirectional=True)
        self.recon = nn.Linear(2 * H, TARGET_DIM)
        self.to_hour = nn.Sequential(nn.Linear(2 * H + 2, H), nn.Tanh())
        self.head = nn.Linear(H, 2)  # [1 h, 24 h]

    def tokens(self, raw, feats, present):
        """raw (B,T,K,C,L), feats (B,T,K,F), present (B,T,K) -> token embeddings (B,T,E), frac."""
        B, T, K = present.shape
        flat = present.reshape(-1)
        emb = torch.zeros(B * T * K, self.cfg.window_emb, device=raw.device)
        if flat.any():
            r = raw.reshape(B * T * K, N_RAW, -1)[flat]
            f = feats.reshape(B * T * K, N_FEAT)[flat]
            emb[flat] = self.encoder(r, f)
        emb = emb.reshape(B, T, K, -1)
        w = present.float().unsqueeze(-1)
        n = w.sum(2)
        tok = (emb * w).sum(2) / n.clamp(min=1)
        frac = n / K
        tok = torch.where(n > 0, tok, self.missing.expand_as(tok))
        return tok, frac

    def forward(self, raw, feats, present, tok_hod, hod_t, token_mask=None):
        tok, frac = self.tokens(raw, feats, present)
        if token_mask is not None:
            tok = torch.where(token_mask.unsqueeze(-1), self.mask_tok.expand_as(tok), tok)
            frac = torch.where(token_mask.unsqueeze(-1), torch.zeros_like(frac), frac)
        x = torch.cat([tok, frac, torch.sin(tok_hod).unsqueeze(-1), torch.cos(tok_hod).unsqueeze(-1)], -1)
        out, h_n = self.gru(x)
        last = torch.cat([h_n[0], h_n[1]], dim=-1)
        hour = self.to_hour(torch.cat([last, torch.sin(hod_t)[:, None], torch.cos(hod_t)[:, None]], -1))
        return {"recon": self.recon(out), "hour_emb": hour, "logits": self.head(hour)}


# ----------------------------------------------------------------------------- batching
class HourBatcher:
    """Builds network inputs for (patient, hour index) pairs from normalised window arrays."""

    def __init__(self, patients: dict[str, PatientData], cfg: SharedNetConfig, window_s: int = 60):
        self.p = patients
        self.cfg = cfg
        self.W = window_s
        self.K = cfg.token_min * 60 // window_s           # windows per token
        self.T = cfg.context_h * 60 // cfg.token_min      # tokens per sample
        for pd_ in patients.values():
            if pd_.feats_n is None:
                raise RuntimeError(f"patient {pd_.pid}: call fit_normalizer first")

    def gather(self, pid: str, hour_idx: np.ndarray, device="cpu") -> dict[str, torch.Tensor]:
        d = self.p[pid]
        t = d.hours[hour_idx]
        n_win = self.T * self.K
        start_min = (t // self.W - d.m0 - n_win).astype(np.int64)
        mins = start_min[:, None] + np.arange(n_win)[None, :]
        ok = (mins >= 0) & (mins < len(d.minute_index))
        rows = np.full(mins.shape, -1, dtype=np.int64)
        rows[ok] = d.minute_index[mins[ok]]
        present = rows >= 0
        safe = np.where(present, rows, 0)
        raw = d.raw_n[safe].astype(np.float32)            # (B, n_win, C, L)
        feats = d.feats_n[safe].astype(np.float32)
        raw[~present] = 0
        feats[~present] = 0
        B = len(hour_idx)
        raw = raw.reshape(B, self.T, self.K, N_RAW, -1)
        feats = feats.reshape(B, self.T, self.K, N_FEAT)
        present = present.reshape(B, self.T, self.K)
        # reconstruction target per token: mean/std of raw channels + mean features
        rmean = raw.mean(-1)
        rstd = raw.std(-1)
        tgt = np.concatenate([rmean, rstd, feats], axis=-1)      # (B,T,K,48)
        w = present[..., None].astype(np.float32)
        tok_target = (tgt * w).sum(2) / np.maximum(w.sum(2), 1)
        tok_present = present.any(2)
        tok_time = t[:, None] - (self.T - np.arange(self.T))[None, :] * self.cfg.token_min * 60
        hod_tok = ((tok_time + d.utc_offset(t)[:, None]) % 86400) / 86400 * 2 * math.pi
        hod_t = d.hod[hour_idx] / 24 * 2 * math.pi
        f = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        return {"raw": f(raw), "feats": f(feats), "present": f(present, torch.bool),
                "tok_hod": f(hod_tok), "hod_t": f(hod_t), "target": f(tok_target),
                "tok_present": f(tok_present, torch.bool)}


def _sample_hours(d: PatientData, t_end: float, max_n: int | None, rng, need_worn=True):
    sel = d.hours < t_end
    if need_worn:
        sel &= d.worn
    idx = np.flatnonzero(sel)
    if max_n and len(idx) > max_n:
        idx = rng.choice(idx, max_n, replace=False)
    return idx


def _recon_loss(out, batch, token_mask):
    m = token_mask & batch["tok_present"]
    if not m.any():
        return out["recon"].sum() * 0
    return F.mse_loss(out["recon"][m], batch["target"][m])


# ----------------------------------------------------------------------------- training
def pretrain(model: SharedNet, batcher: HourBatcher, train_end: dict[str, float],
             epochs: int | None = None, verbose: bool = True) -> list[float]:
    cfg = model.cfg
    epochs = cfg.pretrain_epochs if epochs is None else epochs
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    history = []
    for ep in range(epochs):
        model.train()
        jobs = [(pid, i) for pid in batcher.p
                for i in _sample_hours(batcher.p[pid], train_end[pid], cfg.max_hours_per_patient, rng)]
        rng.shuffle(jobs)
        losses, t0 = [], time.time()
        for b in range(0, len(jobs), cfg.batch_size):
            chunk = jobs[b: b + cfg.batch_size]
            loss_sum = 0.0
            opt.zero_grad()
            for pid in {p for p, _ in chunk}:
                idx = np.array([i for p, i in chunk if p == pid])
                batch = batcher.gather(pid, idx, cfg.device)
                tm = torch.rand(batch["tok_present"].shape, device=cfg.device) < cfg.mask_ratio
                out = model(batch["raw"], batch["feats"], batch["present"], batch["tok_hod"],
                            batch["hod_t"], token_mask=tm)
                loss = _recon_loss(out, batch, tm) * len(idx) / len(chunk)
                loss.backward()
                loss_sum += float(loss)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss_sum)
        history.append(float(np.mean(losses)) if losses else float("nan"))
        if verbose:
            print(f"pretrain epoch {ep + 1}/{epochs}: recon loss {history[-1]:.4f} "
                  f"({len(jobs)} hours, {time.time() - t0:.0f}s)")
    return history


def finetune(model: SharedNet, batcher: HourBatcher, train_end: dict[str, float],
             epochs: int | None = None, recon_weight: float = 0.2, verbose: bool = True) -> list[float]:
    """Supervised step on all patients; labels must end before train_end (t + 24 h <= end)."""
    cfg = model.cfg
    epochs = cfg.finetune_epochs if epochs is None else epochs
    rng = np.random.default_rng(cfg.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.5, weight_decay=1e-4)
    pools = {}
    for pid, d in batcher.p.items():
        ok = (d.hours + 24 * 3600 <= train_end[pid]) & d.worn & ~d.post_ictal
        pos = np.flatnonzero(ok & (d.y[24] == 1))
        neg = np.flatnonzero(ok & (d.y[24] == 0))
        pools[pid] = (pos, neg)
    n_pos = sum(len(p) for p, _ in pools.values())
    n_all = sum(len(p) + len(n) for p, n in pools.values())
    if n_pos == 0:
        print("finetune: no positive hours in the training periods, skipped")
        return []
    history = []
    for ep in range(epochs):
        model.train()
        jobs = []
        for pid, (pos, neg) in pools.items():
            k = min(len(neg), max(len(pos), 1) * cfg.neg_per_pos)
            jobs += [(pid, i) for i in pos] + [(pid, i) for i in rng.choice(neg, k, replace=False)]
        rng.shuffle(jobs)
        losses = []
        for b in range(0, len(jobs), cfg.batch_size):
            chunk = jobs[b: b + cfg.batch_size]
            opt.zero_grad()
            tot = 0.0
            for pid in {p for p, _ in chunk}:
                idx = np.array([i for p, i in chunk if p == pid])
                d = batcher.p[pid]
                batch = batcher.gather(pid, idx, cfg.device)
                tm = torch.rand(batch["tok_present"].shape, device=cfg.device) < cfg.mask_ratio * 0.5
                out = model(batch["raw"], batch["feats"], batch["present"], batch["tok_hod"],
                            batch["hod_t"], token_mask=tm)
                y = torch.as_tensor(np.stack([d.y[1][idx], d.y[24][idx]], 1), dtype=torch.float32)
                pw = torch.tensor([3.0, 1.0])  # 1 h positives are rarer inside the balanced set
                loss = F.binary_cross_entropy_with_logits(out["logits"], y, pos_weight=pw)
                loss = (loss + recon_weight * _recon_loss(out, batch, tm)) * len(idx) / len(chunk)
                loss.backward()
                tot += float(loss)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(tot)
        history.append(float(np.mean(losses)))
        if verbose:
            print(f"finetune epoch {ep + 1}/{epochs}: loss {history[-1]:.4f} "
                  f"({n_pos} positive / {n_all} usable hours)")
    return history


@torch.no_grad()
def embed(model: SharedNet, batcher: HourBatcher, pid: str, hour_idx: np.ndarray | None = None,
          batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Hour embeddings and the shared heads' probabilities [P1h, P24h] for every hour index."""
    model.eval()
    d = batcher.p[pid]
    hour_idx = np.arange(len(d.hours)) if hour_idx is None else np.asarray(hour_idx)
    embs, probs = [], []
    for b in range(0, len(hour_idx), batch_size):
        batch = batcher.gather(pid, hour_idx[b: b + batch_size], model.cfg.device)
        out = model(batch["raw"], batch["feats"], batch["present"], batch["tok_hod"], batch["hod_t"])
        embs.append(out["hour_emb"].cpu().numpy())
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
    return np.concatenate(embs), np.concatenate(probs)


def save(model: SharedNet, path):
    torch.save({"state": model.state_dict(), "cfg": model.cfg.__dict__}, path)


def load(path) -> SharedNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SharedNet(SharedNetConfig(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    return m
