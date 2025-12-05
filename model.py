import math
import sys
import random
from pathlib import Path
import argparse

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def filter_5core(df, user_col, item_col, min_cnt=5):
    while True:
        before = len(df)
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()

        df = df[df[user_col].isin(user_counts[user_counts >= min_cnt].index)]
        df = df[df[item_col].isin(item_counts[item_counts >= min_cnt].index)]

        after = len(df)
        if after == before:
            break
    return df


def load_sequences(
    data_root,
    dataset="",
    user_col="reviewerID",
    item_col="asin",
    time_col="unixReviewTime",
    min_seq_len=3,
):
    data_root = Path(data_root)
    json_path = data_root / dataset
    if not json_path.exists():
        raise FileNotFoundError(f".json not found at: {json_path}")

    df = pd.read_json(json_path, lines=True)

    for col in [user_col, item_col, time_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in .json columns: {df.columns.tolist()}"
            )

    df = df[[user_col, item_col, time_col]].dropna()

    dataset_str = str(dataset).lower()
    if ("beauty" not in dataset_str):
        df = filter_5core(df, user_col=user_col, item_col=item_col, min_cnt=5)

    df = df.sort_values([user_col, time_col])

    user_sequences = {}
    for u, g in df.groupby(user_col):
        seq = g[item_col].tolist()
        if len(seq) >= min_seq_len:
            user_sequences[u] = seq

    if not user_sequences:
        raise ValueError("No user sequences constructed (all shorter than min_seq_len).")

    return user_sequences


def build_item_mapping(user_sequences):
    item_set = set()
    for seq in user_sequences.values():
        item_set.update(seq)
    item2id = {item: idx + 1 for idx, item in enumerate(sorted(item_set))}
    id2item = {v: k for k, v in item2id.items()}
    return item2id, id2item


def build_luo_instances(user_sequences, item2id, max_len=50, alpha=1.0, inject_bias=True):
    train_seqs, train_tgts = [], []
    val_seqs, val_tgts = [], []
    test_seqs, test_tgts = [], []

    num_items = len(item2id)
    item_pop = [0.0] * (num_items + 1)

    mapped_seqs = []
    for seq in user_sequences.values():
        mapped = [item2id[i] for i in seq if i in item2id]
        if len(mapped) < 3:
            continue
        mapped_seqs.append(mapped)
        for iid in mapped:
            item_pop[iid] += 1.0

    if inject_bias:
        pop_pow = [c**alpha for c in item_pop]
        pop_pow[0] = 0.0
        Z = sum(pop_pow[1:])
        if Z > 0.0:
            item_prob = [c / Z for c in pop_pow]
        else:
            item_prob = [0.0] * (num_items + 1)
    else:
        item_prob = None

    for mapped in mapped_seqs:
        T = len(mapped)

        for t in range(1, T - 2):
            src = mapped[:t]
            tgt = mapped[t]
            if len(src) >= max_len:
                src = src[-max_len:]
            else:
                src = [0] * (max_len - len(src)) + src

            if inject_bias:
                p_keep = item_prob[tgt] if item_prob is not None else 1.0
                if p_keep <= 0.0:
                    continue
                if random.random() > p_keep:
                    continue

            train_seqs.append(src)
            train_tgts.append(tgt)

        val_src = mapped[: T - 2]
        val_tgt = mapped[T - 2]
        if len(val_src) >= max_len:
            val_src = val_src[-max_len:]
        else:
            val_src = [0] * (max_len - len(val_src)) + val_src
        val_seqs.append(val_src)
        val_tgts.append(val_tgt)

        test_src = mapped[: T - 1]
        test_tgt = mapped[T - 1]
        if len(test_src) >= max_len:
            test_src = test_src[-max_len:]
        else:
            test_src = [0] * (max_len - len(test_src)) + test_src
        test_seqs.append(test_src)
        test_tgts.append(test_tgt)

    return (train_seqs, train_tgts), (val_seqs, val_tgts), (test_seqs, test_tgts), item_pop


class SeqDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class BaseCalibratedModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.psi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.W_US = nn.Linear(d_model, d_model)
        self.W_gamma = nn.Linear(2 * d_model, 1)
        self.alpha_layer = nn.Linear(d_model, 1)

    def compute_session_rep_from_hidden(self, seq, hidden_states):
        B, L, D = hidden_states.size()
        mask = (seq > 0).unsqueeze(-1)

        lengths = mask.sum(dim=1).clamp(min=1)
        h_masked = hidden_states * mask
        z_U = h_masked.sum(dim=1) / lengths

        k = min(5, L)
        recent_h = hidden_states[:, -k:, :]
        recent_mask = (seq[:, -k:] > 0).unsqueeze(-1)
        recent_len = recent_mask.sum(dim=1).clamp(min=1)
        recent_h_masked = recent_h * recent_mask
        r_mean = recent_h_masked.sum(dim=1) / recent_len

        r = self.rho(self.psi(r_mean))

        concat = torch.cat([z_U, r], dim=-1)
        delta = torch.sigmoid(self.W_gamma(concat))

        z_mix = delta * z_U + (1.0 - delta) * r
        z_US = self.W_US(z_mix)
        return z_US


class SASRec(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=2,
        dropout=0.2,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_items + 1)

    def forward(self, seq, return_session_rep=False):
        device = seq.device
        B, L = seq.size()
        item_emb = self.item_emb(seq)
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.encoder(x)
        logits = self.fc_out(x)

        if return_session_rep:
            z_US = self.compute_session_rep_from_hidden(seq, x)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        out = self.forward(seq, return_session_rep=return_session_rep)
        if return_session_rep:
            logits, z_US = out
            last_logits = logits[:, -1, :]
            return last_logits, z_US
        else:
            logits = out
            last_logits = logits[:, -1, :]
            return last_logits


class BERT4Rec(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=2,
        dropout=0.2,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_items + 1)

    def forward(self, seq, return_session_rep=False):
        device = seq.device
        B, L = seq.size()
        item_emb = self.item_emb(seq)
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.encoder(x)
        logits = self.fc_out(x)
        if return_session_rep:
            z_US = self.compute_session_rep_from_hidden(seq, x)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        out = self.forward(seq, return_session_rep=return_session_rep)
        if return_session_rep:
            logits, z_US = out
            last_logits = logits[:, -1, :]
            return last_logits, z_US
        else:
            logits = out
            last_logits = logits[:, -1, :]
            return last_logits


class DuoRec(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=2,
        dropout=0.2,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_items + 1)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, seq, return_session_rep=False):
        device = seq.device
        B, L = seq.size()
        item_emb = self.item_emb(seq)
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.encoder(x)
        logits = self.fc_out(x)
        if return_session_rep:
            z_US = self.compute_session_rep_from_hidden(seq, x)
            return logits, z_US
        return logits

    def get_seq_rep(self, seq):
        device = seq.device
        B, L = seq.size()
        item_emb = self.item_emb(seq)
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.encoder(x)
        mask = (seq > 0).float()
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1
        h_last = x[torch.arange(B, device=device), lengths]
        z = self.proj_head(h_last)
        return h_last, z

    def predict_next(self, seq, return_session_rep=False, return_proj=False):
        device = seq.device
        B, L = seq.size()
        item_emb = self.item_emb(seq)
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.encoder(x)
        logits = self.fc_out(x)
        last_logits = logits[:, -1, :]
        outputs = (last_logits,)
        if return_session_rep:
            z_US = self.compute_session_rep_from_hidden(seq, x)
            outputs += (z_US,)
        if return_proj:
            mask = (seq > 0).float()
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            h_last = x[torch.arange(B, device=device), lengths]
            z = self.proj_head(h_last)
            outputs += (z,)
        if len(outputs) == 1:
            return outputs[0]
        return outputs


class CORE(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=2,
        dropout=0.2,
        temperature=1.0,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)

        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        self.sess_dropout = nn.Dropout(dropout)
        self.item_dropout = nn.Dropout(dropout)

        self.temperature = temperature

    def _compute_alpha(self, seq):
        mask = (seq > 0).float()          # (B, L)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        alpha = mask / denom              # (B, L)
        return alpha.unsqueeze(-1)        # (B, L, 1)

    def forward(self, seq, return_session_rep=False):
        B, L = seq.size()
        x = self.item_emb(seq)           # (B, L, D)
        x = self.sess_dropout(x)

        alpha = self._compute_alpha(seq) # (B, L, 1)
        seq_output = torch.sum(alpha * x, dim=1)  # (B, D)
        seq_output = F.normalize(seq_output, dim=-1)

        if return_session_rep:
            hidden_states = seq_output.unsqueeze(1).expand(B, L, self.d_model)
            z_US = self.compute_session_rep_from_hidden(seq, hidden_states)
            return seq_output, z_US

        return seq_output

    def predict_next(self, seq, return_session_rep=False):
        out = self.forward(seq, return_session_rep=return_session_rep)
        if return_session_rep:
            seq_output, z_US = out
        else:
            seq_output = out
            z_US = None
    
        item_emb = self.item_emb.weight  # (num_items+1, d_model)
    
        if self.training:
            item_emb = self.item_dropout(item_emb)
    
        item_emb = F.normalize(item_emb, dim=-1)
    
        logits = torch.matmul(seq_output, item_emb.t()) / self.temperature  # (B, N+1)
    
        if return_session_rep:
            return logits, z_US
        return logits



class SRT(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=2,
        dropout=0.2,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _encode(self, seq):
        device = seq.device
        B, L = seq.size()
        item_emb = self.item_emb(seq)
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.encoder(x)
        return x

    def _last_hidden(self, seq, hidden_states):
        device = seq.device
        mask = (seq > 0).float()
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1
        idx = torch.arange(seq.size(0), device=device)
        h_last = hidden_states[idx, lengths]
        return h_last

    def get_seq_rep(self, seq):
        x = self._encode(seq)
        h_last = self._last_hidden(seq, x)
        return h_last

    def forward(self, seq, return_session_rep=False):
        x = self._encode(seq)
        h_last = self._last_hidden(seq, x)
        logits = torch.matmul(h_last, self.item_emb.weight.t())
        if return_session_rep:
            z_US = self.compute_session_rep_from_hidden(seq, x)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        return self.forward(seq, return_session_rep=return_session_rep)


class SRGNN(BaseCalibratedModel):
    def __init__(self, num_items, d_model=128, dropout=0.2, num_steps=1):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.d_model = d_model
        self.num_steps = num_steps

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        self.W_in = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        self.W_z = nn.Linear(d_model, d_model)
        self.U_z = nn.Linear(d_model, d_model, bias=False)
        self.W_r = nn.Linear(d_model, d_model)
        self.U_r = nn.Linear(d_model, d_model, bias=False)
        self.W_h = nn.Linear(d_model, d_model)
        self.U_h = nn.Linear(d_model, d_model, bias=False)

        self.att_W_q = nn.Linear(d_model, d_model, bias=False)
        self.att_W_k = nn.Linear(d_model, d_model)
        self.att_q = nn.Parameter(torch.randn(d_model))
        self.session_proj = nn.Linear(2 * d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _build_graph(self, items, device):
        uniq, inv = torch.unique(items, return_inverse=True)
        n_nodes = uniq.size(0)
        if n_nodes == 1:
            A_in = torch.zeros(1, 1, device=device)
            A_out = torch.zeros(1, 1, device=device)
            return uniq, inv, A_in, A_out
        A_in = torch.zeros(n_nodes, n_nodes, device=device)
        A_out = torch.zeros(n_nodes, n_nodes, device=device)
        src = inv[:-1]
        dst = inv[1:]
        A_in[dst, src] += 1.0
        A_out[src, dst] += 1.0
        row_in = A_in.sum(dim=1, keepdim=True) + 1e-8
        row_out = A_out.sum(dim=1, keepdim=True) + 1e-8
        A_in = A_in / row_in
        A_out = A_out / row_out
        return uniq, inv, A_in, A_out

    def _gnn_propagation(self, h, A_in, A_out):
        for _ in range(self.num_steps):
            m_in = A_in @ self.W_in(h)
            m_out = A_out @ self.W_out(h)
            m = m_in + m_out
            z = torch.sigmoid(self.W_z(m) + self.U_z(h))
            r = torch.sigmoid(self.W_r(m) + self.U_r(h))
            h_tilde = torch.tanh(self.W_h(m) + self.U_h(r * h))
            h = (1.0 - z) * h + z * h_tilde
        return self.dropout(h)

    def forward(self, seq, return_session_rep=False):
        device = seq.device
        B, L = seq.size()
        session_reps = []
        hidden_for_calib = []
        for b in range(B):
            s = seq[b]
            items = s[s > 0]
            if items.numel() == 0:
                h_s = torch.zeros(self.d_model, device=device)
                session_reps.append(h_s)
                hidden_for_calib.append(h_s.view(1, 1, -1).repeat(1, L, 1))
                continue
            uniq, inv, A_in, A_out = self._build_graph(items, device)
            node_emb = self.item_emb(uniq)
            h = self._gnn_propagation(node_emb, A_in, A_out)
            seq_h = h[inv]
            last_h = seq_h[-1]
            att_in = self.att_W_q(seq_h) + self.att_W_k(last_h).unsqueeze(0)
            e = torch.tanh(att_in)
            alpha = torch.softmax(e @ self.att_q, dim=0)
            s_g = (alpha.unsqueeze(-1) * seq_h).sum(dim=0)
            s_cat = torch.cat([s_g, last_h], dim=-1)
            s_rep = self.session_proj(s_cat)
            session_reps.append(s_rep)
            hidden_for_calib.append(s_rep.view(1, 1, -1).repeat(1, L, 1))
        session_batch = torch.stack(session_reps, dim=0)
        logits = session_batch @ self.item_emb.weight.t()
        if return_session_rep:
            hidden_batch = torch.cat(hidden_for_calib, dim=0)
            z_US = self.compute_session_rep_from_hidden(seq, hidden_batch)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        out = self.forward(seq, return_session_rep=return_session_rep)
        return out


class GCSAN(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=1,
        dropout=0.2,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.gnn_W_in = nn.Linear(d_model, d_model, bias=False)
        self.gnn_W_out = nn.Linear(d_model, d_model, bias=False)
        self.gnn_W_z = nn.Linear(d_model, d_model)
        self.gnn_U_z = nn.Linear(d_model, d_model, bias=False)
        self.gnn_W_r = nn.Linear(d_model, d_model)
        self.gnn_U_r = nn.Linear(d_model, d_model, bias=False)
        self.gnn_W_h = nn.Linear(d_model, d_model)
        self.gnn_U_h = nn.Linear(d_model, d_model, bias=False)

        self.gnn_steps = 1

        self.gate_layer = nn.Linear(2 * d_model, d_model)
        self.output_proj = nn.Linear(2 * d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _build_graph(self, items, device):
        uniq, inv = torch.unique(items, return_inverse=True)
        n_nodes = uniq.size(0)
        if n_nodes == 1:
            A_in = torch.zeros(1, 1, device=device)
            A_out = torch.zeros(1, 1, device=device)
            return uniq, inv, A_in, A_out
        A_in = torch.zeros(n_nodes, n_nodes, device=device)
        A_out = torch.zeros(n_nodes, n_nodes, device=device)
        src = inv[:-1]
        dst = inv[1:]
        A_in[dst, src] += 1.0
        A_out[src, dst] += 1.0
        row_in = A_in.sum(dim=1, keepdim=True) + 1e-8
        row_out = A_out.sum(dim=1, keepdim=True) + 1e-8
        A_in = A_in / row_in
        A_out = A_out / row_out
        return uniq, inv, A_in, A_out

    def _gnn_propagation(self, h, A_in, A_out):
        for _ in range(self.gnn_steps):
            m_in = A_in @ self.gnn_W_in(h)
            m_out = A_out @ self.gnn_W_out(h)
            m = m_in + m_out
            z = torch.sigmoid(self.gnn_W_z(m) + self.gnn_U_z(h))
            r = torch.sigmoid(self.gnn_W_r(m) + self.gnn_U_r(h))
            h_tilde = torch.tanh(self.gnn_W_h(m) + self.gnn_U_h(r * h))
            h = (1.0 - z) * h + z * h_tilde
        return self.dropout(h)

    def forward(self, seq, return_session_rep=False):
        device = seq.device
        B, L = seq.size()
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        item_emb_seq = self.item_emb(seq)
        pos_emb_seq = self.pos_emb(positions)
        x_global = item_emb_seq + pos_emb_seq
        x_global = self.dropout(self.layer_norm(x_global))
        global_out = self.self_attn_encoder(x_global)

        session_reps = []
        hidden_for_calib = []

        for b in range(B):
            s = seq[b]
            items = s[s > 0]
            if items.numel() == 0:
                rep = torch.zeros(self.d_model, device=device)
                session_reps.append(rep)
                hidden_for_calib.append(rep.view(1, 1, -1).repeat(1, L, 1))
                continue
            uniq, inv, A_in, A_out = self._build_graph(items, device)
            node_emb = self.item_emb(uniq)
            h_local_nodes = self._gnn_propagation(node_emb, A_in, A_out)
            h_local_seq = h_local_nodes[inv]
            h_global_seq = global_out[b, : items.size(0), :]
            pad_len = L - items.size(0)
            if pad_len > 0:
                pad_local = torch.zeros(pad_len, self.d_model, device=device)
                pad_global = torch.zeros(pad_len, self.d_model, device=device)
                h_local_seq = torch.cat([pad_local, h_local_seq], dim=0)
                h_global_seq = torch.cat([pad_global, h_global_seq], dim=0)
            h_local_seq = h_local_seq[-L:, :]
            h_global_seq = h_global_seq[-L:, :]

            gate_input = torch.cat([h_local_seq, h_global_seq], dim=-1)
            gate = torch.sigmoid(self.gate_layer(gate_input))
            h_comb = gate * h_local_seq + (1.0 - gate) * h_global_seq

            mask = (s > 0).float()
            last_idx = (mask.sum() - 1).long().clamp(min=0)
            last_h = h_comb[last_idx]

            att_scores = torch.matmul(h_comb, last_h)
            att_scores = att_scores + (mask - 1.0) * 1e9
            alpha = torch.softmax(att_scores, dim=0)
            s_g = (alpha.unsqueeze(-1) * h_comb).sum(dim=0)
            s_cat = torch.cat([s_g, last_h], dim=-1)
            s_rep = self.output_proj(s_cat)
            session_reps.append(s_rep)
            hidden_for_calib.append(s_rep.view(1, 1, -1).repeat(1, L, 1))

        session_batch = torch.stack(session_reps, dim=0)
        logits = session_batch @ self.item_emb.weight.t()
        if return_session_rep:
            hidden_batch = torch.cat(hidden_for_calib, dim=0)
            z_US = self.compute_session_rep_from_hidden(seq, hidden_batch)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        out = self.forward(seq, return_session_rep=return_session_rep)
        return out


class GCEGNN(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        dropout=0.2,
        num_steps_local=1,
        num_steps_global=1,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model
        self.num_steps_local = num_steps_local
        self.num_steps_global = num_steps_global

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        self.local_W_in = nn.Linear(d_model, d_model, bias=False)
        self.local_W_out = nn.Linear(d_model, d_model, bias=False)
        self.local_W_z = nn.Linear(d_model, d_model)
        self.local_U_z = nn.Linear(d_model, d_model, bias=False)
        self.local_W_r = nn.Linear(d_model, d_model)
        self.local_U_r = nn.Linear(d_model, d_model, bias=False)
        self.local_W_h = nn.Linear(d_model, d_model)
        self.local_U_h = nn.Linear(d_model, d_model, bias=False)

        self.global_W = nn.Linear(d_model, d_model)
        self.global_gate = nn.Linear(2 * d_model, d_model)
        self.session_proj = nn.Linear(2 * d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _build_local_graph(self, items, device):
        uniq, inv = torch.unique(items, return_inverse=True)
        n_nodes = uniq.size(0)
        if n_nodes == 1:
            A_in = torch.zeros(1, 1, device=device)
            A_out = torch.zeros(1, 1, device=device)
            return uniq, inv, A_in, A_out
        A_in = torch.zeros(n_nodes, n_nodes, device=device)
        A_out = torch.zeros(n_nodes, n_nodes, device=device)
        src = inv[:-1]
        dst = inv[1:]
        A_in[dst, src] += 1.0
        A_out[src, dst] += 1.0
        row_in = A_in.sum(dim=1, keepdim=True) + 1e-8
        row_out = A_out.sum(dim=1, keepdim=True) + 1e-8
        A_in = A_in / row_in
        A_out = A_out / row_out
        return uniq, inv, A_in, A_out

    def _local_gnn(self, h, A_in, A_out):
        for _ in range(self.num_steps_local):
            m_in = A_in @ self.local_W_in(h)
            m_out = A_out @ self.local_W_out(h)
            m = m_in + m_out
            z = torch.sigmoid(self.local_W_z(m) + self.local_U_z(h))
            r = torch.sigmoid(self.local_W_r(m) + self.local_U_r(h))
            h_tilde = torch.tanh(self.local_W_h(m) + self.local_U_h(r * h))
            h = (1.0 - z) * h + z * h_tilde
        return self.dropout(h)

    def _global_episode_gnn(self, h):
        n_nodes = h.size(0)
        if n_nodes <= 1:
            return h
        A = torch.ones(n_nodes, n_nodes, device=h.device) - torch.eye(n_nodes, device=h.device)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
        z = A @ self.global_W(h)
        for _ in range(self.num_steps_global - 1):
            z = A @ self.global_W(z)
        return self.dropout(z)

    def forward(self, seq, return_session_rep=False):
        device = seq.device
        B, L = seq.size()
        session_reps = []
        hidden_for_calib = []

        for b in range(B):
            s = seq[b]
            items = s[s > 0]
            if items.numel() == 0:
                rep = torch.zeros(self.d_model, device=device)
                session_reps.append(rep)
                hidden_for_calib.append(rep.view(1, 1, -1).repeat(1, L, 1))
                continue
            uniq, inv, A_in, A_out = self._build_local_graph(items, device)
            base_emb = self.item_emb(uniq)

            h_local_nodes = self._local_gnn(base_emb, A_in, A_out)
            h_global_nodes = self._global_episode_gnn(base_emb)

            h_local_seq = h_local_nodes[inv]
            h_global_seq = h_global_nodes[inv]

            gate_input = torch.cat([h_local_seq, h_global_seq], dim=-1)
            gate = torch.sigmoid(self.global_gate(gate_input))
            h_seq = gate * h_local_seq + (1.0 - gate) * h_global_seq

            mask = (s > 0).float()
            last_idx = (mask.sum() - 1).long().clamp(min=0)
            last_h = h_seq[last_idx]
            att_scores = torch.matmul(h_seq, last_h)
            alpha = torch.softmax(att_scores, dim=0)
            s_g = (alpha.unsqueeze(-1) * h_seq).sum(dim=0)

            s_cat = torch.cat([s_g, last_h], dim=-1)
            s_rep = self.session_proj(s_cat)
            session_reps.append(s_rep)
            hidden_for_calib.append(s_rep.view(1, 1, -1).repeat(1, L, 1))

        session_batch = torch.stack(session_reps, dim=0)
        logits = session_batch @ self.item_emb.weight.t()
        if return_session_rep:
            hidden_batch = torch.cat(hidden_for_calib, dim=0)
            z_US = self.compute_session_rep_from_hidden(seq, hidden_batch)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        out = self.forward(seq, return_session_rep=return_session_rep)
        return out


class TAGNN(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        d_model=128,
        dropout=0.2,
        num_steps=1,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.d_model = d_model
        self.num_steps = num_steps

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        self.W_in = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.W_z = nn.Linear(d_model, d_model)
        self.U_z = nn.Linear(d_model, d_model, bias=False)
        self.W_r = nn.Linear(d_model, d_model)
        self.U_r = nn.Linear(d_model, d_model, bias=False)
        self.W_h = nn.Linear(d_model, d_model)
        self.U_h = nn.Linear(d_model, d_model, bias=False)

        self.att_W_t = nn.Linear(d_model, d_model)
        self.att_W_s = nn.Linear(d_model, d_model)
        self.att_q = nn.Parameter(torch.randn(d_model))

        self.session_proj = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_graph(self, items, device):
        uniq, inv = torch.unique(items, return_inverse=True)
        n_nodes = uniq.size(0)
        if n_nodes == 1:
            A_in = torch.zeros(1, 1, device=device)
            A_out = torch.zeros(1, 1, device=device)
            return uniq, inv, A_in, A_out
        A_in = torch.zeros(n_nodes, n_nodes, device=device)
        A_out = torch.zeros(n_nodes, n_nodes, device=device)
        src = inv[:-1]
        dst = inv[1:]
        A_in[dst, src] += 1.0
        A_out[src, dst] += 1.0
        row_in = A_in.sum(dim=1, keepdim=True) + 1e-8
        row_out = A_out.sum(dim=1, keepdim=True) + 1e-8
        A_in = A_in / row_in
        A_out = A_out / row_out
        return uniq, inv, A_in, A_out

    def _gnn_propagation(self, h, A_in, A_out):
        for _ in range(self.num_steps):
            m_in = A_in @ self.W_in(h)
            m_out = A_out @ self.W_out(h)
            m = m_in + m_out
            z = torch.sigmoid(self.W_z(m) + self.U_z(h))
            r = torch.sigmoid(self.W_r(m) + self.U_r(h))
            h_tilde = torch.tanh(self.W_h(m) + self.U_h(r * h))
            h = (1.0 - z) * h + z * h_tilde
        return self.dropout(h)

    def forward(self, seq, return_session_rep=False):
        device = seq.device
        B, L = seq.size()
        session_reps = []
        hidden_for_calib = []

        for b in range(B):
            s = seq[b]
            items = s[s > 0]
            if items.numel() == 0:
                rep = torch.zeros(self.d_model, device=device)
                session_reps.append(rep)
                hidden_for_calib.append(rep.view(1, 1, -1).repeat(1, L, 1))
                continue

            uniq, inv, A_in, A_out = self._build_graph(items, device)
            node_emb = self.item_emb(uniq)
            h_nodes = self._gnn_propagation(node_emb, A_in, A_out)
            h_seq = h_nodes[inv]

            last_idx = len(items) - 1
            h_last = h_seq[last_idx]

            att_in = self.att_W_t(h_last).unsqueeze(0) + self.att_W_s(h_seq)
            e = torch.tanh(att_in)
            alpha = torch.softmax(e @ self.att_q, dim=0)
            s_t = (alpha.unsqueeze(-1) * h_seq).sum(dim=0)

            s_cat = torch.cat([s_t, h_last], dim=-1)
            s_rep = self.session_proj(s_cat)

            session_reps.append(s_rep)
            hidden_for_calib.append(s_rep.view(1, 1, -1).repeat(1, L, 1))

        session_batch = torch.stack(session_reps, dim=0)
        logits = session_batch @ self.item_emb.weight.t()
        if return_session_rep:
            hidden_batch = torch.cat(hidden_for_calib, dim=0)
            z_US = self.compute_session_rep_from_hidden(seq, hidden_batch)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        out = self.forward(seq, return_session_rep=return_session_rep)
        return out



class SelfGNN(BaseCalibratedModel):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=1,
        dropout=0.2,
        n_gnn_layers=1,
    ):
        nn.Module.__init__(self)
        BaseCalibratedModel.__init__(self, d_model)
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model
        self.n_gnn_layers = n_gnn_layers

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        self.gnn_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(d_model, d_model, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.inst_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc_out = nn.Linear(d_model, num_items + 1)

    def _build_adj_and_nodes(self, items, device):
        uniq, inverse = torch.unique(items, return_inverse=True)
        n_nodes = uniq.size(0)
        if n_nodes == 1:
            A = torch.zeros(n_nodes, n_nodes, device=device)
        else:
            src = inverse[:-1]
            dst = inverse[1:]
            A = torch.zeros(n_nodes, n_nodes, device=device)
            A[src, dst] = 1.0

        row_sum = A.sum(dim=1, keepdim=True) + 1e-8
        A_norm = A / row_sum
        return uniq, A_norm, inverse

    def _short_term_gnn(self, seq):
        device = seq.device
        B, L = seq.size()
        reps = []
        for b in range(B):
            s = seq[b]
            items = s[s > 0]
            if items.numel() == 0:
                reps.append(torch.zeros(self.d_model, device=device))
                continue
            uniq, A_norm, inv = self._build_adj_and_nodes(items, device)
            e = self.item_emb(uniq)
            for _ in range(self.n_gnn_layers):
                z = torch.matmul(A_norm, e)
                z = F.leaky_relu(z)
                e = z + e
            e = self.gnn_dropout(e)
            reps.append(e.mean(dim=0))
        return torch.stack(reps, dim=0)

    def _interval_and_instance(self, seq):
        device = seq.device
        B, L = seq.size()
        item_e = self.item_emb(seq)
        mask = (seq > 0).unsqueeze(-1).float()
        item_e = item_e * mask

        h_seq, _ = self.gru(item_e)

        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        h_pos = h_seq + pos_emb

        h_attn = self.inst_encoder(h_pos)

        lengths = mask.sum(dim=1).clamp(min=1.0)
        e_bar_u = (h_seq * mask).sum(dim=1) / lengths
        e_tilde_u = (h_attn * mask).sum(dim=1) / lengths
        return e_bar_u, e_tilde_u

    def forward(self, seq, return_session_rep=False):
        short_u = self._short_term_gnn(seq)
        e_bar_u, e_tilde_u = self._interval_and_instance(seq)
        e_hat_u = short_u + e_bar_u + e_tilde_u

        logits = torch.matmul(e_hat_u, self.item_emb.weight.t())

        if return_session_rep:
            B, L = seq.size()
            hidden = e_hat_u.view(B, 1, -1).repeat(1, L, 1)
            z_US = self.compute_session_rep_from_hidden(seq, hidden)
            return logits, z_US
        return logits

    def predict_next(self, seq, return_session_rep=False):
        return self.forward(seq, return_session_rep=return_session_rep)


def build_model(model_name, num_items, max_len=50, d_model=128, n_heads=2, n_layers=2, dropout=0.2):
    model_name = model_name.lower()
    if model_name == "sasrec":
        return SASRec(num_items=num_items, max_len=max_len, d_model=d_model,
                      n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    elif model_name == "bert4rec":
        return BERT4Rec(num_items=num_items, max_len=max_len, d_model=d_model,
                        n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    elif model_name == "duorec":
        return DuoRec(num_items=num_items, max_len=max_len, d_model=d_model,
                      n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    elif model_name == "core":
        return CORE(num_items=num_items, max_len=max_len, d_model=d_model,
                    n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    elif model_name == "srt":
        return SRT(num_items=num_items, max_len=max_len, d_model=d_model,
                   n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    elif model_name == "srgnn":
        return SRGNN(num_items=num_items, d_model=d_model, dropout=dropout)
    elif model_name == "gcsan":
        return GCSAN(num_items=num_items, max_len=max_len, d_model=d_model,
                     n_heads=n_heads, n_layers=1, dropout=dropout)
    elif model_name == "gce-gnn" or model_name == "gcegnn":
        return GCEGNN(num_items=num_items, max_len=max_len, d_model=d_model,
                      dropout=dropout)
    elif model_name == "tagnn":
        return TAGNN(num_items=num_items, d_model=d_model, dropout=dropout)
    elif model_name == "selfgnn":
        return SelfGNN(num_items=num_items, max_len=max_len, d_model=d_model,
                       n_heads=n_heads, n_layers=1, dropout=dropout, n_gnn_layers=1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_hr_ndcg(model, data_loader, device, k=20):
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total = 0
    with torch.no_grad():
        for seq_batch, tgt_batch in data_loader:
            seq_batch = seq_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            logits = model.predict_next(seq_batch)
            _, indices = torch.topk(logits, k, dim=1)
            batch_size = seq_batch.size(0)
            for i in range(batch_size):
                target = tgt_batch[i].item()
                topk_items = indices[i].tolist()
                if target in topk_items:
                    total_hr += 1.0
                    rank = topk_items.index(target)
                    total_ndcg += 1.0 / math.log2(rank + 2)
                total += 1
    if total == 0:
        return 0.0, 0.0
    hr = total_hr / total
    ndcg = total_ndcg / total
    return hr, ndcg


def evaluate_hr_ndcg_neg_sampling(model, data_loader, device, num_items, k=20, num_neg=100):
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total = 0

    with torch.no_grad():
        for seq_batch, tgt_batch in data_loader:
            seq_batch = seq_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            B, L = seq_batch.size()

            logits_all = model.predict_next(seq_batch)

            for i in range(B):
                target = tgt_batch[i].item()
                if target == 0:
                    continue

                user_items = set(seq_batch[i].tolist())
                user_items.add(target)

                negs = []
                while len(negs) < num_neg:
                    neg = random.randint(1, num_items)
                    if neg not in user_items:
                        negs.append(neg)

                candidates = [target] + negs
                cand_ids = torch.tensor(candidates, device=device, dtype=torch.long)
                cand_scores = logits_all[i, cand_ids]

                topk_scores, topk_idx = torch.topk(cand_scores, k)
                topk_items = cand_ids[topk_idx].tolist()

                if target in topk_items:
                    total_hr += 1.0
                    rank = topk_items.index(target)
                    total_ndcg += 1.0 / math.log2(rank + 2)
                total += 1

    if total == 0:
        return 0.0, 0.0
    hr = total_hr / total
    ndcg = total_ndcg / total
    return hr, ndcg


def train_model(
    model_name="sasrec",
    data_root=".",
    dataset="",
    user_col="reviewerID",
    item_col="asin",
    time_col="unixReviewTime",
    max_len=50,
    batch_size=256,
    d_model=128,
    n_heads=2,
    n_layers=2,
    lr=1e-3,
    n_epochs=200,
    patience=20,
    k_eval=20,
    mode=0,
    alpha=1.0,
    lambda_cal=1e-3,
    lambda_l2=1e-6,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading data ({dataset})...")

    user_sequences = load_sequences(
        data_root=data_root,
        dataset=dataset,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        min_seq_len=3,
    )

    item2id, id2item = build_item_mapping(user_sequences)
    (train_seqs, train_tgts), (val_seqs, val_tgts), (test_seqs, test_tgts), item_pop = \
        build_luo_instances(
            user_sequences,
            item2id=item2id,
            max_len=max_len,
            alpha=alpha,
            inject_bias=True,
        )

    train_dataset = SeqDataset(train_seqs, train_tgts)
    val_dataset = SeqDataset(val_seqs, val_tgts)
    test_dataset = SeqDataset(test_seqs, test_tgts)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_items = len(item2id)
    print(
        f"[{model_name}] #users = {len(user_sequences)}, #items = {num_items}, "
        f"#train = {len(train_dataset)}, #val = {len(val_dataset)}, #test = {len(test_dataset)}"
    )

    ds_str = str(dataset).lower()
    use_neg_sampling_eval = ("toys" in ds_str) or ("sports" in ds_str)

    model = build_model(
        model_name=model_name,
        num_items=num_items,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    propensity = None
    if mode == 1:
        eps = 1e-8
        pop = torch.tensor(item_pop, dtype=torch.float)
        pop[0] = 0.0
        if pop[1:].sum() <= 0:
            pop[1:] = 1.0
        pop[1:] = pop[1:] / (pop[1:].sum() + eps)
        pop[0] = eps
        propensity = pop.to(device)

    if mode == 2:
        eps = 1e-8
        pop = torch.tensor(item_pop, dtype=torch.float)
        pop[0] = 0.0
        pop[1:] = pop[1:] + 1e-3
        q_pop = pop.pow(0.75)
        if q_pop[1:].sum() <= 0:
            q_pop[1:] = 1.0
        q_pop[1:] = q_pop[1:] / (q_pop[1:].sum() + eps)
        q_pop[0] = eps

        mu = torch.zeros_like(q_pop)
        if num_items > 0:
            mu[1:] = 1.0 / num_items

        log_q = torch.log(q_pop + eps)

        omega = mu / (q_pop + eps)
        omega_sum = omega[1:].sum() + eps
        w_tilde = omega / omega_sum

        log_q = log_q.to(device)
        w_tilde = w_tilde.to(device)
        mu = mu.to(device)
    else:
        log_q = None
        w_tilde = None
        mu = None

    srt_q = None
    srt_log_q = None
    srt_num_neg = 100
    srt_tau = 1.0
    if model_name.lower() == "srt":
        eps = 1e-8
        pop = torch.tensor(item_pop, dtype=torch.float)
        pop[0] = 0.0
        if pop[1:].sum() <= 0:
            pop[1:] = 1.0
        pop = pop / (pop.sum() + eps)
        pop[0] = eps
        srt_q = pop.to(device)
        srt_log_q = torch.log(srt_q + eps).to(device)

    best_val_ndcg = 0.0
    best_state = None
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for seq_batch, tgt_batch in train_loader:
            seq_batch = seq_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()

            if model_name.lower() == "srt":
                B = seq_batch.size(0)
                seq_rep = model.get_seq_rep(seq_batch)

                with torch.no_grad():
                    neg_ids = torch.multinomial(srt_q[1:], B * srt_num_neg, replacement=True) + 1
                neg_ids = neg_ids.view(B, srt_num_neg).to(device)

                pos_ids = tgt_batch
                pos_emb = model.item_emb(pos_ids)
                neg_emb = model.item_emb(neg_ids)

                pos_logits = (seq_rep * pos_emb).sum(dim=-1) / srt_tau - srt_log_q[pos_ids]
                neg_logits = (seq_rep.unsqueeze(1) * neg_emb).sum(dim=-1) / srt_tau - srt_log_q[neg_ids]

                logits_cand = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
                labels = torch.zeros(B, dtype=torch.long, device=device)

                loss = F.cross_entropy(logits_cand, labels)

            elif mode == 2:
                logits, z_US = model.predict_next(seq_batch, return_session_rep=True)
                logits = torch.clamp(logits, -20.0, 20.0)
                B, _ = logits.size()

                alpha_vec = F.softplus(model.alpha_layer(z_US)).view(B, 1)
                tilde_s = logits - alpha_vec * log_q.unsqueeze(0)
                tilde_s = torch.clamp(tilde_s, -20.0, 20.0)

                log_probs = F.log_softmax(tilde_s, dim=1)
                batch_idx = torch.arange(B, device=device)
                log_p_y = log_probs[batch_idx, tgt_batch]

                iw = w_tilde[tgt_batch]
                iw = torch.where(tgt_batch > 0, iw, torch.zeros_like(iw))

                loss_iw_ce = - (iw * log_p_y).sum() / (iw.sum() + 1e-8)

                probs = F.softmax(tilde_s, dim=1)
                p_mean = probs.mean(dim=0)

                p_nonpad = p_mean[1:]
                mu_nonpad = mu[1:]

                p_nonpad = p_nonpad / (p_nonpad.sum() + 1e-8)
                mu_nonpad = mu_nonpad / (mu_nonpad.sum() + 1e-8)

                eps = 1e-8
                p_nonpad = p_nonpad + eps
                mu_nonpad = mu_nonpad + eps

                l_cal = (p_nonpad * (torch.log(p_nonpad) - torch.log(mu_nonpad))).sum()

                l2_reg = torch.tensor(0.0, device=device)
                if lambda_l2 > 0.0:
                    for param in model.parameters():
                        l2_reg = l2_reg + param.pow(2).sum()

                loss = loss_iw_ce + lambda_cal * l_cal + lambda_l2 * l2_reg

            elif mode == 1:
                logits = model.predict_next(seq_batch)
                logits = torch.clamp(logits, -20.0, 20.0)
                B, _ = logits.size()

                log_probs = F.log_softmax(logits, dim=1)
                batch_idx = torch.arange(B, device=device)
                log_p_y = log_probs[batch_idx, tgt_batch]

                p_tgt = propensity[tgt_batch]
                iw = 1.0 / (p_tgt + 1e-8)
                iw = torch.where(tgt_batch > 0, iw, torch.zeros_like(iw))

                loss = - (iw * log_p_y).sum() / (iw.sum() + 1e-8)

            else:
                logits = model.predict_next(seq_batch)
                logits = torch.clamp(logits, -20.0, 20.0)
                loss = criterion(logits, tgt_batch)

            if torch.isnan(loss):
                print("NaN loss detected, breaking")
                if best_state is not None:
                    model.load_state_dict(best_state)
                return model, item2id, id2item

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * seq_batch.size(0)

        avg_train_loss = total_loss / len(train_dataset)

        if use_neg_sampling_eval:
            val_hr, val_ndcg = evaluate_hr_ndcg_neg_sampling(
                model, val_loader, device, num_items, k=k_eval, num_neg=100
            )
        else:
            val_hr, val_ndcg = evaluate_hr_ndcg(
                model, val_loader, device, k=k_eval
            )

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(
            f"[{model_name}] Epoch {epoch:03d} | Train Loss = {avg_train_loss:.4f} "
            f"| Val HR@{k_eval} = {val_hr:.4f} | Val NDCG@{k_eval} = {val_ndcg:.4f} "
            f"| Best Epoch = {best_epoch} (NDCG={best_val_ndcg:.4f})"
        )

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if use_neg_sampling_eval:
        test_hr, test_ndcg = evaluate_hr_ndcg_neg_sampling(
            model, test_loader, device, num_items, k=k_eval, num_neg=100
        )
    else:
        test_hr, test_ndcg = evaluate_hr_ndcg(
            model, test_loader, device, k=k_eval
        )

    print(f"[{model_name}] Final Test HR@{k_eval} = {test_hr:.4f} | Test NDCG@{k_eval} = {test_ndcg:.4f}")

    return model, item2id, id2item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["toys", "beauty", "sports", "yelp"],
        default="toys",
        help="Dataset selection",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sasrec", "bert4rec", "duorec", "core", "srt", 
                 "srgnn", "gcsan", "gce-gnn", "tagnn", "selfgnn"],
        default="sasrec",
        help="Baseline model",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0: vanilla CE, 1: IPS/SNIPS, 2: BaSe",
    )
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--k_eval", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_cal", type=float, default=1e-3)
    parser.add_argument("--lambda_l2", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "toys":
        data_root = "./datasets/"
        dataset_file = "Toys.json"
        user_col = "reviewerID"
        item_col = "asin"
        time_col = "unixReviewTime"
    elif args.dataset == "sports":
        data_root = "./datasets/"
        dataset_file = "Sports.json"
        user_col = "reviewerID"
        item_col = "asin"
        time_col = "unixReviewTime"
    elif args.dataset == "beauty":
        data_root = "./datasets/"
        dataset_file = "Beauty.json"
        user_col = "reviewerID"
        item_col = "asin"
        time_col = "unixReviewTime"
    elif args.dataset == "yelp":
        data_root = "./datasets/yelp"
        dataset_file = "yelp_academic_dataset_review.json"
        user_col = "user_id"
        item_col = "business_id"
        time_col = "date"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Running model = {args.model}")
    print(f"Running mode  = {args.mode} (0: CE, 1: IPS/SNIPS, 2: BaSe)")
    print(f"Using dataset = {args.dataset} ({dataset_file})")

    model, item2id, id2item = train_model(
        model_name=args.model,
        data_root=data_root,
        dataset=dataset_file,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        max_len=args.max_len,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        patience=args.patience,
        k_eval=args.k_eval,
        mode=args.mode,
        alpha=args.alpha,
        lambda_cal=args.lambda_cal,
        lambda_l2=args.lambda_l2,
        lr=args.lr,
        device=args.device,
    )
