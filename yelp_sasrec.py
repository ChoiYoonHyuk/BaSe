import math
import sys
import random
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_sequences(
    data_root,
    dataset,
    user_col,
    item_col,
    time_col,
    min_seq_len=3,
):
    data_root = Path(data_root)
    json_path = data_root / dataset
    if not json_path.exists():
        raise FileNotFoundError(f"{dataset} not found at: {json_path}")

    df = pd.read_json(json_path, lines=True)

    for col in [user_col, item_col, time_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {dataset} columns: {df.columns.tolist()}"
            )

    df = df[[user_col, item_col, time_col]].dropna()
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


class SASRec(nn.Module):
    def __init__(
        self,
        num_items,
        max_len=50,
        d_model=128,
        n_heads=2,
        n_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.recent_k = min(5, max_len)

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
            z_US = self.compute_session_rep(seq, x)
            return logits, z_US
        return logits

    def compute_session_rep(self, seq, hidden_states):
        B, L, D = hidden_states.size()
        mask = (seq > 0).unsqueeze(-1)

        lengths = mask.sum(dim=1).clamp(min=1)
        h_masked = hidden_states * mask
        z_U = h_masked.sum(dim=1) / lengths

        k = min(self.recent_k, L)
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


def train_sasrec(
    data_root=".",
    dataset="Beauty.json",
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
    lambda_cal=1e-2,
    lambda_l2=1e-6,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
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
        f"[{dataset}] #users = {len(user_sequences)}, #items = {num_items}, "
        f"#train instances = {len(train_dataset)}, #val = {len(val_dataset)}, #test = {len(test_dataset)}"
    )

    model = SASRec(
        num_items=num_items,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

            if mode == 2:
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

                loss_iw_ce = - (iw * log_p_y).mean()

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
        val_hr, val_ndcg = evaluate_hr_ndcg(model, val_loader, device, k=k_eval)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch:03d} | Train Loss = {avg_train_loss:.4f} "
            f"| Val HR@{k_eval} = {val_hr:.4f} | Val NDCG@{k_eval} = {val_ndcg:.4f} "
            f"| Best Epoch = {best_epoch} (NDCG={best_val_ndcg:.4f})"
        )

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_hr, test_ndcg = evaluate_hr_ndcg(model, test_loader, device, k=k_eval)
    print(f"Final Test HR@{k_eval} = {test_hr:.4f} | Test NDCG@{k_eval} = {test_ndcg:.4f}")

    return model, item2id, id2item


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            mode_idx = int(sys.argv[1])
        except ValueError:
            mode_idx = 0
    else:
        mode_idx = 0

    print(f"Running mode = {mode_idx} (0: SASRec, 1: IPS/SNIPS, 2: BaSe)")

    model, item2id, id2item = train_sasrec(
        data_root="./datasets/",
        dataset="yelp_academic_dataset_review.json",  # Yelp 리뷰 파일
        user_col="user_id",
        item_col="business_id",
        time_col="date",
        max_len=50,
        batch_size=256,
        n_epochs=200,
        patience=20,
        k_eval=20,
        mode=mode_idx,
        alpha=1.0,
    )
