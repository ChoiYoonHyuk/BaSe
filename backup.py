import json
from collections import defaultdict
import random
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_amazon_beauty(path, min_uc=5, min_ic=5):
    interactions = []
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            uid = obj["reviewerID"]
            iid = obj["asin"]
            t = int(obj["unixReviewTime"])
            interactions.append((uid, iid, t))
    changed = True
    while changed:
        changed = False
        user_count = defaultdict(int)
        item_count = defaultdict(int)
        for u, i, _ in interactions:
            user_count[u] += 1
            item_count[i] += 1
        filtered = []
        for u, i, t in interactions:
            if user_count[u] >= min_uc and item_count[i] >= min_ic:
                filtered.append((u, i, t))
        if len(filtered) != len(interactions):
            changed = True
            interactions = filtered
    user_interactions = defaultdict(list)
    for u, i, t in interactions:
        user_interactions[u].append((t, i))
    user_sequences = {}
    for u, li in user_interactions.items():
        li.sort(key=lambda x: x[0])
        seq = [i for _, i in li]
        if len(seq) >= 3:
            user_sequences[u] = seq
    return user_sequences


def build_item_mapping(user_sequences):
    item_set = set()
    for seq in user_sequences.values():
        item_set.update(seq)
    item2id = {item: idx + 1 for idx, item in enumerate(sorted(item_set))}
    id2item = {v: k for k, v in item2id.items()}
    return item2id, id2item


def build_luo_instances(user_sequences, item2id, max_len=50):
    train_seqs, train_tgts = [], []
    val_seqs, val_tgts = [], []
    test_seqs, test_tgts = [], []
    for seq in user_sequences.values():
        mapped = [item2id[i] for i in seq if i in item2id]
        if len(mapped) < 3:
            continue
        T = len(mapped)
        for t in range(1, T - 2):
            src = mapped[:t]
            tgt = mapped[t]
            if len(src) >= max_len:
                src = src[-max_len:]
            else:
                src = [0] * (max_len - len(src)) + src
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
    return (train_seqs, train_tgts), (val_seqs, val_tgts), (test_seqs, test_tgts)


class BeautySeqDataset(Dataset):
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

    def forward(self, seq):
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
        return logits

    def predict_next(self, seq):
        logits = self.forward(seq)
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
            scores, indices = torch.topk(logits, k, dim=1)
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
    data_path="Beauty.json",
    max_len=50,
    batch_size=256,
    d_model=128,
    n_heads=2,
    n_layers=2,
    lr=1e-3,
    n_epochs=200,
    patience=20,
    k_eval=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print("Loading data...")
    user_sequences = load_amazon_beauty(data_path, min_uc=5, min_ic=5)
    item2id, id2item = build_item_mapping(user_sequences)
    (train_seqs, train_tgts), (val_seqs, val_tgts), (test_seqs, test_tgts) = build_luo_instances(
        user_sequences, item2id, max_len=max_len
    )

    train_dataset = BeautySeqDataset(train_seqs, train_tgts)
    val_dataset = BeautySeqDataset(val_seqs, val_tgts)
    test_dataset = BeautySeqDataset(test_seqs, test_tgts)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_items = len(item2id)
    print(
        f"#users = {len(user_sequences)}, #items = {num_items}, "
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
    model, item2id, id2item = train_sasrec(
        data_path="Beauty.json",
        max_len=50,
        batch_size=256,
        n_epochs=200,
        patience=20,
        k_eval=20,
    )
