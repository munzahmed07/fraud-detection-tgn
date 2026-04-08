"""
Module 3: Temporal Graph Network with Temporal Feature Augmentation
AI-Powered Mule Account and Financial Fraud Network Detection System

Architecture: TGN (Rossi et al. 2020) + Temporal Feature Engineering
Reference: https://arxiv.org/abs/2006.10637

Components:
  1. Temporal Feature Engineering
       - Dwell time (time between receiving and forwarding funds)
       - Transaction velocity (how fast money moves through account)
       - Time-window aggregations (rolling fraud signals)
       - Time since last transaction per account

  2. TGN Memory Module
       - Per-node GRU state vector updated after each interaction
       - Captures long-term account behavioral history

  3. Time Encoding (time2vec)
       - Learnable embedding of timestamp differences
       - Encodes recency of interactions

  4. Temporal Attention (TransformerConv)
       - Attention-weighted neighbor aggregation
       - Recent neighbors weighted higher than old ones

  5. Fraud Classifier MLP
       - Takes augmented node embeddings
       - Outputs fraud probability per account
"""

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
from torch_geometric.nn import TGNMemory, TransformerConv, SAGEConv
from torch_geometric.data import Data, TemporalData
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
import pandas as pd
import numpy as np
import os
import sys
import warnings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


device = torch.device('cpu')  # Stable for 374k nodes; GPU OOMs on 4GB VRAM
print(f"[INFO] PyTorch {torch.__version__} | Device: {device}")


# ─────────────────────────────────────────────
#  1. Temporal Feature Engineering
# ─────────────────────────────────────────────

def compute_temporal_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rich temporal features per account.
    These capture the TIME DIMENSION of fraud behavior — the core of TGN.

    Key signals:
      - Dwell time: how long funds sit before being forwarded (mules forward fast)
      - Velocity: transactions per hour
      - Time window aggregations: burst activity in short windows
      - Recency: time since last transaction
    """
    print("[INFO] Computing temporal node features...")
    df = df.copy()

    # ── Sent-side features ──
    sent = df.groupby('nameOrig').agg(
        sent_count=('amount', 'count'),
        sent_total=('amount', 'sum'),
        sent_avg=('amount', 'mean'),
        sent_max=('amount', 'max'),
        first_sent_step=('step', 'min'),
        last_sent_step=('step', 'max'),
        fraud_sent=('isFraud', 'sum'),
        balance_drains=('balance_drained', 'sum'),
        high_risk_sent=('is_high_risk_type', 'sum'),
        burst_send=('step', lambda x: (x.diff().dropna() <= 1).sum()),
    ).reset_index().rename(columns={'nameOrig': 'account'})

    # Transaction velocity: txns per hour of activity
    sent['send_velocity'] = sent['sent_count'] / (
        sent['last_sent_step'] - sent['first_sent_step'] + 1
    )

    # ── Received-side features ──
    recv = df.groupby('nameDest').agg(
        recv_count=('amount', 'count'),
        recv_total=('amount', 'sum'),
        recv_avg=('amount', 'mean'),
        first_recv_step=('step', 'min'),
        last_recv_step=('step', 'max'),
        burst_recv=('step', lambda x: (x.diff().dropna() <= 1).sum()),
    ).reset_index().rename(columns={'nameDest': 'account'})

    recv['recv_velocity'] = recv['recv_count'] / (
        recv['last_recv_step'] - recv['first_recv_step'] + 1
    )

    # ── Merge ──
    features = pd.merge(sent, recv, on='account', how='outer').fillna(0)

    # ── Dwell time: key mule signal ──
    # Time between first receive and first send (how fast they forward)
    features['dwell_time'] = (
        features['first_sent_step'] - features['first_recv_step']
    ).clip(lower=0)

    # Rapid forwarder: dwell <= 2 hours
    features['is_rapid_forwarder'] = (
        features['dwell_time'] <= 2).astype(float)

    # ── Forward ratio ──
    features['forward_ratio'] = (
        features['sent_total'] / (features['recv_total'] + 1e-6)
    ).clip(0, 1)

    # ── Fan-in / fan-out ──
    features['fan_in_out'] = features['recv_count'] * features['sent_count']

    # ── Time span of activity ──
    features['activity_span'] = np.maximum(
        features['last_sent_step'] - features['first_sent_step'],
        features['last_recv_step'] - features['first_recv_step']
    )

    # ── Ground truth ──
    features['is_fraud'] = (features['fraud_sent'] > 0).astype(float)

    print(f"[INFO] Temporal features computed for {len(features):,} accounts")
    print(f"[INFO] Fraud accounts: {features['is_fraud'].sum():.0f}")
    print(
        f"[INFO] Rapid forwarders: {features['is_rapid_forwarder'].sum():.0f}")
    print(
        f"[INFO] Avg dwell (fraud):  {features[features['is_fraud'] == 1]['dwell_time'].mean():.1f}h")
    print(
        f"[INFO] Avg dwell (normal): {features[features['is_fraud'] == 0]['dwell_time'].mean():.1f}h")

    return features


# ─────────────────────────────────────────────
#  2. TGN Memory Module
# ─────────────────────────────────────────────

def build_tgn_memory(n_nodes: int, edge_dim: int,
                     memory_dim: int = 32, time_dim: int = 32):
    """
    Build TGN memory module — per-node GRU state.
    This is the core temporal component from Rossi et al. 2020.
    """
    memory = TGNMemory(
        n_nodes, edge_dim, memory_dim, time_dim,
        message_module=IdentityMessage(edge_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    )
    return memory


class TemporalAttentionEmbedding(nn.Module):
    """
    Temporal graph attention (TransformerConv with time encoding).
    Weights neighbor messages by recency — recent transactions matter more.
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2,
            heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = (last_update[edge_index[0]] - t).float()
        rel_t_enc = self.time_enc(rel_t.unsqueeze(-1))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


# ─────────────────────────────────────────────
#  3. Full TGN + Temporal Features Model
# ─────────────────────────────────────────────

class TGNFraudDetector(nn.Module):
    """
    TGN with temporal feature augmentation.

    Pipeline:
      temporal_features (pre-computed) → Linear projection
                    +
      TGN memory embedding (online, per-event)
                    ↓
      Concatenate → MLP → fraud probability
    """

    def __init__(self, temporal_feat_dim: int, memory_dim: int,
                 embed_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Project temporal features to same space as memory
        self.feat_proj = nn.Sequential(
            Linear(temporal_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(hidden_dim, memory_dim)
        )

        # Combine memory embedding + temporal features
        combined_dim = memory_dim + memory_dim  # memory + temporal projection

        # Fraud classifier
        self.classifier = nn.Sequential(
            Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(hidden_dim, 32),
            nn.ReLU(),
            Linear(32, 1)
        )

    def forward(self, memory_embed, temporal_feats):
        """
        memory_embed:   [N, memory_dim] — from TGN memory module
        temporal_feats: [N, temporal_feat_dim] — pre-computed temporal features
        """
        feat_proj = self.feat_proj(temporal_feats)
        combined = torch.cat([memory_embed, feat_proj], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ─────────────────────────────────────────────
#  4. Data Preparation
# ─────────────────────────────────────────────

TEMPORAL_FEATURE_COLS = [
    'sent_count', 'recv_count', 'sent_total', 'recv_total',
    'sent_avg', 'recv_avg', 'send_velocity', 'recv_velocity',
    'dwell_time', 'is_rapid_forwarder', 'forward_ratio',
    'fan_in_out', 'activity_span', 'balance_drains',
    'high_risk_sent', 'burst_send', 'burst_recv'
]


def prepare_data(csv_path: str = "output/transactions_processed.csv"):
    """Prepare both temporal event sequence and node feature matrix."""
    print("[INFO] Loading transaction data...")

    if not os.path.exists(csv_path):
        print("[ERROR] Run graph_construction.py first.")
        exit(1)

    df = pd.read_csv(csv_path).sort_values('step').reset_index(drop=True)
    print(f"[INFO] {len(df):,} transactions loaded")

    # ── Step 1: Compute temporal node features ──
    temporal_features = compute_temporal_node_features(df)

    # ── Step 2: Build node index ──
    all_accounts = temporal_features['account'].values
    node2idx = {acc: i for i, acc in enumerate(all_accounts)}
    n_nodes = len(node2idx)

    src = torch.tensor([node2idx[a] for a in df['nameOrig']], dtype=torch.long)
    dst = torch.tensor([node2idx[a] for a in df['nameDest']], dtype=torch.long)
    t = torch.tensor(df['step'].values, dtype=torch.long)

    # ── Step 3: Edge features ──
    edge_cols = ['amount_normalized', 'type_encoded', 'is_high_risk_type',
                 'balance_drained', 'dest_balance_unchanged',
                 'amount_to_balance_ratio', 'is_odd_hour']
    avail = [c for c in edge_cols if c in df.columns]
    msg_np = StandardScaler().fit_transform(
        df[avail].fillna(0).values.astype(np.float32))
    msg = torch.tensor(msg_np, dtype=torch.float)
    edge_dim = msg.shape[1]

    # ── Step 4: Node feature matrix (temporal features) ──
    feat_cols = [
        c for c in TEMPORAL_FEATURE_COLS if c in temporal_features.columns]
    feat_np = temporal_features[feat_cols].fillna(0).values.astype(np.float32)
    feat_np = StandardScaler().fit_transform(feat_np)
    node_feats = torch.tensor(feat_np, dtype=torch.float)
    feat_dim = node_feats.shape[1]

    # ── Step 5: Labels ──
    y_node = torch.tensor(
        temporal_features['is_fraud'].values, dtype=torch.float)
    y_edge = torch.tensor(df['isFraud'].values, dtype=torch.float)

    print(
        f"[INFO] Nodes: {n_nodes:,} | Edge dim: {edge_dim} | Feat dim: {feat_dim}")
    print(
        f"[INFO] Fraud nodes: {y_node.sum().int().item():,} ({y_node.mean().item():.3%})")
    print(
        f"[INFO] Fraud edges: {y_edge.sum().int().item():,} ({y_edge.mean().item():.3%})")

    # ── Step 6: Temporal data ──
    data = TemporalData(src=src, dst=dst, t=t, msg=msg)

    n = len(src)
    tr, vl = int(0.70 * n), int(0.85 * n)
    train_data = data[:tr]
    val_data = data[tr:vl]
    test_data = data[vl:]

    print(f"[INFO] Split — Train: {tr:,} | Val: {vl-tr:,} | Test: {n-vl:,}")

    return (data, train_data, val_data, test_data,
            node_feats, y_node, y_edge,
            node2idx, n_nodes, edge_dim, feat_dim)


# ─────────────────────────────────────────────
#  5. TGN Memory Extraction
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_memory_embeddings(memory, gnn, data, n_nodes,
                              memory_dim, batch_size=500):
    """
    Run through all temporal events to build up memory state.
    Returns final memory embedding for every node.
    """
    print("[INFO] Extracting TGN memory embeddings...")
    memory.eval()
    gnn.eval()
    memory.reset_state()
    neighbor_loader = LastNeighborLoader(n_nodes, size=10, device=device)

    for batch in TemporalDataLoader(data, batch_size=batch_size):
        src = batch.src
        dst = batch.dst
        t = batch.t
        msg = batch.msg

        n_id, edge_index, e_id = neighbor_loader(
            torch.cat([src, dst]).unique())
        assoc = torch.empty(n_nodes, dtype=torch.long)
        assoc[n_id] = torch.arange(n_id.size(0))

        z, last_update = memory(n_id)
        z = z.detach()
        last_update = last_update.detach()

        if edge_index.size(1) > 0:
            z = gnn(z, last_update, edge_index,
                    data.t[e_id], data.msg[e_id])

        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

    # Extract final memory for all nodes
    all_node_ids = torch.arange(n_nodes)
    final_memory, _ = memory(all_node_ids)
    print(f"[INFO] Memory extracted: {final_memory.shape}")
    return final_memory.detach()


# ─────────────────────────────────────────────
#  6. Training
# ─────────────────────────────────────────────

def train_classifier(model, memory_embeds, node_feats, y_node,
                     epochs=100, lr=0.001):
    """
    Train the fraud classifier on combined memory + temporal features.
    Uses train/val/test split on nodes.
    """
    print("\n[INFO] Training fraud classifier...")

    # Split nodes (stratified)
    idx = np.arange(len(y_node))
    y_np = y_node.numpy()
    fraud = np.where(y_np == 1)[0]
    normal = np.where(y_np == 0)[0]

    # Sample normal to balance
    n_sample = min(len(normal), len(fraud) * 50)
    normal_s = np.random.choice(normal, size=n_sample, replace=False)
    idx_bal = np.concatenate([fraud, normal_s])
    np.random.shuffle(idx_bal)

    tr = int(0.70 * len(idx_bal))
    vl = int(0.85 * len(idx_bal))
    train_idx = idx_bal[:tr]
    val_idx = idx_bal[tr:vl]
    test_idx = idx_bal[vl:]

    # Class weight
    n_fraud = len(fraud)
    n_normal = len(normal_s)
    pos_wt = torch.tensor([n_normal / (n_fraud + 1e-6)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10)

    best_val_auc = 0.0
    patience_ctr = 0
    patience_limit = 20

    X = torch.cat([memory_embeds, node_feats], dim=-1) if False else None
    # Use model forward properly
    mem = memory_embeds
    feat = node_feats

    print(f"{'Epoch':>6} | {'Loss':>8} | {'Val AUC':>9} | {'Status'}")
    print("-" * 45)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(mem[train_idx], feat[train_idx])
        loss = criterion(logits, y_node[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(mem[val_idx], feat[val_idx])
            val_probs = torch.sigmoid(val_logits).numpy()
            val_labels = y_node[val_idx].numpy()

        if val_labels.sum() > 0:
            val_auc = roc_auc_score(val_labels, val_probs)
        else:
            val_auc = 0.5

        scheduler.step(val_auc)
        status = ""

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            torch.save(model.state_dict(), "output/best_tgn.pt")
            status = "✓ saved"
        else:
            patience_ctr += 1
            if patience_ctr >= patience_limit:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or status:
            print(f"{epoch:>6} | {loss.item():>8.4f} | {val_auc:>9.4f} | {status}")

    print(f"\n[INFO] Best Val AUC: {best_val_auc:.4f}")

    # Test evaluation
    model.load_state_dict(torch.load("output/best_tgn.pt"))
    model.eval()
    with torch.no_grad():
        test_logits = model(mem[test_idx], feat[test_idx])
        test_probs = torch.sigmoid(test_logits).numpy()
        test_labels = y_node[test_idx].numpy()

    test_auc = roc_auc_score(
        test_labels, test_probs) if test_labels.sum() > 0 else 0.5
    print(f"\n[RESULTS] Test Set:")
    print(f"  ROC-AUC:       {test_auc:.4f}")
    if test_labels.sum() > 0:
        ap = average_precision_score(test_labels, test_probs)
        preds = (test_probs >= 0.5).astype(int)
        print(f"  Avg Precision: {ap:.4f}")
        print(classification_report(test_labels, preds,
              target_names=['Normal', 'Fraud'], zero_division=0))

    return model, test_auc


# ─────────────────────────────────────────────
#  7. Main Pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("  MODULE 3: TGN + TEMPORAL FEATURE AUGMENTATION")
    print("  Rossi et al. 2020 — Temporal Graph Networks")
    print("=" * 60)

    MEMORY_DIM = 32
    TIME_DIM = 32
    EMBED_DIM = 32
    BATCH_SIZE = 1000
    EPOCHS = 150

    # ── Step 1: Prepare data ──
    (data, train_data, val_data, test_data,
     node_feats, y_node, y_edge,
     node2idx, n_nodes, edge_dim, feat_dim) = prepare_data()

    # ── Step 2: Build TGN memory + attention ──
    print(f"\n[INFO] Building TGN memory module (Rossi et al. 2020)...")
    memory = build_tgn_memory(n_nodes, edge_dim, MEMORY_DIM, TIME_DIM)
    gnn = TemporalAttentionEmbedding(
        in_channels=MEMORY_DIM, out_channels=EMBED_DIM,
        msg_dim=edge_dim, time_enc=memory.time_enc
    )

    # ── Step 3: Run TGN to get memory embeddings ──
    print(f"[INFO] Running TGN forward pass to build memory state...")
    memory_embeds = extract_memory_embeddings(
        memory, gnn, data, n_nodes, MEMORY_DIM, BATCH_SIZE
    )

    # ── Step 4: Build combined model ──
    model = TGNFraudDetector(
        temporal_feat_dim=feat_dim,
        memory_dim=EMBED_DIM,
        embed_dim=EMBED_DIM,
        hidden_dim=64
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Classifier parameters: {n_params:,}")

    # ── Step 5: Train classifier ──
    model, test_auc = train_classifier(
        model, memory_embeds, node_feats, y_node,
        epochs=EPOCHS, lr=0.001
    )

    # ── Step 6: Get fraud scores for all nodes ──
    print("\n[INFO] Computing fraud scores for all accounts...")
    model.eval()
    with torch.no_grad():
        all_logits = model(memory_embeds, node_feats)
        fraud_probs = torch.sigmoid(all_logits).numpy()

    print(
        f"[INFO] Score range: [{fraud_probs.min():.3f}, {fraud_probs.max():.3f}]")
    print(f"[INFO] High-risk accounts (>0.5): {(fraud_probs > 0.5).sum():,}")

    # ── Step 7: Save outputs ──
    np.save("output/fraud_probabilities.npy", fraud_probs)

    y_np = y_node.numpy().astype(int)
    node_df = pd.DataFrame([
        {'account': acc, 'node_idx': idx, 'is_fraud': int(y_np[idx])}
        for acc, idx in node2idx.items()
    ])
    node_df['fraud_prob'] = fraud_probs[node_df['node_idx'].values]
    node_df = node_df.sort_values('fraud_prob', ascending=False)
    node_df.to_csv("output/tgn_fraud_scores.csv", index=False)

    print(f"\n[TOP 10 HIGHEST RISK ACCOUNTS]")
    print(node_df.head(10)[
          ['account', 'fraud_prob', 'is_fraud']].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"[✓] TGN complete. Test ROC-AUC: {test_auc:.4f}")
    print(f"    Model:  output/best_tgn.pt")
    print(f"    Scores: output/fraud_probabilities.npy")
    print(f"    Next:   Run xai_explainer.py")
    print("=" * 60)
