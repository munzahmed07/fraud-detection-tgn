"""
Module 10: GNN Explainer — Network-Level Fraud Explanation
AI-Powered Mule Account and Financial Fraud Network Detection System

Answers: "Why was THIS account flagged in THIS network?"

Uses torch_geometric.explain.GNNExplainer to identify:
  1. Which neighbouring accounts influenced the fraud prediction
  2. Which transaction edges are most suspicious
  3. Visualises the fraud subgraph for top flagged accounts

This goes beyond SHAP (feature-level) to network-level explanation.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer

os.makedirs("output/gnn_explanations", exist_ok=True)

print(f"[INFO] PyTorch {torch.__version__}")


# ─────────────────────────────────────────────
#  1. Lightweight GNN for Explanation
#     (GNNExplainer works with standard GNN,
#      not TGN memory — we use a GraphSAGE
#      retrained on the graph for this purpose)
# ─────────────────────────────────────────────

class ExplainableGNN(nn.Module):
    """
    Simple 2-layer GraphSAGE for GNNExplainer compatibility.
    Trained on node features + graph structure.
    GNNExplainer requires a standard message-passing GNN.
    """
    def __init__(self, in_channels, hidden=64, out=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden // 2)
        self.lin   = nn.Linear(hidden // 2, out)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(self.lin(x), dim=1)


# ─────────────────────────────────────────────
#  2. Load & Prepare Graph Data
# ─────────────────────────────────────────────

def load_graph_data():
    print("[INFO] Loading graph data for GNNExplainer...")

    required = ["output/graph_data.npz",
                "output/tgn_fraud_scores.csv",
                "output/transactions_processed.csv"]
    for f in required:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}"); exit(1)

    graph_data = np.load("output/graph_data.npz", allow_pickle=True)
    tgn_scores = pd.read_csv("output/tgn_fraud_scores.csv")
    df         = pd.read_csv("output/transactions_processed.csv")

    # Node features
    from sklearn.preprocessing import StandardScaler
    X_raw  = graph_data['node_features'].astype(np.float32)
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    # Also add TGN fraud probability as a node feature
    node_list  = graph_data['node_list']
    score_map  = tgn_scores.set_index('account')['fraud_prob'].to_dict()
    tgn_scores_arr = np.array([score_map.get(str(n), 0.0)
                                for n in node_list], dtype=np.float32)
    X = np.column_stack([X, tgn_scores_arr])

    x          = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
    labels     = graph_data['labels'].astype(np.int64)
    y          = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.node_list = node_list

    print(f"[INFO] Graph: {data.num_nodes:,} nodes | "
          f"{data.num_edges:,} edges | "
          f"Feature dim: {data.num_node_features} | "
          f"Fraud nodes: {y.sum().item():,}")

    return data, node_list, tgn_scores, df


# ─────────────────────────────────────────────
#  3. Train Explainable GNN
# ─────────────────────────────────────────────

def train_explainable_gnn(data, epochs=80, lr=0.005):
    print("[INFO] Training explainable GNN (GraphSAGE for GNNExplainer)...")

    device = torch.device('cpu')
    model  = ExplainableGNN(data.num_node_features).to(device)
    data   = data.to(device)

    # Class weights for imbalance
    n_fraud  = int(data.y.sum().item())
    n_normal = data.num_nodes - n_fraud
    weight   = torch.tensor([1.0, n_normal / (n_fraud + 1e-6)]).to(device)
    criterion = nn.NLLLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Train/val masks
    idx      = torch.randperm(data.num_nodes)
    tr_end   = int(0.7 * data.num_nodes)
    tr_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    tr_mask[idx[:tr_end]]   = True
    val_mask[idx[tr_end:]]  = True

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[tr_mask], data.y[tr_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                out   = model(data.x, data.edge_index)
                pred  = out.argmax(dim=1)
                fraud_nodes = (data.y == 1)
                recall = (pred[fraud_nodes] == 1).float().mean().item() \
                         if fraud_nodes.sum() > 0 else 0
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                  f"Fraud Recall: {recall:.3f}")
            if recall > best_val:
                best_val = recall
                torch.save(model.state_dict(), "output/explainable_gnn.pt")

    model.load_state_dict(torch.load("output/explainable_gnn.pt",
                                      map_location='cpu'))
    print(f"[INFO] Best fraud recall: {best_val:.3f}")
    return model


# ─────────────────────────────────────────────
#  4. Run GNNExplainer
# ─────────────────────────────────────────────

def run_gnn_explainer(model, data, tgn_scores, top_n=10):
    """
    Run GNNExplainer on top N highest-risk accounts.
    Returns explanations with important edges and node masks.
    """
    print(f"\n[INFO] Running GNNExplainer on top {top_n} flagged accounts...")

    model.eval()

    # PyG Explainer setup
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    # Get top N fraud accounts by TGN score
    top_accounts = tgn_scores.nlargest(top_n, 'fraud_prob')
    node_list    = data.node_list

    # Build account -> node index map
    acc_to_idx = {str(acc): i for i, acc in enumerate(node_list)}

    explanations = []

    for rank, (_, row) in enumerate(top_accounts.iterrows()):
        acc_id   = str(row['account'])
        fraud_prob = float(row['fraud_prob'])
        is_fraud = int(row['is_fraud'])

        if acc_id not in acc_to_idx:
            continue

        node_idx = acc_to_idx[acc_id]

        try:
            explanation = explainer(
                data.x, data.edge_index,
                index=node_idx
            )

            # Extract important edges
            edge_mask = explanation.edge_mask.detach().numpy()
            edge_index_np = data.edge_index.numpy()

            # Top important edges (top 10% by mask score)
            threshold    = np.percentile(edge_mask, 90)
            important_edges = np.where(edge_mask >= threshold)[0]

            important_connections = []
            for eid in important_edges[:15]:  # limit to top 15
                src_idx = int(edge_index_np[0, eid])
                dst_idx = int(edge_index_np[1, eid])
                src_acc = str(node_list[src_idx])
                dst_acc = str(node_list[dst_idx])
                important_connections.append({
                    'src': src_acc,
                    'dst': dst_acc,
                    'edge_importance': round(float(edge_mask[eid]), 4),
                    'src_is_fraud': int(data.y[src_idx].item()),
                    'dst_is_fraud': int(data.y[dst_idx].item()),
                })

            # Node feature importance from mask
            node_mask = explanation.node_mask.detach().numpy()
            feature_names = [
                'in_degree', 'out_degree', 'pagerank', 'betweenness',
                'graph_risk_score', 'balance_drained_count',
                'fan_in_fan_out_ratio', 'tgn_fraud_prob'
            ]
            feat_importance = {}
            if node_mask.shape[-1] == len(feature_names):
                feat_importance = {
                    feature_names[i]: round(float(node_mask.mean(0)[i]), 4)
                    for i in range(len(feature_names))
                }

            exp_dict = {
                'rank':                   rank + 1,
                'account_id':             acc_id,
                'fraud_prob':             round(fraud_prob, 4),
                'is_fraud':               is_fraud,
                'node_idx':               node_idx,
                'important_connections':  important_connections,
                'feature_importance':     feat_importance,
                'n_important_edges':      len(important_connections),
            }
            explanations.append(exp_dict)
            print(f"  [{rank+1:02d}] Account {acc_id} "
                  f"(fraud_prob={fraud_prob:.3f}) — "
                  f"{len(important_connections)} key connections found")

        except Exception as e:
            print(f"  [WARN] Could not explain {acc_id}: {e}")
            continue

    print(f"[INFO] GNNExplainer completed for {len(explanations)} accounts")
    return explanations


# ─────────────────────────────────────────────
#  5. Visualise Explanation Subgraphs
# ─────────────────────────────────────────────

def visualise_explanation(explanation, data, save_path=None):
    """
    Visualise the important subgraph around a flagged account.
    Shows which neighbours and edges triggered the fraud alert.
    """
    acc_id  = explanation['account_id']
    prob    = explanation['fraud_prob']
    conns   = explanation['important_connections']

    if not conns:
        return

    if save_path is None:
        save_path = f"output/gnn_explanations/subgraph_{acc_id[:12]}.png"

    # Build subgraph
    G = nx.DiGraph()
    G.add_node(acc_id, is_target=True,
               is_fraud=explanation['is_fraud'])

    for c in conns:
        G.add_node(c['src'],
                   is_target=(c['src'] == acc_id),
                   is_fraud=c['src_is_fraud'])
        G.add_node(c['dst'],
                   is_target=(c['dst'] == acc_id),
                   is_fraud=c['dst_is_fraud'])
        G.add_edge(c['src'], c['dst'],
                   weight=c['edge_importance'])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    pos = nx.spring_layout(G, k=1.5, seed=42)

    # Node colors
    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        nd = G.nodes[node]
        if nd.get('is_target'):
            node_colors.append('#ff4b4b')   # Red = target account
            node_sizes.append(800)
        elif nd.get('is_fraud'):
            node_colors.append('#e67e22')   # Orange = other fraud node
            node_sizes.append(400)
        else:
            node_colors.append('#3498db')   # Blue = normal neighbour
            node_sizes.append(200)

    # Edge colors by importance
    edge_colors  = []
    edge_widths  = []
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    for u, v in G.edges():
        w = G[u][v]['weight'] / (max_w + 1e-9)
        edge_colors.append(plt.cm.Reds(0.3 + 0.7 * w))
        edge_widths.append(0.5 + 3.0 * w)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           width=edge_widths, alpha=0.8,
                           arrows=True, arrowsize=15,
                           connectionstyle='arc3,rad=0.1')

    # Label only the target and fraud nodes
    labels = {n: n[:10] for n in G.nodes()
              if G.nodes[n].get('is_target') or G.nodes[n].get('is_fraud')}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_color='white', font_size=7, ax=ax)

    legend_elements = [
        mpatches.Patch(color='#ff4b4b', label=f'Target Account ({acc_id[:12]})'),
        mpatches.Patch(color='#e67e22', label='Other Fraud Node'),
        mpatches.Patch(color='#3498db', label='Normal Neighbour'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              facecolor='#2c2c54', labelcolor='white', fontsize=8)

    ax.set_title(
        f"GNNExplainer Subgraph — Account {acc_id}\n"
        f"Fraud Probability: {prob*100:.1f}%  |  "
        f"{len(conns)} key connections highlighted\n"
        f"Edge thickness = importance to fraud prediction",
        color='white', fontsize=10, pad=12
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def visualise_all_explanations(explanations, data, top_n=5):
    print(f"\n[INFO] Generating subgraph visualisations for top {top_n} accounts...")
    for exp in explanations[:top_n]:
        visualise_explanation(exp, data)


# ─────────────────────────────────────────────
#  6. Summary Chart
# ─────────────────────────────────────────────

def plot_explanation_summary(explanations, save_path="output/gnn_explanations/explanation_summary.png"):
    """Bar chart showing number of important connections per flagged account."""
    if not explanations:
        return

    accounts = [e['account_id'][:12] for e in explanations]
    n_conns  = [e['n_important_edges'] for e in explanations]
    colors   = ['#ff4b4b' if e['is_fraud'] else '#3498db'
                for e in explanations]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GNNExplainer — Network-Level Fraud Explanation Summary",
                 fontsize=13, fontweight='bold')

    # Important connections per account
    ax = axes[0]
    bars = ax.bar(range(len(accounts)), n_conns,
                  color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(accounts)))
    ax.set_xticklabels(accounts, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Number of Important Connections")
    ax.set_title("Key Network Connections per Flagged Account\n(Red = confirmed fraud)")
    legend_elements = [
        mpatches.Patch(color='#ff4b4b', label='Confirmed Fraud'),
        mpatches.Patch(color='#3498db', label='Suspected (High Risk)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # Fraud probability vs connections
    ax2 = axes[1]
    probs = [e['fraud_prob'] for e in explanations]
    fraud_flags = [e['is_fraud'] for e in explanations]
    scatter_colors = ['#ff4b4b' if f else '#3498db' for f in fraud_flags]
    ax2.scatter(n_conns, probs, c=scatter_colors, s=80, alpha=0.8, edgecolors='white')
    ax2.set_xlabel("Number of Important Network Connections")
    ax2.set_ylabel("Fraud Probability")
    ax2.set_title("Network Connectivity vs Fraud Probability")
    ax2.set_ylim([0, 1.1])
    ax2.grid(alpha=0.3)
    ax2.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Summary chart saved to: {save_path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 10: GNN EXPLAINER")
    print("  Network-Level Fraud Explanation")
    print("=" * 60)

    # ── Step 1: Load graph data ──
    data, node_list, tgn_scores, df = load_graph_data()

    # ── Step 2: Train explainable GNN ──
    model = train_explainable_gnn(data, epochs=80)

    # ── Step 3: Run GNNExplainer on top 10 accounts ──
    explanations = run_gnn_explainer(model, data, tgn_scores, top_n=10)

    # ── Step 4: Visualise subgraphs ──
    visualise_all_explanations(explanations, data, top_n=5)

    # ── Step 5: Summary chart ──
    plot_explanation_summary(explanations)

    # ── Step 6: Save JSON ──
    with open("output/gnn_explanations/gnn_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2, default=str)

    # ── Print sample explanation ──
    if explanations:
        print(f"\n[SAMPLE EXPLANATION — Rank 1 Account]")
        print("-" * 55)
        exp = explanations[0]
        print(f"  Account:      {exp['account_id']}")
        print(f"  Fraud Prob:   {exp['fraud_prob']*100:.1f}%")
        print(f"  Key Connections: {exp['n_important_edges']}")
        print(f"\n  Top 5 important network connections:")
        for c in exp['important_connections'][:5]:
            fraud_flag = " [FRAUD]" if c['src_is_fraud'] or c['dst_is_fraud'] else ""
            print(f"    {c['src'][:12]} -> {c['dst'][:12]}"
                  f"  (importance: {c['edge_importance']:.3f}){fraud_flag}")
        print("-" * 55)

    print()
    print("=" * 60)
    print("[✓] GNNExplainer complete.")
    print("    Files generated:")
    print("      - output/gnn_explanations/subgraph_*.png")
    print("      - output/gnn_explanations/explanation_summary.png")
    print("      - output/gnn_explanations/gnn_explanations.json")
    print("    Next: Add results to your report")
    print("=" * 60)
