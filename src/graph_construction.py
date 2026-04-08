"""
Module 2: Transaction Graph Construction
AI-Powered Mule Account and Financial Fraud Network Detection System

Builds a dynamic directed transaction graph where:
  - Nodes = bank accounts
  - Edges = transactions (directed, timestamped, weighted)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from data_ingestion import (
    load_paysim_data,
    generate_synthetic_paysim,
    preprocess,
    get_account_level_features
)


# ─────────────────────────────────────────────
#  Core Graph Builder
# ─────────────────────────────────────────────

def build_transaction_graph(df, account_features, time_window=None):
    print("[INFO] Building transaction graph...")

    if time_window is not None:
        df = df[df['step'] <= time_window].copy()
        print(
            f"[INFO] Time window applied: steps 1–{time_window} ({len(df):,} transactions)")

    G = nx.DiGraph()
    account_lookup = account_features.set_index('account').to_dict('index')
    all_accounts = set(df['nameOrig'].unique()) | set(df['nameDest'].unique())

    for acc in all_accounts:
        attrs = account_lookup.get(acc, {})
        G.add_node(acc,
                   sent_count=attrs.get('sent_count', 0),
                   recv_count=attrs.get('recv_count', 0),
                   sent_total=attrs.get('sent_total', 0),
                   recv_total=attrs.get('recv_total', 0),
                   balance_drained_count=attrs.get('balance_drained_count', 0),
                   fan_in_fan_out_ratio=attrs.get('fan_in_fan_out_ratio', 0),
                   is_mule_candidate=attrs.get('is_mule_candidate', 0),
                   is_fraud=attrs.get('is_fraud', 0))

    for _, row in df.iterrows():
        G.add_edge(
            row['nameOrig'], row['nameDest'],
            amount=row['amount'],
            amount_normalized=row['amount_normalized'],
            step=row['step'],
            hour_of_day=row['hour_of_day'],
            type=row['type'],
            type_encoded=row['type_encoded'],
            is_fraud=row['isFraud'],
            balance_drained=row['balance_drained'],
            dest_balance_unchanged=row['dest_balance_unchanged'],
            is_high_risk_type=row['is_high_risk_type']
        )

    n_fraud_edges = sum(1 for _, _, d in G.edges(
        data=True) if d.get('is_fraud') == 1)
    n_fraud_nodes = sum(1 for _, d in G.nodes(
        data=True) if d.get('is_fraud') == 1)
    print(f"[INFO] Graph built: {G.number_of_nodes():,} nodes | "
          f"{G.number_of_edges():,} edges | "
          f"{n_fraud_nodes} fraud nodes | {n_fraud_edges} fraud edges")
    return G


def compute_graph_risk_metrics(G):
    print("[INFO] Computing graph risk metrics...")

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    pagerank = nx.pagerank(G, weight='amount_normalized', max_iter=200)

    if G.number_of_nodes() <= 2000:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    else:
        print(
            "[WARN] Graph too large for full betweenness — using approximation (k=500)")
        betweenness = nx.betweenness_centrality(G, normalized=True, k=500)

    records = []
    for node, data in G.nodes(data=True):
        records.append({
            'account': node,
            'in_degree': in_degree.get(node, 0),
            'out_degree': out_degree.get(node, 0),
            'pagerank': pagerank.get(node, 0),
            'betweenness': betweenness.get(node, 0),
            'is_fraud': data.get('is_fraud', 0),
            'is_mule_candidate': data.get('is_mule_candidate', 0),
            'balance_drained_count': data.get('balance_drained_count', 0),
            'fan_in_fan_out_ratio': data.get('fan_in_fan_out_ratio', 0),
        })

    metrics_df = pd.DataFrame(records)
    metrics_df['graph_risk_score'] = (
        0.30 * metrics_df['pagerank'] / (metrics_df['pagerank'].max() + 1e-9) +
        0.25 * metrics_df['betweenness'] / (metrics_df['betweenness'].max() + 1e-9) +
        0.25 * metrics_df['in_degree'] / (metrics_df['in_degree'].max() + 1e-9) +
        0.20 * metrics_df['balance_drained_count'] /
        (metrics_df['balance_drained_count'].max() + 1e-9)
    )

    print(f"[INFO] Top 5 high-risk accounts:")
    top5 = metrics_df.sort_values('graph_risk_score', ascending=False).head(5)
    print(top5[['account', 'graph_risk_score', 'in_degree',
          'out_degree', 'is_fraud']].to_string(index=False))
    return metrics_df


def extract_fraud_subgraph(G, max_nodes=80):
    fraud_nodes = [n for n, d in G.nodes(data=True) if d.get('is_fraud') == 1]
    fraud_nodes = fraud_nodes[:max_nodes]
    neighbors = set(fraud_nodes)
    for fn in fraud_nodes:
        neighbors.update(G.predecessors(fn))
        neighbors.update(G.successors(fn))
    subgraph = G.subgraph(neighbors).copy()
    print(f"[INFO] Fraud subgraph: {subgraph.number_of_nodes()} nodes, "
          f"{subgraph.number_of_edges()} edges")
    return subgraph


def visualize_fraud_subgraph(subgraph, metrics_df,
                             save_path="output/fraud_subgraph.png"):
    print("[INFO] Generating fraud subgraph visualization...")
    risk_lookup = metrics_df.set_index('account')['graph_risk_score'].to_dict()

    node_colors, node_sizes = [], []
    for node, data in subgraph.nodes(data=True):
        risk = risk_lookup.get(node, 0)
        if data.get('is_fraud', 0) == 1:
            node_colors.append('#e74c3c')
        elif data.get('is_mule_candidate', 0) == 1:
            node_colors.append('#e67e22')
        else:
            node_colors.append('#3498db')
        node_sizes.append(200 + risk * 1500)

    edge_colors, edge_widths = [], []
    for u, v, data in subgraph.edges(data=True):
        if data.get('is_fraud', 0) == 1:
            edge_colors.append('#c0392b')
            edge_widths.append(2.0)
        elif data.get('is_high_risk_type', 0) == 1:
            edge_colors.append('#e67e22')
            edge_widths.append(1.2)
        else:
            edge_colors.append('#bdc3c7')
            edge_widths.append(0.5)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    pos = nx.spring_layout(subgraph, k=0.8, seed=42)

    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.7, arrows=True,
                           arrowsize=12, connectionstyle='arc3,rad=0.1')

    fraud_labels = {n: n[:8] for n, d in subgraph.nodes(data=True)
                    if d.get('is_fraud', 0) == 1}
    nx.draw_networkx_labels(subgraph, pos, labels=fraud_labels,
                            font_color='white', font_size=6, ax=ax)

    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Confirmed Fraud Account'),
        mpatches.Patch(color='#e67e22', label='Mule Candidate Account'),
        mpatches.Patch(color='#3498db', label='Normal Account'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              facecolor='#2c2c54', labelcolor='white', fontsize=9)
    ax.set_title("Financial Fraud Transaction Network\n(Node size = Graph Risk Score)",
                 color='white', fontsize=13, pad=15)
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(
        save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Visualization saved to: {save_path}")


def visualize_risk_distribution(metrics_df, save_path="output/risk_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Graph Risk Score Analysis", fontsize=14, fontweight='bold')

    ax = axes[0]
    fraud = metrics_df[metrics_df['is_fraud'] == 1]['graph_risk_score']
    normal = metrics_df[metrics_df['is_fraud'] == 0]['graph_risk_score']
    ax.hist(normal, bins=40, alpha=0.6, color='#3498db',
            label='Normal', density=True)
    ax.hist(fraud,  bins=40, alpha=0.7, color='#e74c3c',
            label='Fraud',  density=True)
    ax.set_xlabel('Graph Risk Score')
    ax.set_ylabel('Density')
    ax.set_title('Risk Score Distribution')
    ax.legend()

    ax2 = axes[1]
    top_accounts = metrics_df.sort_values(
        'graph_risk_score', ascending=False).head(15)
    colors = ['#e74c3c' if f else '#3498db' for f in top_accounts['is_fraud']]
    ax2.barh(range(len(top_accounts)),
             top_accounts['graph_risk_score'], color=colors)
    ax2.set_yticks(range(len(top_accounts)))
    ax2.set_yticklabels([a[:10] for a in top_accounts['account']], fontsize=7)
    ax2.set_xlabel('Graph Risk Score')
    ax2.set_title('Top 15 High-Risk Accounts\n(Red = Fraud, Blue = Normal)')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Risk distribution chart saved to: {save_path}")


def export_for_pytorch_geometric(G, metrics_df, save_path="output/graph_data.npz"):
    print("[INFO] Exporting graph data for PyTorch Geometric...")

    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    feature_cols = ['in_degree', 'out_degree', 'pagerank', 'betweenness',
                    'graph_risk_score', 'balance_drained_count', 'fan_in_fan_out_ratio']
    mdf = metrics_df.set_index('account')
    node_features = np.array([
        [mdf.loc[n, col] if n in mdf.index else 0.0 for col in feature_cols]
        for n in nodes
    ], dtype=np.float32)

    edge_index = np.array([
        [node_idx[u], node_idx[v]] for u, v in G.edges()
    ], dtype=np.int64).T

    edge_features = np.array([
        [d.get('amount_normalized', 0), d.get('type_encoded', 0),
         d.get('is_high_risk_type', 0), d.get('balance_drained', 0), d.get('step', 0)]
        for _, _, d in G.edges(data=True)
    ], dtype=np.float32)

    labels = np.array([
        mdf.loc[n, 'is_fraud'] if n in mdf.index else 0
        for n in nodes
    ], dtype=np.int64)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(
        save_path) else '.', exist_ok=True)
    np.savez(save_path, edge_index=edge_index, node_features=node_features,
             edge_features=edge_features, labels=labels, node_list=np.array(nodes))

    print(
        f"[INFO] Exported: {len(nodes):,} nodes, {edge_index.shape[1]:,} edges")
    print(f"[INFO] Saved to: {save_path}")
    print(
        f"[INFO] Fraud nodes: {labels.sum():,} / {len(labels):,} ({labels.mean():.2%})")
    return node_idx


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # ── TOGGLE ──────────────────────────────────────────────────────
    USE_REAL_DATA = True
    PAYSIM_PATH = "../data/raw/PS_20174392719_1491204439457_log.csv"
    # ────────────────────────────────────────────────────────────────

    if USE_REAL_DATA:
        df_raw = load_paysim_data(PAYSIM_PATH)
        # Sample 200k rows for speed — remove for final full run
        if len(df_raw) > 200000:
            print(
                f"[INFO] Sampling 200,000 rows from {len(df_raw):,} for speed...")
            df_raw = df_raw.sample(
                n=200000, random_state=42).reset_index(drop=True)
    else:
        df_raw = generate_synthetic_paysim(n_transactions=5000)

    df = preprocess(df_raw)
    account_features = get_account_level_features(df)
    G = build_transaction_graph(df, account_features)
    metrics_df = compute_graph_risk_metrics(G)

    metrics_df.to_csv("output/graph_metrics.csv", index=False)
    df.to_csv("output/transactions_processed.csv", index=False)
    account_features.to_csv("output/account_features.csv", index=False)

    fraud_subgraph = extract_fraud_subgraph(G, max_nodes=60)
    visualize_fraud_subgraph(fraud_subgraph, metrics_df)
    visualize_risk_distribution(metrics_df)
    export_for_pytorch_geometric(G, metrics_df)

    print("\n[✓] Graph construction pipeline complete.")
    print("    Next step: Run tgnn_model.py")
