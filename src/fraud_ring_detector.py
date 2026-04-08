"""
Module 5: Fraud Ring Detection using Community Detection
AI-Powered Mule Account and Financial Fraud Network Detection System

Detects coordinated fraud rings by:
  1. Louvain community detection — finds tightly connected account clusters
  2. Fraud ring scoring — ranks communities by fraud density
  3. Suspicious community profiling — characterizes each detected ring
  4. Ring visualization — maps out the full fraud network structure
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import os
import json
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_ingestion import preprocess, get_account_level_features
from graph_construction import build_transaction_graph

# Optional: python-louvain for best community detection
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("[WARN] python-louvain not installed. Using NetworkX greedy modularity.")
    print("       For better results: pip install python-louvain")


# ─────────────────────────────────────────────
#  1. Community Detection
# ─────────────────────────────────────────────

def detect_communities(G: nx.DiGraph) -> dict:
    """
    Detect communities (potential fraud rings) in the transaction graph.

    Uses Louvain algorithm if available (best quality),
    falls back to greedy modularity decomposition.

    Returns:
        dict mapping node -> community_id
    """
    print("[INFO] Running community detection...")

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # Remove isolated nodes for faster processing
    G_undirected.remove_nodes_from(list(nx.isolates(G_undirected)))
    print(f"[INFO] Graph after removing isolates: "
          f"{G_undirected.number_of_nodes():,} nodes, "
          f"{G_undirected.number_of_edges():,} edges")

    if LOUVAIN_AVAILABLE:
        print("[INFO] Using Louvain algorithm...")
        partition = community_louvain.best_partition(G_undirected, random_state=42)
    else:
        print("[INFO] Using greedy modularity decomposition...")
        communities = nx.community.greedy_modularity_communities(G_undirected)
        partition = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                partition[node] = comm_id

    n_communities = len(set(partition.values()))
    print(f"[INFO] Communities detected: {n_communities:,}")
    return partition


# ─────────────────────────────────────────────
#  2. Fraud Ring Scoring
# ─────────────────────────────────────────────

def score_communities(G: nx.DiGraph,
                      partition: dict,
                      min_size: int = 3) -> pd.DataFrame:
    """
    Score each community by its fraud density and suspicious behavior.

    A high-scoring community = likely fraud ring.

    Scoring factors:
      - Fraud node density (% of nodes that are confirmed fraud)
      - Internal transaction volume
      - Balance drain rate within community
      - High-risk transaction type ratio
      - Average node degree (tightly connected = suspicious)
    """
    print("[INFO] Scoring communities for fraud ring likelihood...")

    # Group nodes by community
    community_nodes = defaultdict(list)
    for node, comm_id in partition.items():
        community_nodes[comm_id].append(node)

    records = []
    for comm_id, nodes in community_nodes.items():
        if len(nodes) < min_size:
            continue

        subgraph = G.subgraph(nodes)

        # Node-level stats
        n_nodes      = len(nodes)
        n_fraud      = sum(1 for n in nodes if G.nodes[n].get('is_fraud', 0) == 1)
        fraud_density = n_fraud / n_nodes

        # Edge-level stats
        edges = list(subgraph.edges(data=True))
        n_edges = len(edges)

        if n_edges == 0:
            continue

        total_amount     = sum(d.get('amount', 0) for _, _, d in edges)
        avg_amount       = total_amount / n_edges
        fraud_edges      = sum(1 for _, _, d in edges if d.get('is_fraud', 0) == 1)
        high_risk_edges  = sum(1 for _, _, d in edges if d.get('is_high_risk_type', 0) == 1)
        balance_drains   = sum(1 for _, _, d in edges if d.get('balance_drained', 0) == 1)

        fraud_edge_ratio     = fraud_edges / n_edges
        high_risk_edge_ratio = high_risk_edges / n_edges
        drain_ratio          = balance_drains / n_edges

        # Graph structure stats
        avg_degree    = n_edges / n_nodes
        density       = nx.density(subgraph)

        # Internal vs external connections
        all_edges_orig = sum(1 for n in nodes for _ in G.successors(n))
        internal_ratio = n_edges / (all_edges_orig + 1e-6)

        # ── Community Risk Score ──
        ring_score = (
            0.35 * fraud_density +
            0.20 * fraud_edge_ratio +
            0.15 * high_risk_edge_ratio +
            0.15 * drain_ratio +
            0.10 * min(density * 10, 1.0) +
            0.05 * min(internal_ratio, 1.0)
        )

        records.append({
            'community_id':        comm_id,
            'n_nodes':             n_nodes,
            'n_edges':             n_edges,
            'n_fraud_nodes':       n_fraud,
            'fraud_density':       round(fraud_density, 4),
            'fraud_edge_ratio':    round(fraud_edge_ratio, 4),
            'high_risk_edge_ratio':round(high_risk_edge_ratio, 4),
            'drain_ratio':         round(drain_ratio, 4),
            'total_amount':        round(total_amount, 2),
            'avg_amount':          round(avg_amount, 2),
            'avg_degree':          round(avg_degree, 3),
            'density':             round(density, 6),
            'internal_ratio':      round(internal_ratio, 4),
            'ring_score':          round(ring_score, 4),
            'nodes':               nodes
        })

    community_df = pd.DataFrame(records).sort_values('ring_score', ascending=False)

    # Classify rings
    community_df['ring_label'] = pd.cut(
        community_df['ring_score'],
        bins=[0, 0.05, 0.15, 0.30, 1.0],
        labels=['NORMAL', 'SUSPICIOUS', 'HIGH_RISK', 'FRAUD_RING']
    )

    print(f"\n[COMMUNITY SUMMARY]")
    print(f"  Total communities scored: {len(community_df):,}")
    print(f"  FRAUD_RING:  {(community_df['ring_label'] == 'FRAUD_RING').sum()}")
    print(f"  HIGH_RISK:   {(community_df['ring_label'] == 'HIGH_RISK').sum()}")
    print(f"  SUSPICIOUS:  {(community_df['ring_label'] == 'SUSPICIOUS').sum()}")
    print(f"  NORMAL:      {(community_df['ring_label'] == 'NORMAL').sum()}")

    print(f"\n[TOP 5 FRAUD RINGS]")
    top5 = community_df.head(5)
    for _, row in top5.iterrows():
        print(f"  Community {row['community_id']:>6} | "
              f"Score: {row['ring_score']:.3f} | "
              f"Nodes: {row['n_nodes']:>4} | "
              f"Fraud nodes: {row['n_fraud_nodes']:>3} | "
              f"Label: {row['ring_label']}")

    return community_df


# ─────────────────────────────────────────────
#  3. Fraud Ring Profiling
# ─────────────────────────────────────────────

def profile_fraud_rings(G: nx.DiGraph,
                        community_df: pd.DataFrame,
                        top_n: int = 10) -> list:
    """
    Generate detailed profiles for the top N fraud ring communities.
    Each profile describes the ring's structure, behavior, and key accounts.
    """
    print(f"\n[INFO] Profiling top {top_n} fraud rings...")

    profiles = []
    top_rings = community_df.head(top_n)

    for _, ring in top_rings.iterrows():
        nodes  = ring['nodes']
        subgraph = G.subgraph(nodes)

        # Key accounts in this ring
        fraud_nodes = [n for n in nodes if G.nodes[n].get('is_fraud', 0) == 1]

        # Highest degree node = likely ring coordinator
        degrees = dict(subgraph.degree())
        coordinator = max(degrees, key=degrees.get) if degrees else None

        # Total money flow through ring
        total_flow = sum(d.get('amount', 0) for _, _, d in subgraph.edges(data=True))

        # Transaction types in ring
        types = [d.get('type', 'UNKNOWN') for _, _, d in subgraph.edges(data=True)]
        type_counts = pd.Series(types).value_counts().to_dict()

        profile = {
            'community_id':    int(ring['community_id']),
            'ring_label':      str(ring['ring_label']),
            'ring_score':      float(ring['ring_score']),
            'n_nodes':         int(ring['n_nodes']),
            'n_edges':         int(ring['n_edges']),
            'n_fraud_nodes':   int(ring['n_fraud_nodes']),
            'fraud_density':   float(ring['fraud_density']),
            'total_flow':      float(total_flow),
            'coordinator':     coordinator,
            'fraud_accounts':  fraud_nodes[:10],  # Top 10
            'transaction_types': type_counts,
            'avg_amount':      float(ring['avg_amount']),
            'drain_ratio':     float(ring['drain_ratio']),
        }
        profiles.append(profile)

        print(f"\n  ── Ring {profile['community_id']} [{profile['ring_label']}] ──")
        print(f"     Score:        {profile['ring_score']:.3f}")
        print(f"     Size:         {profile['n_nodes']} accounts, {profile['n_edges']} transactions")
        print(f"     Fraud nodes:  {profile['n_fraud_nodes']} ({profile['fraud_density']:.1%})")
        print(f"     Total flow:   ₹{profile['total_flow']:,.0f}")
        print(f"     Coordinator:  {profile['coordinator']}")
        print(f"     Tx types:     {type_counts}")

    return profiles


# ─────────────────────────────────────────────
#  4. Visualization
# ─────────────────────────────────────────────

def visualize_top_rings(G: nx.DiGraph,
                        community_df: pd.DataFrame,
                        top_n: int = 5,
                        save_path: str = "output/fraud_rings.png"):
    """
    Visualize the top N fraud ring communities side by side.
    Each subplot shows one ring with nodes colored by fraud status.
    """
    print(f"\n[INFO] Visualizing top {top_n} fraud rings...")

    top_rings = community_df.head(top_n)
    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 6))
    fig.patch.set_facecolor('#1a1a2e')

    if top_n == 1:
        axes = [axes]

    for idx, (_, ring) in enumerate(top_rings.iterrows()):
        ax = axes[idx]
        ax.set_facecolor('#1a1a2e')

        nodes    = ring['nodes']
        subgraph = G.subgraph(nodes).copy()

        if subgraph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'Empty', ha='center', va='center',
                    color='white', transform=ax.transAxes)
            continue

        pos = nx.spring_layout(subgraph, k=1.2, seed=42)

        node_colors = []
        node_sizes  = []
        for node in subgraph.nodes():
            is_fraud = G.nodes[node].get('is_fraud', 0)
            if is_fraud:
                node_colors.append('#e74c3c')
                node_sizes.append(300)
            else:
                node_colors.append('#3498db')
                node_sizes.append(150)

        edge_colors = ['#e74c3c' if d.get('is_fraud', 0) else '#555577'
                       for _, _, d in subgraph.edges(data=True)]

        nx.draw_networkx_nodes(subgraph, pos, ax=ax,
                               node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(subgraph, pos, ax=ax,
                               edge_color=edge_colors,
                               width=1.0, alpha=0.6,
                               arrows=True, arrowsize=8)

        label = ring['ring_label']
        score = ring['ring_score']
        color = '#e74c3c' if label == 'FRAUD_RING' else \
                '#e67e22' if label == 'HIGH_RISK' else '#f1c40f'

        ax.set_title(f"Community {ring['community_id']}\n"
                     f"{label} | Score: {score:.3f}\n"
                     f"{ring['n_nodes']} accounts | "
                     f"{ring['n_fraud_nodes']} fraud",
                     color=color, fontsize=9, pad=8)
        ax.axis('off')

    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Fraud Account'),
        mpatches.Patch(color='#3498db', label='Normal Account'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               facecolor='#2c2c54', labelcolor='white',
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Detected Fraud Ring Communities",
                 color='white', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Fraud ring visualization saved to: {save_path}")


def visualize_ring_stats(community_df: pd.DataFrame,
                         save_path: str = "output/ring_stats.png"):
    """Plot community-level fraud ring statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fraud Ring Community Analysis", fontsize=14, fontweight='bold')

    # 1. Ring score distribution
    ax = axes[0]
    colors = community_df['ring_label'].map({
        'FRAUD_RING': '#e74c3c',
        'HIGH_RISK':  '#e67e22',
        'SUSPICIOUS': '#f1c40f',
        'NORMAL':     '#3498db'
    }).fillna('#3498db')
    ax.scatter(community_df['n_nodes'],
               community_df['ring_score'],
               c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Community Size (nodes)')
    ax.set_ylabel('Ring Score')
    ax.set_title('Community Size vs Ring Score')

    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='FRAUD_RING'),
        mpatches.Patch(color='#e67e22', label='HIGH_RISK'),
        mpatches.Patch(color='#f1c40f', label='SUSPICIOUS'),
        mpatches.Patch(color='#3498db', label='NORMAL'),
    ]
    ax.legend(handles=legend_elements, fontsize=7)

    # 2. Fraud density distribution
    ax2 = axes[1]
    ax2.hist(community_df['fraud_density'], bins=40,
             color='#e74c3c', alpha=0.7, edgecolor='white')
    ax2.set_xlabel('Fraud Node Density')
    ax2.set_ylabel('Number of Communities')
    ax2.set_title('Fraud Density per Community')

    # 3. Ring label distribution
    ax3 = axes[2]
    label_counts = community_df['ring_label'].value_counts()
    bar_colors   = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db']
    bars = ax3.bar(label_counts.index, label_counts.values,
                   color=bar_colors[:len(label_counts)], edgecolor='white')
    ax3.set_xlabel('Ring Classification')
    ax3.set_ylabel('Count')
    ax3.set_title('Community Risk Classification')
    for bar, val in zip(bars, label_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Ring stats chart saved to: {save_path}")


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("  MODULE 5: FRAUD RING DETECTION")
    print("=" * 60)

    # ── Check for python-louvain ──
    if not LOUVAIN_AVAILABLE:
        print("\n[TIP] Install python-louvain for better community detection:")
        print("      pip install python-louvain\n")

    # ── Load saved data ──
    if not os.path.exists("output/transactions_processed.csv"):
        print("[ERROR] Run graph_construction.py first.")
        exit(1)

    print("[INFO] Loading saved data...")
    df               = pd.read_csv("output/transactions_processed.csv")
    account_features = pd.read_csv("output/account_features.csv")
    print(f"[INFO] Loaded {len(df):,} transactions, "
          f"{len(account_features):,} accounts")

    # ── Build graph ──
    G = build_transaction_graph(df, account_features)

    print()

    # ── Step 1: Community detection ──
    partition = detect_communities(G)

    print()

    # ── Step 2: Score communities ──
    community_df = score_communities(G, partition, min_size=3)

    print()

    # ── Step 3: Profile top rings ──
    profiles = profile_fraud_rings(G, community_df, top_n=10)

    # ── Step 4: Visualize ──
    visualize_top_rings(G, community_df, top_n=5)
    visualize_ring_stats(community_df)

    # ── Save outputs ──
    # Drop nodes column for CSV (it's a list)
    community_df_csv = community_df.drop(columns=['nodes'])
    community_df_csv.to_csv("output/community_scores.csv", index=False)

    with open("output/fraud_ring_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, default=str)

    # Save node -> community mapping
    partition_df = pd.DataFrame([
        {'account': node, 'community_id': comm_id}
        for node, comm_id in partition.items()
    ])
    partition_df.to_csv("output/node_communities.csv", index=False)

    print()
    print("=" * 60)
    print("[✓] Fraud ring detection complete.")
    print("    Files generated:")
    print("      - output/community_scores.csv")
    print("      - output/fraud_ring_profiles.json")
    print("      - output/node_communities.csv")
    print("      - output/fraud_rings.png")
    print("      - output/ring_stats.png")
    print("    Next step: Run xai_explainer.py")
    print("=" * 60)
