"""
Module 4: Transaction Velocity Modeling & Multi-Hop Fraud Path Detection
AI-Powered Mule Account and Financial Fraud Network Detection System

Key fraud signals detected:
  1. Transaction velocity — how fast funds move through an account
  2. Fan-in / Fan-out detection — mule accounts receive from many, send to many
  3. Multi-hop path tracing — trace full fraud path from source to cash-out
  4. Rapid fund forwarding — account receives and immediately forwards
  5. Layering detection — funds passing through N+ intermediary accounts
"""

from graph_construction import build_transaction_graph, compute_graph_risk_metrics
from data_ingestion import generate_synthetic_paysim, preprocess, get_account_level_features
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import json
import os
import sys

# Allow running from src/ directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
#  1. Transaction Velocity Analysis
# ─────────────────────────────────────────────

def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-account transaction velocity features using fast vectorized operations.
    Mule accounts show high send velocity, low dwell time, high forward ratio.
    """
    print("[INFO] Computing transaction velocity features (vectorized)...")

    # ── Sent stats (groupby nameOrig) ──
    sent_stats = df.groupby('nameOrig').agg(
        sent_count=('amount', 'count'),
        sent_total=('amount', 'sum'),
        send_velocity=('step', lambda x: len(x) / (x.max() - x.min() + 1)),
        burst_send_count=('step', lambda x: (x.diff().dropna() <= 1).sum()),
        min_send_step=('step', 'min'),
        max_send_step=('step', 'max'),
        is_fraud=('isFraud', 'max'),
    ).reset_index().rename(columns={'nameOrig': 'account'})

    # ── Received stats (groupby nameDest) ──
    recv_stats = df.groupby('nameDest').agg(
        recv_count=('amount', 'count'),
        recv_total=('amount', 'sum'),
        recv_velocity=('step', lambda x: len(x) / (x.max() - x.min() + 1)),
        burst_recv_count=('step', lambda x: (x.diff().dropna() <= 1).sum()),
        min_recv_step=('step', 'min'),
    ).reset_index().rename(columns={'nameDest': 'account'})

    # ── Merge ──
    velocity_df = pd.merge(sent_stats, recv_stats,
                           on='account', how='outer').fillna(0)

    # ── Dwell time: difference between first receive and first send ──
    velocity_df['avg_dwell_time'] = (
        velocity_df['min_send_step'] - velocity_df['min_recv_step']
    ).clip(lower=0)

    # Rapid forwarder: dwell time <= 2 hours
    velocity_df['rapid_forward_count'] = (
        velocity_df['avg_dwell_time'] <= 2).astype(int)

    # ── Forward ratio ──
    velocity_df['forward_ratio'] = (
        velocity_df['sent_total'] / (velocity_df['recv_total'] + 1e-6)
    ).clip(0, 1)

    # ── Velocity Risk Score ──
    def norm(col):
        mn, mx = velocity_df[col].min(), velocity_df[col].max()
        return (velocity_df[col] - mn) / (mx - mn + 1e-9)

    dwell_risk = 1 - norm('avg_dwell_time').clip(0, 1)

    velocity_df['velocity_risk_score'] = (
        0.30 * norm('send_velocity') +
        0.20 * norm('recv_velocity') +
        0.25 * dwell_risk +
        0.15 * norm('rapid_forward_count') +
        0.10 * norm('forward_ratio')
    ).clip(0, 1)

    fraud_accs = velocity_df[velocity_df['is_fraud'] == 1]
    print(
        f"[INFO] Velocity features computed for {len(velocity_df):,} accounts")
    print(
        f"[INFO] Avg dwell time (fraud):  {fraud_accs['avg_dwell_time'].mean():.1f} hours")
    print(
        f"[INFO] Avg dwell time (normal): {velocity_df[velocity_df['is_fraud'] == 0]['avg_dwell_time'].mean():.1f} hours")
    print(
        f"[INFO] Rapid forwarders (dwell ≤2hr): {velocity_df['rapid_forward_count'].sum():,}")

    return velocity_df


# ─────────────────────────────────────────────
#  2. Fan-in / Fan-out Detection
# ─────────────────────────────────────────────

def detect_fan_in_fan_out(G: nx.DiGraph, threshold_in: int = 3,
                          threshold_out: int = 3) -> pd.DataFrame:
    """
    Detect mule accounts using fan-in / fan-out pattern.

    Fan-in:  Account receives from MANY different senders
    Fan-out: Account then sends to MANY different receivers

    Combined fan-in + fan-out = classic mule layering behavior
    """
    print("[INFO] Detecting fan-in / fan-out patterns...")

    records = []
    for node in G.nodes():
        in_neighbors = list(G.predecessors(node))
        out_neighbors = list(G.successors(node))

        fan_in = len(in_neighbors)
        fan_out = len(out_neighbors)

        # Unique senders and receivers (not just transaction count)
        is_fan_in_out = (fan_in >= threshold_in) and (fan_out >= threshold_out)

        # Amount concentration: does it receive large amounts from few sources?
        in_amounts = [G[u][node].get('amount', 0) for u in in_neighbors]
        out_amounts = [G[node][v].get('amount', 0) for v in out_neighbors]

        records.append({
            'account': node,
            'fan_in': fan_in,
            'fan_out': fan_out,
            'fan_in_out_score': fan_in * fan_out,  # Combined signal
            'is_fan_in_out': int(is_fan_in_out),
            'total_in_amount': sum(in_amounts),
            'total_out_amount': sum(out_amounts),
            'amount_forward_ratio': sum(out_amounts) / (sum(in_amounts) + 1e-6),
            'is_fraud': G.nodes[node].get('is_fraud', 0)
        })

    fandf = pd.DataFrame(records)

    n_flagged = fandf['is_fan_in_out'].sum()
    n_fraud_flagged = fandf[fandf['is_fan_in_out'] == 1]['is_fraud'].sum()

    print(f"[INFO] Fan-in/fan-out accounts flagged: {n_flagged}")
    print(
        f"[INFO] Of which are fraud: {n_fraud_flagged} ({n_fraud_flagged/max(n_flagged, 1):.1%})")

    return fandf


# ─────────────────────────────────────────────
#  3. Multi-Hop Fraud Path Tracing
# ─────────────────────────────────────────────

def trace_fraud_paths(G: nx.DiGraph, df: pd.DataFrame,
                      max_hops: int = 5) -> list:
    """
    Trace complete fraud paths from source account to final cash-out.

    Mule layering pattern:
      Victim → Mule1 → Mule2 → Mule3 → Cash-Out Account

    Returns list of detected fraud chains with full path details.
    """
    print(f"[INFO] Tracing multi-hop fraud paths (max {max_hops} hops)...")

    # Identify source fraud accounts (sent fraud transactions)
    fraud_sources = df[df['isFraud'] == 1]['nameOrig'].unique()

    # Identify cash-out accounts (CASH_OUT type transactions)
    cashout_accounts = set(df[df['type'] == 'CASH_OUT']['nameDest'].unique())

    fraud_paths = []
    visited_paths = set()

    for source in fraud_sources:
        if source not in G:
            continue

        # BFS to find all reachable cash-out accounts within max_hops
        try:
            for target in cashout_accounts:
                if target not in G or target == source:
                    continue

                # Find shortest path
                try:
                    path = nx.shortest_path(G, source=source, target=target)
                    if 2 <= len(path) <= max_hops + 1:
                        path_key = tuple(path)
                        if path_key not in visited_paths:
                            visited_paths.add(path_key)

                            # Calculate total amount along path
                            path_amounts = []
                            path_times = []
                            for i in range(len(path) - 1):
                                edge_data = G.get_edge_data(
                                    path[i], path[i+1], {})
                                path_amounts.append(edge_data.get('amount', 0))
                                path_times.append(edge_data.get('step', 0))

                            time_span = max(path_times) - \
                                min(path_times) if path_times else 0

                            fraud_paths.append({
                                'path': path,
                                'hops': len(path) - 1,
                                'source': source,
                                'destination': target,
                                'total_amount': sum(path_amounts),
                                'max_amount': max(path_amounts) if path_amounts else 0,
                                'time_span_hours': time_span,
                                'velocity': sum(path_amounts) / (time_span + 1),
                                'path_str': ' → '.join(path)
                            })
                except nx.NetworkXNoPath:
                    continue
        except Exception:
            continue

        # Limit to avoid very long processing
        if len(fraud_paths) >= 100:
            break

    # Sort by amount (highest value fraud paths first)
    fraud_paths.sort(key=lambda x: x['total_amount'], reverse=True)

    print(f"[INFO] Fraud paths detected: {len(fraud_paths)}")
    if fraud_paths:
        print(
            f"[INFO] Avg hops: {np.mean([p['hops'] for p in fraud_paths]):.1f}")
        print(f"[INFO] Max amount path: ₹{fraud_paths[0]['total_amount']:,.0f} "
              f"via {fraud_paths[0]['hops']} hops")
        print(f"\n[TOP 3 FRAUD PATHS]")
        for i, p in enumerate(fraud_paths[:3]):
            print(f"  {i+1}. {p['path_str']}")
            print(f"     Amount: ₹{p['total_amount']:,.0f} | "
                  f"Hops: {p['hops']} | "
                  f"Time span: {p['time_span_hours']} hours")

    return fraud_paths


# ─────────────────────────────────────────────
#  4. Combined Velocity Risk Score
# ─────────────────────────────────────────────

def compute_combined_risk(velocity_df: pd.DataFrame,
                          fandf: pd.DataFrame,
                          metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine graph risk, velocity risk, and fan-in/fan-out signals
    into a single unified fraud risk score per account.
    """
    print("[INFO] Computing combined risk scores...")

    # Merge all signals
    combined = metrics_df[['account', 'graph_risk_score', 'is_fraud']].copy()

    vel = velocity_df[['account', 'velocity_risk_score', 'rapid_forward_count',
                       'avg_dwell_time', 'forward_ratio']].copy()
    fan = fandf[['account', 'fan_in', 'fan_out', 'fan_in_out_score',
                 'is_fan_in_out']].copy()

    combined = combined.merge(vel, on='account', how='left')
    combined = combined.merge(fan, on='account', how='left')
    combined = combined.fillna(0)

    # Normalize fan_in_out_score
    max_fan = combined['fan_in_out_score'].max()
    combined['fan_risk'] = combined['fan_in_out_score'] / (max_fan + 1e-9)

    # Final combined risk score
    combined['combined_risk_score'] = (
        0.40 * combined['graph_risk_score'] +
        0.35 * combined['velocity_risk_score'] +
        0.25 * combined['fan_risk']
    ).clip(0, 1)

    # Risk tier classification
    combined['risk_tier'] = pd.cut(
        combined['combined_risk_score'],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )

    print(f"\n[RISK TIER DISTRIBUTION]")
    print(combined['risk_tier'].value_counts().to_string())
    print(f"\n[TOP 10 HIGHEST RISK ACCOUNTS]")
    top10 = combined.sort_values(
        'combined_risk_score', ascending=False).head(10)
    print(top10[['account', 'combined_risk_score', 'risk_tier',
                 'is_fraud', 'rapid_forward_count']].to_string(index=False))

    return combined


# ─────────────────────────────────────────────
#  5. Visualization
# ─────────────────────────────────────────────

def visualize_velocity_analysis(velocity_df: pd.DataFrame,
                                fraud_paths: list,
                                save_path: str = "output/velocity_analysis.png"):
    """Plot velocity analysis charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Transaction Velocity & Multi-Hop Fraud Analysis",
                 fontsize=14, fontweight='bold')

    fraud = velocity_df[velocity_df['is_fraud'] == 1]
    normal = velocity_df[velocity_df['is_fraud'] == 0]

    # 1. Dwell time distribution
    ax = axes[0, 0]
    ax.hist(normal['avg_dwell_time'].clip(0, 100), bins=40,
            alpha=0.6, color='#3498db', label='Normal', density=True)
    ax.hist(fraud['avg_dwell_time'].clip(0, 100), bins=20,
            alpha=0.7, color='#e74c3c', label='Fraud', density=True)
    ax.set_xlabel('Avg Dwell Time (hours)')
    ax.set_ylabel('Density')
    ax.set_title('Fund Dwell Time Distribution')
    ax.legend()

    # 2. Velocity risk score
    ax2 = axes[0, 1]
    ax2.hist(normal['velocity_risk_score'], bins=40,
             alpha=0.6, color='#3498db', label='Normal', density=True)
    ax2.hist(fraud['velocity_risk_score'], bins=20,
             alpha=0.7, color='#e74c3c', label='Fraud', density=True)
    ax2.set_xlabel('Velocity Risk Score')
    ax2.set_title('Velocity Risk Score Distribution')
    ax2.legend()

    # 3. Forward ratio
    ax3 = axes[1, 0]
    ax3.scatter(normal['forward_ratio'], normal['send_velocity'],
                alpha=0.3, color='#3498db', s=10, label='Normal')
    ax3.scatter(fraud['forward_ratio'], fraud['send_velocity'],
                alpha=0.8, color='#e74c3c', s=40, label='Fraud', zorder=5)
    ax3.set_xlabel('Forward Ratio (sent/received)')
    ax3.set_ylabel('Send Velocity')
    ax3.set_title('Forward Ratio vs Send Velocity')
    ax3.legend()

    # 4. Hop distribution of fraud paths
    ax4 = axes[1, 1]
    if fraud_paths:
        hops = [p['hops'] for p in fraud_paths]
        ax4.hist(hops, bins=range(1, max(hops)+2), color='#e74c3c',
                 alpha=0.8, edgecolor='white')
        ax4.set_xlabel('Number of Hops')
        ax4.set_ylabel('Count')
        ax4.set_title('Fraud Path Hop Distribution')
        ax4.set_xticks(range(1, max(hops)+1))
    else:
        ax4.text(0.5, 0.5, 'No fraud paths detected',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Fraud Path Hop Distribution')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Velocity analysis chart saved to: {save_path}")


def visualize_fraud_path(G: nx.DiGraph, fraud_paths: list,
                         save_path: str = "output/fraud_paths.png"):
    """Visualize top fraud paths as a subgraph."""
    if not fraud_paths:
        print("[WARN] No fraud paths to visualize")
        return

    # Take top 5 paths by amount
    top_paths = fraud_paths[:5]

    # Build subgraph from path nodes
    path_nodes = set()
    for p in top_paths:
        path_nodes.update(p['path'])

    subgraph = G.subgraph(path_nodes).copy()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    pos = nx.spring_layout(subgraph, k=1.5, seed=42)

    # Color nodes by role
    node_colors = []
    node_sizes = []
    for node in subgraph.nodes():
        is_source = any(node == p['source'] for p in top_paths)
        is_dest = any(node == p['destination'] for p in top_paths)
        is_middle = not is_source and not is_dest

        if is_source:
            node_colors.append('#e74c3c')   # Red = fraud source
            node_sizes.append(600)
        elif is_dest:
            node_colors.append('#f39c12')   # Orange = cash-out
            node_sizes.append(500)
        else:
            node_colors.append('#9b59b6')   # Purple = mule intermediary
            node_sizes.append(350)

    nx.draw_networkx_nodes(subgraph, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, ax=ax,
                           edge_color='#e74c3c', width=1.5, alpha=0.6,
                           arrows=True, arrowsize=15,
                           connectionstyle='arc3,rad=0.15')
    nx.draw_networkx_labels(subgraph, pos, ax=ax,
                            font_color='white', font_size=6)

    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Fraud Source'),
        mpatches.Patch(color='#9b59b6', label='Mule Intermediary'),
        mpatches.Patch(color='#f39c12', label='Cash-Out Account'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              facecolor='#2c2c54', labelcolor='white')
    ax.set_title("Multi-Hop Fraud Path Network (Top 5 Paths by Amount)",
                 color='white', fontsize=12, pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Fraud path visualization saved to: {save_path}")


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("  MODULE 4: VELOCITY & MULTI-HOP FRAUD DETECTION")
    print("=" * 60)

    # Load processed data (reuse from Module 1 & 2)
    if os.path.exists("output/transactions_processed.csv"):
        print("[INFO] Loading saved processed transactions...")
        df = pd.read_csv("output/transactions_processed.csv")
        account_features = pd.read_csv("output/account_features.csv")
        metrics_df = pd.read_csv("output/graph_metrics.csv")
        print(
            f"[INFO] Loaded {len(df):,} transactions, {len(metrics_df):,} accounts")
    else:
        print("[ERROR] Run graph_construction.py first to generate output files.")
        exit(1)

    # Rebuild graph (fast — no betweenness recomputation)
    G = build_transaction_graph(df, account_features)

    print()

    # Step 1: Velocity analysis
    velocity_df = compute_velocity_features(df)

    print()

    # Step 2: Fan-in / Fan-out detection
    fandf = detect_fan_in_fan_out(G, threshold_in=3, threshold_out=2)

    print()

    # Step 3: Multi-hop path tracing
    fraud_paths = trace_fraud_paths(G, df, max_hops=5)

    print()

    # Step 4: Combined risk scoring
    combined_risk = compute_combined_risk(velocity_df, fandf, metrics_df)

    # Save outputs
    velocity_df.to_csv("output/velocity_features.csv", index=False)
    fandf.to_csv("output/fan_detection.csv", index=False)
    combined_risk.to_csv("output/combined_risk_scores.csv", index=False)

    # Save fraud paths as JSON
    paths_serializable = [{k: (list(v) if isinstance(v, list) else v)
                           for k, v in p.items()} for p in fraud_paths]
    with open("output/fraud_paths.json", "w") as f:
        json.dump(paths_serializable, f, indent=2)

    # Visualizations
    visualize_velocity_analysis(velocity_df, fraud_paths)
    visualize_fraud_path(G, fraud_paths)

    print()
    print("=" * 60)
    print("[✓] Velocity & multi-hop detection complete.")
    print("    Outputs saved to: output/")
    print("    Files generated:")
    print("      - output/velocity_features.csv")
    print("      - output/fan_detection.csv")
    print("      - output/combined_risk_scores.csv")
    print("      - output/fraud_paths.json")
    print("      - output/velocity_analysis.png")
    print("      - output/fraud_paths.png")
    print("    Next step: Run fraud_ring_detector.py")
    print("=" * 60)
