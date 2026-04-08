"""
Module 6: Explainable AI (XAI) — SHAP + Fraud Narratives
AI-Powered Mule Account and Financial Fraud Network Detection System

Explains fraud predictions from the TGN model using:
  1. SHAP TreeExplainer  — which features drive each fraud prediction
  2. Per-account explanations — structured breakdown per flagged account
  3. Fraud narratives — human-readable paragraph per flagged account
  4. Visualizations — SHAP summary plots + account-level charts
"""

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Run: pip install shap")


# ─────────────────────────────────────────────
#  Feature Names
#  Must match tgnn_model.py TEMPORAL_FEATURE_COLS
# ─────────────────────────────────────────────

TEMPORAL_FEATURE_COLS = [
    'sent_count', 'recv_count', 'sent_total', 'recv_total',
    'sent_avg', 'recv_avg', 'send_velocity', 'recv_velocity',
    'dwell_time', 'is_rapid_forwarder', 'forward_ratio',
    'fan_in_out', 'activity_span', 'balance_drains',
    'high_risk_sent', 'burst_send', 'burst_recv'
]

FEATURE_DISPLAY_NAMES = {
    'sent_count':         'Outgoing Transaction Count',
    'recv_count':         'Incoming Transaction Count',
    'sent_total':         'Total Amount Sent',
    'recv_total':         'Total Amount Received',
    'sent_avg':           'Avg Amount Sent',
    'recv_avg':           'Avg Amount Received',
    'send_velocity':      'Send Velocity (tx/hr)',
    'recv_velocity':      'Receive Velocity (tx/hr)',
    'dwell_time':         'Fund Dwell Time (hrs)',
    'is_rapid_forwarder': 'Rapid Forwarder Flag',
    'forward_ratio':      'Forward Ratio (sent/recv)',
    'fan_in_out':         'Fan-In/Fan-Out Score',
    'activity_span':      'Activity Time Span',
    'balance_drains':     'Balance Drain Count',
    'high_risk_sent':     'High-Risk Transactions Sent',
    'burst_send':         'Burst Send Count',
    'burst_recv':         'Burst Receive Count',
}


# ─────────────────────────────────────────────
#  1. Load Data
# ─────────────────────────────────────────────

def load_data():
    """Load all outputs from previous modules."""
    print("[INFO] Loading model outputs...")

    required = [
        "output/fraud_probabilities.npy",
        "output/tgn_fraud_scores.csv",
        "output/transactions_processed.csv"
    ]
    for f in required:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f} — run tgnn_model.py first")
            exit(1)

    fraud_probs = np.load("output/fraud_probabilities.npy")
    tgn_scores = pd.read_csv("output/tgn_fraud_scores.csv")
    df = pd.read_csv("output/transactions_processed.csv")

    print(f"[INFO] Loaded {len(tgn_scores):,} account scores")
    print(
        f"[INFO] Fraud prob range: [{fraud_probs.min():.3f}, {fraud_probs.max():.3f}]")

    return fraud_probs, tgn_scores, df


def build_feature_matrix(df: pd.DataFrame, tgn_scores: pd.DataFrame):
    """Rebuild temporal feature matrix matching tgnn_model.py."""
    print("[INFO] Building feature matrix for SHAP...")

    # Sent features
    sent = df.groupby('nameOrig').agg(
        sent_count=('amount', 'count'),
        sent_total=('amount', 'sum'),
        sent_avg=('amount', 'mean'),
        first_sent_step=('step', 'min'),
        last_sent_step=('step', 'max'),
        fraud_sent=('isFraud', 'sum'),
        balance_drains=('balance_drained', 'sum'),
        high_risk_sent=('is_high_risk_type', 'sum'),
        burst_send=('step', lambda x: (x.diff().dropna() <= 1).sum()),
    ).reset_index().rename(columns={'nameOrig': 'account'})
    sent['send_velocity'] = sent['sent_count'] / (
        sent['last_sent_step'] - sent['first_sent_step'] + 1)

    # Received features
    recv = df.groupby('nameDest').agg(
        recv_count=('amount', 'count'),
        recv_total=('amount', 'sum'),
        recv_avg=('amount', 'mean'),
        first_recv_step=('step', 'min'),
        last_recv_step=('step', 'max'),
        burst_recv=('step', lambda x: (x.diff().dropna() <= 1).sum()),
    ).reset_index().rename(columns={'nameDest': 'account'})
    recv['recv_velocity'] = recv['recv_count'] / (
        recv['last_recv_step'] - recv['first_recv_step'] + 1)

    features = pd.merge(sent, recv, on='account', how='outer').fillna(0)
    features['dwell_time'] = (
        features['first_sent_step'] - features['first_recv_step']).clip(lower=0)
    features['is_rapid_forwarder'] = (
        features['dwell_time'] <= 2).astype(float)
    features['forward_ratio'] = (
        features['sent_total'] / (features['recv_total'] + 1e-6)).clip(0, 1)
    features['fan_in_out'] = features['recv_count'] * features['sent_count']
    features['activity_span'] = np.maximum(
        features['last_sent_step'] - features['first_sent_step'],
        features['last_recv_step'] - features['first_recv_step']
    )
    features['is_fraud'] = (features['fraud_sent'] > 0).astype(float)

    # Merge with fraud probabilities
    score_lookup = tgn_scores.set_index('account')['fraud_prob'].to_dict()
    features['fraud_prob'] = features['account'].map(score_lookup).fillna(0)

    feat_cols = [c for c in TEMPORAL_FEATURE_COLS if c in features.columns]
    print(
        f"[INFO] Feature matrix: {len(features):,} accounts × {len(feat_cols)} features")
    return features, feat_cols


# ─────────────────────────────────────────────
#  2. SHAP Analysis
# ─────────────────────────────────────────────

def run_shap_analysis(features: pd.DataFrame, feat_cols: list):
    """
    Train surrogate GBM and compute SHAP values.
    SHAP explains which temporal features drive each fraud prediction.
    """
    print("\n[INFO] Running SHAP feature importance analysis...")

    X = features[feat_cols].fillna(0).values.astype(np.float32)
    y = features['is_fraud'].values.astype(int)
    fp = features['fraud_prob'].values

    # Balance for surrogate training
    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]
    n_sample = min(len(normal_idx), len(fraud_idx) * 30)
    sample_idx = np.concatenate([
        fraud_idx,
        np.random.choice(normal_idx, size=n_sample, replace=False)
    ])
    np.random.shuffle(sample_idx)

    X_s = X[sample_idx]
    y_s = y[sample_idx]

    scaler = StandardScaler()
    X_s_sc = scaler.fit_transform(X_s)
    X_sc = scaler.transform(X)

    split = int(0.8 * len(X_s))
    gbm = GradientBoostingClassifier(
        n_estimators=150, max_depth=4,
        learning_rate=0.1, random_state=42
    )
    gbm.fit(X_s_sc[:split], y_s[:split])

    print("[INFO] Surrogate GBM performance:")
    y_pred = gbm.predict(X_s_sc[split:])
    print(classification_report(y_s[split:], y_pred,
          target_names=['Normal', 'Fraud'], zero_division=0))

    if not SHAP_AVAILABLE:
        print("[WARN] SHAP not available — skipping SHAP analysis")
        return None, feat_cols, X_sc, gbm, scaler

    print("[INFO] Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X_sc[:1000])

    if isinstance(shap_values, list):
        shap_fraud = shap_values[1]
    else:
        shap_fraud = shap_values

    mean_abs = np.abs(shap_fraud).mean(axis=0)
    fi_df = pd.DataFrame({
        'feature':     feat_cols,
        'display':     [FEATURE_DISPLAY_NAMES.get(f, f) for f in feat_cols],
        'importance':  mean_abs
    }).sort_values('importance', ascending=False)

    print("\n[SHAP FEATURE IMPORTANCE — Top Fraud Drivers]")
    for _, row in fi_df.iterrows():
        bar = '█' * int(row['importance'] / fi_df['importance'].max() * 25)
        print(f"  {row['display']:<35} {bar} {row['importance']:.4f}")

    return shap_fraud, feat_cols, X_sc, gbm, scaler, fi_df


# ─────────────────────────────────────────────
#  3. Per-Account Explanations
# ─────────────────────────────────────────────

def explain_top_accounts(features: pd.DataFrame,
                         feat_cols: list,
                         shap_vals,
                         top_n: int = 20) -> list:
    """Generate structured explanation for top N flagged accounts."""
    print(f"\n[INFO] Explaining top {top_n} highest-risk accounts...")

    top = features.nlargest(top_n, 'fraud_prob')
    explanations = []

    for _, row in top.iterrows():
        acc_idx = features.index.get_loc(row.name)

        feat_dict = {c: float(row[c]) for c in feat_cols if c in row}

        # SHAP values for this account
        shap_dict = {}
        if shap_vals is not None and acc_idx < len(shap_vals):
            sv = shap_vals[acc_idx]
            shap_dict = {feat_cols[i]: float(sv[i])
                         for i in range(len(feat_cols))}
            top_drivers = sorted(shap_dict.items(),
                                 key=lambda x: abs(x[1]), reverse=True)[:5]
        else:
            top_drivers = sorted(feat_dict.items(),
                                 key=lambda x: abs(x[1]), reverse=True)[:5]

        prob = float(row['fraud_prob'])
        risk = ('CRITICAL' if prob > 0.8 else
                'HIGH' if prob > 0.6 else
                'MEDIUM' if prob > 0.4 else 'LOW')

        explanations.append({
            'account_id':  row['account'],
            'fraud_prob':  round(prob, 4),
            'risk_level':  risk,
            'true_label':  int(row['is_fraud']),
            'features':    {k: round(v, 4) for k, v in feat_dict.items()},
            'shap_values': {k: round(v, 4) for k, v in shap_dict.items()},
            'top_drivers': [(FEATURE_DISPLAY_NAMES.get(k, k), round(v, 4))
                            for k, v in top_drivers],
        })

    print(f"[INFO] Generated {len(explanations)} explanations")
    return explanations


# ─────────────────────────────────────────────
#  4. Fraud Narrative Generator
# ─────────────────────────────────────────────

def generate_narrative(exp: dict) -> str:
    """
    Generate a human-readable fraud investigation narrative.
    This is what AML officers read in the STR report.
    """
    acc = exp['account_id']
    prob = exp['fraud_prob']
    risk = exp['risk_level']
    feats = exp['features']
    drivers = exp['top_drivers']

    parts = []

    # Opening
    parts.append(
        f"Account {acc} has been flagged by the AI system with a fraud "
        f"probability of {prob*100:.1f}% (Risk Level: {risk})."
    )

    # Key behavioral observations
    obs = []

    dwell = feats.get('dwell_time', 0)
    if dwell <= 2:
        obs.append(
            f"funds forwarded within {dwell:.0f} hour(s) of receipt "
            f"(rapid forwarding — classic mule behavior)"
        )
    elif dwell <= 24:
        obs.append(f"short fund dwell time of {dwell:.0f} hours")

    drain = feats.get('balance_drains', 0)
    if drain >= 1:
        obs.append(
            f"complete balance drain detected on {int(drain)} occasion(s)")

    recv = feats.get('recv_count', 0)
    sent = feats.get('sent_count', 0)
    if recv >= 3 and sent >= 3:
        obs.append(
            f"fan-in/fan-out pattern: received from {int(recv)} sources, "
            f"sent to {int(sent)} destinations"
        )

    vel = feats.get('send_velocity', 0)
    if vel > 0.5:
        obs.append(f"elevated send velocity ({vel:.2f} transactions/hour)")

    fwd = feats.get('forward_ratio', 0)
    if fwd > 0.8:
        obs.append(
            f"forwarded {fwd*100:.0f}% of received funds "
            f"(consistent with layering)"
        )

    hr = feats.get('high_risk_sent', 0)
    if hr >= 2:
        obs.append(
            f"{int(hr)} high-risk transaction types (TRANSFER/CASH_OUT)")

    burst = feats.get('burst_send', 0)
    if burst >= 2:
        obs.append(f"{int(burst)} burst transactions within 1-hour windows")

    if obs:
        parts.append("Key risk indicators: " + "; ".join(obs) + ".")

    # Top SHAP drivers
    if drivers:
        driver_names = [d[0] for d in drivers[:3]]
        parts.append(
            f"Primary factors driving this alert: "
            f"{', '.join(driver_names)}."
        )

    # Pattern conclusion
    if prob > 0.7:
        parts.append(
            "The combination of rapid fund forwarding, balance drain events, "
            "and high-risk transaction patterns is strongly consistent with "
            "mule account behavior used to launder illicit funds."
        )
    elif prob > 0.4:
        parts.append(
            "This account exhibits suspicious transaction patterns that "
            "warrant further investigation by AML officers."
        )

    # Recommended action
    actions = {
        'CRITICAL': "IMMEDIATE ACCOUNT FREEZE — Escalate to senior AML officer and file STR within 24 hours.",
        'HIGH':     "FLAG for enhanced due diligence — Request transaction history and KYC re-verification.",
        'MEDIUM':   "MONITOR closely — Set transaction alerts and schedule account review.",
        'LOW':      "LOG for periodic review — No immediate action required."
    }
    parts.append(
        f"Recommended action: {actions.get(risk, 'Review required.')}")

    return " ".join(parts)


# ─────────────────────────────────────────────
#  5. Visualizations
# ─────────────────────────────────────────────

def visualize_shap_summary(fi_df: pd.DataFrame,
                           shap_vals,
                           feat_cols: list,
                           features: pd.DataFrame,
                           save_path: str = "output/shap_summary.png"):
    """SHAP feature importance visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("SHAP Feature Importance — TGN Fraud Detection",
                 fontsize=14, fontweight='bold')

    # Left: bar chart
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(fi_df)))
    bars = ax.barh(range(len(fi_df)),
                   fi_df['importance'].values,
                   color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(fi_df)))
    ax.set_yticklabels(fi_df['display'].values, fontsize=8)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance\n(Impact on Fraud Prediction)')
    ax.invert_yaxis()
    for bar, val in zip(bars, fi_df['importance'].values):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=7)

    # Right: scatter of top features
    ax2 = axes[1]
    if shap_vals is not None:
        top_feats = fi_df.head(6)['feature'].values
        for i, feat in enumerate(top_feats):
            if feat not in feat_cols:
                continue
            fi = feat_cols.index(feat)
            sv = shap_vals[:, fi]
            fv = features[feat].values[:len(shap_vals)]
            fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            jitter = np.random.normal(i, 0.08, size=len(sv))
            scatter = ax2.scatter(sv, jitter, c=fv_n, cmap='RdYlGn_r',
                                  alpha=0.4, s=8)

        ax2.set_yticks(range(len(top_feats)))
        ax2.set_yticklabels(
            [FEATURE_DISPLAY_NAMES.get(f, f) for f in top_feats], fontsize=8)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('SHAP Value')
        ax2.set_title('SHAP Distribution\n(Red = High feature value)')
        plt.colorbar(scatter, ax=ax2, label='Feature Value (normalized)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] SHAP summary saved to: {save_path}")


def visualize_account(exp: dict,
                      save_path: str = None):
    """Per-account explanation chart."""
    acc = exp['account_id']
    prob = exp['fraud_prob']

    if save_path is None:
        save_path = f"output/explanations/explanation_{acc[:12]}.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Fraud Explanation — Account {acc}\n"
        f"Fraud Probability: {prob*100:.1f}%  |  Risk: {exp['risk_level']}",
        fontsize=11, fontweight='bold'
    )

    # Left: feature values
    ax = axes[0]
    feats = {FEATURE_DISPLAY_NAMES.get(k, k): v
             for k, v in exp['features'].items()}
    names = list(feats.keys())[:10]
    values = [feats[n] for n in names]
    colors = ['#e74c3c' if v > np.mean(values) else '#3498db' for v in values]
    ax.barh(range(len(names)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Feature Value')
    ax.set_title('Account Feature Profile')
    ax.invert_yaxis()

    # Right: SHAP waterfall
    ax2 = axes[1]
    if exp['shap_values']:
        sv_items = sorted(
            [(FEATURE_DISPLAY_NAMES.get(k, k), v)
             for k, v in exp['shap_values'].items()],
            key=lambda x: abs(x[1]), reverse=True
        )[:8]
        sv_names = [k for k, _ in sv_items]
        sv_vals = [v for _, v in sv_items]
        colors2 = ['#e74c3c' if v > 0 else '#2ecc71' for v in sv_vals]
        ax2.barh(range(len(sv_names)), sv_vals, color=colors2, alpha=0.8)
        ax2.set_yticks(range(len(sv_names)))
        ax2.set_yticklabels(sv_names, fontsize=8)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('SHAP Value')
        ax2.set_title(
            'Feature Contributions\n(Red=increases risk, Green=decreases)')
        ax2.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/explanations", exist_ok=True)

    print("=" * 60)
    print("  MODULE 6: EXPLAINABLE AI (XAI)")
    print("=" * 60)

    # ── Step 1: Load data ──
    fraud_probs, tgn_scores, df = load_data()

    # ── Step 2: Build feature matrix ──
    features, feat_cols = build_feature_matrix(df, tgn_scores)

    # ── Step 3: SHAP analysis ──
    result = run_shap_analysis(features, feat_cols)

    if result is not None and len(result) == 6:
        shap_vals, feat_cols, X_sc, gbm, scaler, fi_df = result
    else:
        shap_vals = None
        fi_df = pd.DataFrame({'feature': feat_cols, 'display': feat_cols,
                              'importance': np.zeros(len(feat_cols))})

    # ── Step 4: Explain top accounts ──
    explanations = explain_top_accounts(
        features, feat_cols, shap_vals, top_n=20)

    # ── Step 5: Generate narratives ──
    print("\n[INFO] Generating fraud narratives...")
    narratives = []
    for exp in explanations:
        narrative = generate_narrative(exp)
        exp['narrative'] = narrative
        narratives.append({
            'account_id': exp['account_id'],
            'fraud_prob': exp['fraud_prob'],
            'risk_level': exp['risk_level'],
            'narrative':  narrative
        })

    print(f"\n[SAMPLE NARRATIVE — Top Risk Account]")
    print("-" * 60)
    if narratives:
        print(narratives[0]['narrative'])
    print("-" * 60)

    # ── Step 6: Visualize ──
    if shap_vals is not None and fi_df is not None:
        visualize_shap_summary(fi_df, shap_vals, feat_cols, features)
    else:
        print("[WARN] Skipping SHAP visualization (SHAP not available)")

    for exp in explanations[:3]:
        visualize_account(
            exp,
            save_path=f"output/explanations/explanation_{exp['account_id'][:12]}.png"
        )

    # ── Step 7: Save outputs ──
    with open("output/fraud_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2, default=str)

    with open("output/fraud_narratives.json", "w") as f:
        json.dump(narratives, f, indent=2)

    if fi_df is not None:
        fi_df.to_csv("output/shap_feature_importance.csv", index=False)

    print()
    print("=" * 60)
    print("[✓] XAI analysis complete.")
    print("    Files generated:")
    print("      - output/shap_summary.png")
    print("      - output/fraud_explanations.json")
    print("      - output/fraud_narratives.json")
    print("      - output/shap_feature_importance.csv")
    print("      - output/explanations/explanation_*.png")
    print("    Next step: Run str_generator.py")
    print("=" * 60)
