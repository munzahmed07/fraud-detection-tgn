"""
Module 8: AML Officer Investigation Dashboard
AI-Powered Mule Account and Financial Fraud Network Detection System

Streamlit web dashboard for AML officers to:
  - Monitor fraud alerts in real time
  - Investigate flagged accounts
  - View SHAP explanations
  - Download STR reports
  - Explore fraud ring communities
  - Analyze transaction velocity patterns

Run with: streamlit run dashboard.py
"""

import os, sys, json, warnings
import pandas as pd
pd.set_option('styler.render.max_elements', 1000000)
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; border-radius: 8px; padding: 10px; }
    .risk-critical { color: #ff4b4b; font-weight: bold; }
    .risk-high     { color: #ff8c00; font-weight: bold; }
    .risk-medium   { color: #ffd700; font-weight: bold; }
    .risk-low      { color: #00cc66; font-weight: bold; }
    div[data-testid="stSidebarContent"] { background-color: #1e2130; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Data Loaders (cached)
# ─────────────────────────────────────────────

@st.cache_data
def load_all_data():
    """Load all pipeline outputs."""
    data = {}

    # TGN fraud scores
    if os.path.exists("output/tgn_fraud_scores.csv"):
        data['scores'] = pd.read_csv("output/tgn_fraud_scores.csv")
    else:
        st.error("Run tgnn_model.py first")
        st.stop()

    # Transactions
    if os.path.exists("output/transactions_processed.csv"):
        data['transactions'] = pd.read_csv("output/transactions_processed.csv")

    # Graph metrics
    if os.path.exists("output/graph_metrics.csv"):
        data['graph_metrics'] = pd.read_csv("output/graph_metrics.csv")

    # Community scores
    if os.path.exists("output/community_scores.csv"):
        data['communities'] = pd.read_csv("output/community_scores.csv")

    # Velocity features
    if os.path.exists("output/velocity_features.csv"):
        data['velocity'] = pd.read_csv("output/velocity_features.csv")

    # Explanations
    if os.path.exists("output/fraud_explanations.json"):
        with open("output/fraud_explanations.json") as f:
            data['explanations'] = json.load(f)

    # SHAP importance
    if os.path.exists("output/shap_feature_importance.csv"):
        data['shap_fi'] = pd.read_csv("output/shap_feature_importance.csv")

    return data


@st.cache_data
def get_fraud_alerts(scores: pd.DataFrame, threshold: float = 0.5):
    """Get flagged accounts above threshold."""
    alerts = scores[scores['fraud_prob'] >= threshold].copy()
    alerts['risk_level'] = pd.cut(
        alerts['fraud_prob'],
        bins=[0.5, 0.6, 0.8, 1.01],
        labels=['MEDIUM', 'HIGH', 'CRITICAL']
    )
    return alerts.sort_values('fraud_prob', ascending=False)


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

def render_sidebar(data):
    st.sidebar.image("https://img.icons8.com/color/96/fraud.png", width=60)
    st.sidebar.title("🔍 Fraud Detection")
    st.sidebar.caption("AI-Powered AML Investigation System")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview",
         "🚨 Fraud Alerts",
         "🔬 Account Investigation",
         "🕸️ Fraud Ring Analysis",
         "⚡ Velocity Analysis",
         "📈 Model Performance"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()
    threshold = st.sidebar.slider(
        "Risk Score Threshold", 0.0, 1.0, 0.5, 0.05,
        help="Accounts above this score are flagged"
    )

    scores   = data['scores']
    n_total  = len(scores)
    n_flagged = len(scores[scores['fraud_prob'] >= threshold])
    n_fraud  = int(scores['is_fraud'].sum())

    st.sidebar.metric("Total Accounts",  f"{n_total:,}")
    st.sidebar.metric("Flagged Accounts", f"{n_flagged:,}")
    st.sidebar.metric("Confirmed Fraud",  f"{n_fraud:,}")

    return page, threshold


# ─────────────────────────────────────────────
#  Page 1: Overview
# ─────────────────────────────────────────────

def page_overview(data, threshold):
    st.title("📊 System Overview")
    st.caption("AI-Powered Mule Account and Financial Fraud Network Detection System")
    st.divider()

    scores = data['scores']
    txns   = data.get('transactions', pd.DataFrame())

    # ── KPI Cards ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏦 Total Accounts",    f"{len(scores):,}")
    col2.metric("🚨 Fraud Probability > 0.5",
                f"{len(scores[scores['fraud_prob']>=0.5]):,}")
    col3.metric("✅ Confirmed Fraud",   f"{int(scores['is_fraud'].sum()):,}")
    col4.metric("📋 Transactions",      f"{len(txns):,}" if len(txns) else "N/A")
    col5.metric("🎯 Model AUC",         "0.9657")

    st.divider()

    col1, col2 = st.columns(2)

    # Fraud score distribution
    with col1:
        st.subheader("Fraud Score Distribution")
        fig = px.histogram(
            scores, x='fraud_prob', nbins=50,
            color_discrete_sequence=['#2E75B6'],
            labels={'fraud_prob': 'Fraud Probability', 'count': 'Accounts'},
            title="Distribution of Fraud Probabilities"
        )
        fig.add_vline(x=threshold, line_dash="dash",
                      line_color="red", annotation_text="Threshold")
        fig.update_layout(
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk tier breakdown
    with col2:
        st.subheader("Risk Tier Breakdown")
        alerts = get_fraud_alerts(scores, threshold)
        if len(alerts) > 0:
            tier_counts = alerts['risk_level'].value_counts().reset_index()
            tier_counts.columns = ['Risk Level', 'Count']
            colors = {'CRITICAL': '#ff4b4b', 'HIGH': '#ff8c00', 'MEDIUM': '#ffd700'}
            fig2 = px.bar(
                tier_counts, x='Risk Level', y='Count',
                color='Risk Level',
                color_discrete_map=colors,
                title="Flagged Accounts by Risk Tier"
            )
            fig2.update_layout(
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font_color='white', height=350, showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No accounts flagged at current threshold")

    # Transaction type breakdown
    if len(txns) > 0:
        st.subheader("Transaction Type Analysis")
        col3, col4 = st.columns(2)

        with col3:
            type_counts = txns['type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig3 = px.pie(type_counts, values='Count', names='Type',
                          title="Transaction Types",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig3.update_layout(
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font_color='white', height=300
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fraud_by_type = txns.groupby('type')['isFraud'].sum().reset_index()
            fraud_by_type.columns = ['Type', 'Fraud Count']
            fraud_by_type = fraud_by_type[fraud_by_type['Fraud Count'] > 0]
            fig4 = px.bar(fraud_by_type, x='Type', y='Fraud Count',
                          title="Fraud Transactions by Type",
                          color_discrete_sequence=['#ff4b4b'])
            fig4.update_layout(
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font_color='white', height=300
            )
            st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
#  Page 2: Fraud Alerts
# ─────────────────────────────────────────────

def page_fraud_alerts(data, threshold):
    st.title("🚨 Fraud Alerts")
    st.caption("Accounts flagged by the TGN model above the risk threshold")
    st.divider()

    scores = data['scores']
    alerts = get_fraud_alerts(scores, threshold)

    if len(alerts) == 0:
        st.warning("No accounts flagged at current threshold. Lower the threshold in the sidebar.")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts",   len(alerts))
    col2.metric("CRITICAL",       len(alerts[alerts['risk_level']=='CRITICAL']))
    col3.metric("HIGH + MEDIUM",  len(alerts[alerts['risk_level'].isin(['HIGH','MEDIUM'])]))

    st.divider()

    # Alert table
    st.subheader("Alert Table")
    display = alerts.head(200)[['account', 'fraud_prob', 'is_fraud', 'risk_level']].copy()
    display.columns = ['Account ID', 'Fraud Probability', 'Confirmed Fraud', 'Risk Level']
    display['Fraud Probability'] = display['Fraud Probability'].apply(lambda x: f"{x*100:.1f}%")
    display['Confirmed Fraud']   = display['Confirmed Fraud'].apply(lambda x: 'YES' if x else 'Suspected')
    st.dataframe(display, use_container_width=True, height=400)

    # Download buttons
    st.subheader("Download STR Reports")
    report_dir = "output/reports"
    if os.path.exists(report_dir):
        pdfs = [f for f in os.listdir(report_dir) if f.endswith('.pdf')]
        if pdfs:
            cols = st.columns(4)
            for i, pdf in enumerate(pdfs[:8]):
                with cols[i % 4]:
                    with open(os.path.join(report_dir, pdf), 'rb') as f:
                        st.download_button(
                            label=f"📄 {pdf[:20]}...",
                            data=f.read(),
                            file_name=pdf,
                            mime='application/pdf',
                            key=f"dl_{i}"
                        )
        else:
            st.info("Run str_generator.py to generate PDF reports")
    else:
        st.info("Run str_generator.py to generate PDF reports")


# ─────────────────────────────────────────────
#  Page 3: Account Investigation
# ─────────────────────────────────────────────

def page_account_investigation(data, threshold):
    st.title("🔬 Account Investigation")
    st.caption("Deep-dive investigation into a specific flagged account")
    st.divider()

    scores  = data['scores']
    txns    = data.get('transactions', pd.DataFrame())
    expls   = data.get('explanations', [])
    shap_fi = data.get('shap_fi', pd.DataFrame())

    # Account selector
    flagged = scores[scores['fraud_prob'] >= threshold]['account'].tolist()
    if not flagged:
        st.warning("No flagged accounts at current threshold")
        return

    selected = st.selectbox("Select Account to Investigate", flagged)

    # Get account data
    acc_row  = scores[scores['account'] == selected].iloc[0]
    acc_expl = next((e for e in expls if e['account_id'] == selected), None)

    # Risk badge
    prob = acc_row['fraud_prob']
    risk = ('CRITICAL' if prob > 0.8 else 'HIGH' if prob > 0.6 else 'MEDIUM')
    risk_color = {'CRITICAL': '#ff4b4b', 'HIGH': '#ff8c00', 'MEDIUM': '#ffd700'}

    st.markdown(
        f"<h3 style='color:{risk_color[risk]}'>"
        f"Account {selected} — {risk} RISK ({prob*100:.1f}%)"
        f"</h3>", unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk Score")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': risk_color[risk]},
                'steps': [
                    {'range': [0,  40],  'color': '#1a3d1a'},
                    {'range': [40, 60],  'color': '#3d3d00'},
                    {'range': [60, 80],  'color': '#3d2000'},
                    {'range': [80, 100], 'color': '#3d0000'},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8, 'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=250, paper_bgcolor='#1e2130', font_color='white'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if acc_expl:
            st.subheader("AI Narrative")
            st.info(acc_expl.get('narrative', 'No narrative available'))

    with col2:
        if acc_expl and acc_expl.get('features'):
            st.subheader("Feature Profile")
            feats = acc_expl['features']
            feat_df = pd.DataFrame({
                'Feature': list(feats.keys()),
                'Value':   list(feats.values())
            }).sort_values('Value', ascending=False).head(10)

            fig_feats = px.bar(
                feat_df, x='Value', y='Feature',
                orientation='h',
                color='Value',
                color_continuous_scale='RdYlGn_r',
                title="Top Feature Values"
            )
            fig_feats.update_layout(
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font_color='white', height=320, showlegend=False
            )
            st.plotly_chart(fig_feats, use_container_width=True)

    # SHAP explanation
    if acc_expl and acc_expl.get('shap_values'):
        st.subheader("SHAP Explanation — Why was this account flagged?")
        shap_items = sorted(
            acc_expl['shap_values'].items(),
            key=lambda x: abs(x[1]), reverse=True
        )[:10]
        shap_df = pd.DataFrame(shap_items, columns=['Feature', 'SHAP Value'])
        shap_df['Direction'] = shap_df['SHAP Value'].apply(
            lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
        shap_df['Color'] = shap_df['SHAP Value'].apply(
            lambda x: '#ff4b4b' if x > 0 else '#00cc66')

        fig_shap = go.Figure(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=shap_df['Color'],
            text=shap_df['Direction'],
            textposition='outside'
        ))
        fig_shap.add_vline(x=0, line_color='white', line_dash='dash')
        fig_shap.update_layout(
            title="SHAP Feature Contributions",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=350,
            xaxis_title="SHAP Value (positive = increases fraud risk)"
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # Transaction history
    if len(txns) > 0:
        st.subheader("Transaction History")
        acc_txns = txns[
            (txns['nameOrig'] == selected) |
            (txns['nameDest'] == selected)
        ].sort_values('step', ascending=False).head(20)

        if len(acc_txns) > 0:
            display_cols = ['step', 'type', 'nameOrig', 'nameDest',
                            'amount', 'isFraud', 'balance_drained']
            available = [c for c in display_cols if c in acc_txns.columns]
            st.dataframe(
                acc_txns[available].style.apply(
                    lambda x: ['background-color: #3d0000' if v else ''
                               for v in x == 1], subset=['isFraud']
                ) if 'isFraud' in available else acc_txns[available],
                use_container_width=True, height=300
            )
        else:
            st.info("No transactions found for this account")


# ─────────────────────────────────────────────
#  Page 4: Fraud Ring Analysis
# ─────────────────────────────────────────────

def page_fraud_rings(data):
    st.title("🕸️ Fraud Ring Analysis")
    st.caption("Community detection results — coordinated fraud network clusters")
    st.divider()

    comms = data.get('communities', pd.DataFrame())
    if len(comms) == 0:
        st.warning("Run fraud_ring_detector.py first")
        return

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Communities",  f"{len(comms):,}")
    col2.metric("FRAUD_RING",
                f"{len(comms[comms['ring_label']=='FRAUD_RING']):,}")
    col3.metric("HIGH_RISK",
                f"{len(comms[comms['ring_label']=='HIGH_RISK']):,}")
    col4.metric("SUSPICIOUS",
                f"{len(comms[comms['ring_label']=='SUSPICIOUS']):,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ring Classification Distribution")
        label_counts = comms['ring_label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        colors = {'FRAUD_RING': '#ff4b4b', 'HIGH_RISK': '#ff8c00',
                  'SUSPICIOUS': '#ffd700', 'NORMAL': '#00cc66'}
        fig = px.bar(
            label_counts, x='Label', y='Count',
            color='Label', color_discrete_map=colors,
            title="Communities by Risk Classification"
        )
        fig.update_layout(
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=350, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Ring Score vs Community Size")
        fig2 = px.scatter(
            comms.head(500), x='n_nodes', y='ring_score',
            color='ring_label', color_discrete_map=colors,
            title="Community Size vs Risk Score",
            labels={'n_nodes': 'Community Size', 'ring_score': 'Ring Score'}
        )
        fig2.update_layout(
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Top fraud rings table
    st.subheader("Top Fraud Ring Communities")
    top_rings = comms[comms['ring_label'] == 'FRAUD_RING'].head(20)
    if len(top_rings) > 0:
        display = top_rings[[
            'community_id', 'ring_score', 'n_nodes', 'n_edges',
            'n_fraud_nodes', 'fraud_density', 'total_amount', 'ring_label'
        ]].copy()
        display['total_amount'] = display['total_amount'].apply(
            lambda x: f"Rs.{x:,.0f}")
        display['fraud_density'] = display['fraud_density'].apply(
            lambda x: f"{x*100:.1f}%")
        display['ring_score'] = display['ring_score'].apply(
            lambda x: f"{x:.3f}")
        st.dataframe(display, use_container_width=True, height=400)


# ─────────────────────────────────────────────
#  Page 5: Velocity Analysis
# ─────────────────────────────────────────────

def page_velocity(data):
    st.title("⚡ Transaction Velocity Analysis")
    st.caption("Temporal mule behavior patterns — fund dwell time and forwarding velocity")
    st.divider()

    vel = data.get('velocity', pd.DataFrame())
    if len(vel) == 0:
        st.warning("Run velocity_model.py first")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fund Dwell Time Distribution")
        fraud  = vel[vel['is_fraud'] == 1]['avg_dwell_time'].clip(0, 500)
        normal = vel[vel['is_fraud'] == 0]['avg_dwell_time'].clip(0, 500)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=normal, name='Normal', opacity=0.6,
            marker_color='#2E75B6', nbinsx=40,
            histnorm='probability density'
        ))
        fig.add_trace(go.Histogram(
            x=fraud, name='Fraud', opacity=0.8,
            marker_color='#ff4b4b', nbinsx=20,
            histnorm='probability density'
        ))
        fig.update_layout(
            barmode='overlay',
            title="Dwell Time: Fraud vs Normal",
            xaxis_title="Avg Dwell Time (hours)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Velocity Risk Score Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=vel[vel['is_fraud']==0]['velocity_risk_score'],
            name='Normal', opacity=0.6, marker_color='#2E75B6',
            nbinsx=40, histnorm='probability density'
        ))
        fig2.add_trace(go.Histogram(
            x=vel[vel['is_fraud']==1]['velocity_risk_score'],
            name='Fraud', opacity=0.8, marker_color='#ff4b4b',
            nbinsx=20, histnorm='probability density'
        ))
        fig2.update_layout(
            barmode='overlay',
            title="Velocity Risk Score: Fraud vs Normal",
            xaxis_title="Velocity Risk Score",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Key stats
    st.subheader("Temporal Statistics")
    col3, col4, col5 = st.columns(3)
    col3.metric("Avg Dwell (Fraud)",
                f"{vel[vel['is_fraud']==1]['avg_dwell_time'].mean():.1f} hrs")
    col4.metric("Avg Dwell (Normal)",
                f"{vel[vel['is_fraud']==0]['avg_dwell_time'].mean():.1f} hrs")
    col5.metric("Rapid Forwarders",
                f"{vel['rapid_forward_count'].sum():,.0f}")


# ─────────────────────────────────────────────
#  Page 6: Model Performance
# ─────────────────────────────────────────────

def page_model_performance(data):
    st.title("📈 Model Performance")
    st.caption("TGN model evaluation metrics and SHAP feature importance")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC",      "0.9657", delta="+0.04 vs GraphSAGE")
    col2.metric("Val AUC",      "0.9694")
    col3.metric("Fraud Recall", "1.00",   delta="Catches all fraud")
    col4.metric("Avg Precision","0.3419")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SHAP Feature Importance")
        shap_fi = data.get('shap_fi', pd.DataFrame())
        if len(shap_fi) > 0:
            shap_fi = shap_fi.sort_values('importance', ascending=True).tail(12)
            fig = px.bar(
                shap_fi, x='importance', y='display',
                orientation='h',
                color='importance',
                color_continuous_scale='RdYlGn_r',
                title="Mean |SHAP Value| per Feature"
            )
            fig.update_layout(
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font_color='white', height=400,
                showlegend=False,
                xaxis_title="Mean |SHAP Value|",
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Architecture")
        st.markdown("""
        **TGN + Temporal Feature Augmentation**

        | Component | Details |
        |-----------|---------|
        | Memory Module | Per-node GRU (32-dim) |
        | Time Encoding | Learnable time2vec |
        | Attention | TransformerConv (2 heads) |
        | Temporal Features | 17 engineered signals |
        | Classifier | MLP (64→32→1) |
        | Parameters | 19,841 |
        | Training | 47 epochs, early stopping |
        | Hardware | CPU (stable for 374k nodes) |

        **Dataset**

        | Property | Value |
        |----------|-------|
        | Transactions | 200,000 |
        | Accounts | 374,534 |
        | Fraud Rate | 0.072% |
        | Train/Val/Test | 70/15/15 |

        **Reference:** Rossi et al. 2020 —
        *Temporal Graph Networks for Deep Learning on Dynamic Graphs*
        """)

    # Comparison table
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        'Model': ['Rule-Based AML', 'Random Forest', 'GraphSAGE (Static GNN)',
                  'TGN (Our System)'],
        'ROC-AUC':   [0.65, 0.78, 0.9263, 0.9657],
        'Recall':    [0.50, 0.65, 1.00,   1.00],
        'Temporal':  ['No', 'No', 'No', 'Yes'],
        'Network':   ['No', 'No', 'Yes', 'Yes'],
        'Explainable': ['No', 'Partial', 'No', 'Yes'],
    })
    st.dataframe(
        comparison.style.highlight_max(
            subset=['ROC-AUC', 'Recall'], color='#1a3d1a'
        ),
        use_container_width=True
    )


# ─────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────

def main():
    data = load_all_data()
    page, threshold = render_sidebar(data)

    if page == "📊 Overview":
        page_overview(data, threshold)
    elif page == "🚨 Fraud Alerts":
        page_fraud_alerts(data, threshold)
    elif page == "🔬 Account Investigation":
        page_account_investigation(data, threshold)
    elif page == "🕸️ Fraud Ring Analysis":
        page_fraud_rings(data)
    elif page == "⚡ Velocity Analysis":
        page_velocity(data)
    elif page == "📈 Model Performance":
        page_model_performance(data)


if __name__ == "__main__":
    main()
