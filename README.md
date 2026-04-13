# AI-Powered Mule Account and Financial Fraud Network Detection System

> **B.E. Major Project | AI & Machine Learning | University College of Engineering, Osmania University**  
> Faisal Durrani · Abdul Rahman Bin Mahfooz · Munzer Ahmed  
> Supervisor: Dr. B. Sujatha

---

## What This Project Does

Traditional Anti-Money Laundering (AML) systems look at transactions one by one. Fraudsters exploit this by moving stolen money through coordinated networks of **mule accounts** — each account appears normal in isolation, but the network pattern reveals the fraud.

This system models financial transactions as a **temporal graph** and uses a combination of Temporal Graph Neural Networks, community detection, and explainable AI to detect mule accounts, trace fraud chains, and identify coordinated fraud rings — then generates professional PDF investigation reports automatically.

---

## Results

| Metric | Value |
|--------|-------|
| Test ROC-AUC | **0.9657** |
| Validation AUC | **0.9694** |
| Fraud Recall | **100%** |
| PR-AUC | **0.7627** |
| Fraud Rings Detected | **5,719** |
| Accounts Scored | **374,534** |
| STR Reports Generated | **20 PDFs** |

### Model Comparison

| Model | ROC-AUC | PR-AUC | Network-Aware | Explainable |
|-------|---------|--------|---------------|-------------|
| Rule-Based AML | 0.9372 | 0.2109 | No | No |
| Logistic Regression | 0.9726 | 0.5492 | No | No |
| Random Forest | 0.9551 | 0.6142 | No | No |
| XGBoost | 0.9871 | 0.7735 | No | No |
| GraphSAGE | 0.9263 | 0.0600 | Yes | No |
| **Our TGN System** | **0.9865** | **0.7627** | **Yes** | **Yes (SHAP)** |

> **Why not just use XGBoost?** XGBoost scores individual accounts. Our system additionally detects 5,719 coordinated fraud ring communities and traces multi-hop money laundering chains — capabilities that are impossible in a tabular ML model regardless of AUC.

---

## System Architecture

```
PaySim Transactions (6.3M)
         │
         ▼
┌─────────────────────┐
│  Data Ingestion     │  17 temporal features: dwell time, velocity,
│  Feature Engineering│  balance drain, fan-in/fan-out, burst activity
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Transaction Graph  │  374,534 nodes · 200,000 edges
│  Construction       │  PageRank · Betweenness Centrality
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Temporal Graph     │  TGN Memory Module (Rossi et al. 2020)
│  Neural Network     │  Time2Vec · TransformerConv · MLP Classifier
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│Velocity│ │  Fraud Ring  │  Louvain Community Detection
│ Model  │ │  Detector    │  5,719 FRAUD_RING communities
└────┬───┘ └──────┬───────┘
    │              │
    └──────┬───────┘
           │
           ▼
┌─────────────────────┐
│  Explainable AI     │  SHAP TreeExplainer · GNNExplainer
│  (XAI)              │  Auto-generated fraud narratives
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│  STR   │ │  AML Officer │
│Reports │ │  Dashboard   │
│(20 PDF)│ │  (Streamlit) │
└────────┘ └──────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| GNN Framework | PyTorch Geometric 2.7.0 |
| Deep Learning | PyTorch 2.5.1 + CUDA 12.1 |
| Graph Analytics | NetworkX |
| Community Detection | python-louvain |
| Explainability | SHAP · GNNExplainer |
| Dashboard | Streamlit + Plotly |
| Report Generation | fpdf2 |
| Baseline Models | Scikit-learn · XGBoost |
| Dataset | PaySim (Kaggle) |

---

## Project Structure

```
fraud_detection/
├── src/
│   ├── data_ingestion.py        # Module 1/2: Data loading + feature engineering
│   ├── graph_construction.py    # Module 2:   Transaction graph (374k nodes)
│   ├── tgnn_model.py            # Module 3:   TGN + temporal features
│   ├── velocity_model.py        # Module 4:   Dwell time + multi-hop paths
│   ├── fraud_ring_detector.py   # Module 5:   Louvain community detection
│   ├── xai_explainer.py         # Module 6:   SHAP + fraud narratives
│   ├── str_generator.py         # Module 7:   PDF STR generation
│   ├── dashboard.py             # Module 8:   Streamlit AML dashboard
│   ├── eval_metrics.py          # Module 9:   Ablation study + baselines
│   ├── gnn_explainer.py         # Module 10:  GNNExplainer subgraph viz
│   ├── fraud_case_studies.py    # Module 11:  3 fraud case studies
│   ├── run_pipeline.py          # Pipeline runner (all modules)
│   └── output/                  # Generated outputs
│       ├── fraud_probabilities.npy
│       ├── tgn_fraud_scores.csv
│       ├── community_scores.csv
│       ├── fraud_narratives.json
│       ├── reports/             # PDF STR reports
│       ├── eval/                # Confusion matrix, PR curve, ablation
│       ├── gnn_explanations/    # Subgraph visualisations
│       └── case_studies/        # Fraud case study charts
├── data/
│   └── raw/                     # PaySim CSV (download separately)
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/munzahmed07/fraud-detection-tgn.git
cd fraud-detection-tgn
```

### 2. Create conda environment
```bash
conda create -n fraud_detection python=3.10
conda activate fraud_detection
```

### 3. Install dependencies
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install torch-scatter torch-sparse --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install pandas numpy scikit-learn networkx python-louvain shap xgboost
pip install streamlit plotly fpdf2 matplotlib seaborn
```

### 4. Download the dataset
Download PaySim from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place the CSV at:
```
data/raw/PS_20174392719_1491204439457_log.csv
```

### 5. Run the pipeline
```bash
cd src

# Check status
python run_pipeline.py --status

# Run full pipeline
python run_pipeline.py

# Resume from a specific module
python run_pipeline.py --from 3

# Run on full 6.3M dataset (slower)
python run_pipeline.py --full
```

### 6. Launch the dashboard
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## Key Features

**Temporal Graph Neural Network**  
Implements TGN (Rossi et al. 2020) with per-node GRU memory module, learnable time2vec encoding, and temporal attention via TransformerConv. Builds behavioral memory for every account across all transactions chronologically.

**Fraud Ring Detection**  
Louvain community detection on the 374k node transaction graph identifies 5,719 coordinated fraud ring communities — exposing entire criminal networks, not just individual mule accounts.

**Explainable AI**  
SHAP TreeExplainer identifies balance drain count as the top fraud predictor (importance: 1.30). GNNExplainer generates subgraph visualisations showing which specific transaction edges drove each prediction. Auto-generated fraud narratives explain every flagged account in plain English.

**Automated STR Generation**  
20 professional PDF Suspicious Transaction Reports generated automatically for CRITICAL risk accounts. Each 8-section report includes risk gauge, SHAP explanation table, AI narrative, transaction history, and recommended AML action.

**AML Officer Dashboard**  
6-page Streamlit dashboard with fraud alerts, account investigation tools, SHAP waterfall charts, fraud ring analysis, velocity charts, and STR PDF downloads.

---

## Ablation Study

| Model Version | ROC-AUC | PR-AUC |
|---------------|---------|--------|
| Rule-Based AML | 0.9372 | 0.2109 |
| GraphSAGE (Static GNN) | 0.9263 | 0.0600 |
| TGN Memory Score Only | 0.9719 | 0.4888 |
| Temporal Features Only | 0.9864 | 0.7553 |
| **Full System (TGN + Features)** | **0.9865** | **0.7627** |

The ablation confirms: TGN memory (0.9719 → 0.9865 AUC) and temporal features both contribute. The full system achieves the best PR-AUC, which is the correct metric for imbalanced fraud detection.

---

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic simulation of real mobile money transactions calibrated against a real payment service.

| Property | Value |
|----------|-------|
| Total Transactions | 6,362,620 |
| Fraud Rate | 0.13% |
| Transaction Types | TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT |
| Time Span | ~30 days (743 hours) |
| Used in Training | 200,000 (chronological sample) |

---

## Reference

> Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020).  
> **Temporal Graph Networks for Deep Learning on Dynamic Graphs.**  
> arXiv preprint arXiv:2006.10637.

---

## License

This project is submitted as a Major Project for the B.E. degree in Artificial Intelligence & Machine Learning at University College of Engineering (Autonomous), Osmania University, Hyderabad — 2022-2026.
