"""
Module 1: Data Ingestion & Preprocessing
AI-Powered Mule Account and Financial Fraud Network Detection System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os


def load_paysim_data(filepath: str) -> pd.DataFrame:
    """
    Load PaySim dataset from CSV.
    Download from: https://www.kaggle.com/datasets/ealaxi/paysim1
    Expected columns: step, type, amount, nameOrig, oldbalanceOrg,
                      newbalanceOrig, nameDest, oldbalanceDest,
                      newbalanceDest, isFraud, isFlaggedFraud
    """
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df):,} transactions | Fraud rate: {df['isFraud'].mean():.2%}")
    return df


def generate_synthetic_paysim(n_transactions: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic PaySim-like dataset for local testing
    when the real dataset is not yet downloaded.
    Simulates mule account behavior: chains of TRANSFER -> CASH_OUT
    """
    np.random.seed(seed)
    n = n_transactions

    transaction_types = np.random.choice(
        ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],
        size=n,
        p=[0.35, 0.25, 0.25, 0.10, 0.05]
    )

    # Generate accounts — small pool creates realistic graph density
    n_accounts = int(n * 0.3)
    orig_accounts = [f"C{np.random.randint(1000000, 9999999)}" for _ in range(n_accounts)]
    dest_accounts = [f"C{np.random.randint(1000000, 9999999)}" for _ in range(n_accounts)]
    merchant_accounts = [f"M{np.random.randint(100000, 999999)}" for _ in range(50)]

    senders = np.random.choice(orig_accounts, size=n)
    receivers = np.where(
        np.isin(transaction_types, ['PAYMENT', 'DEBIT']),
        np.random.choice(merchant_accounts, size=n),
        np.random.choice(dest_accounts, size=n)
    )

    amounts = np.round(np.random.lognormal(mean=7, sigma=2, size=n), 2)
    steps = np.random.randint(1, 743, size=n)  # PaySim uses 1 hour steps over 30 days

    old_balance_orig = np.round(np.random.lognormal(mean=8, sigma=2, size=n), 2)
    new_balance_orig = np.maximum(0, old_balance_orig - amounts)
    old_balance_dest = np.round(np.random.lognormal(mean=7, sigma=2, size=n), 2)
    new_balance_dest = old_balance_dest + amounts

    # Inject mule account fraud patterns
    # Mule behavior: multiple small TRANSFER -> CASH_OUT chains
    is_fraud = np.zeros(n, dtype=int)
    n_fraud = int(n * 0.013)  # ~1.3% fraud rate (realistic for PaySim)

    # Select mule accounts
    n_mules = max(5, int(n_fraud * 0.3))
    mule_accounts = np.random.choice(dest_accounts, size=n_mules, replace=False)

    fraud_indices = np.random.choice(np.where(transaction_types == 'TRANSFER')[0],
                                     size=min(n_fraud, (transaction_types == 'TRANSFER').sum()),
                                     replace=False)
    is_fraud[fraud_indices] = 1
    receivers[fraud_indices] = np.random.choice(mule_accounts, size=len(fraud_indices))
    amounts[fraud_indices] = old_balance_orig[fraud_indices]  # Full balance drain = red flag
    new_balance_orig[fraud_indices] = 0

    df = pd.DataFrame({
        'step': steps,
        'type': transaction_types,
        'amount': amounts,
        'nameOrig': senders,
        'oldbalanceOrg': old_balance_orig,
        'newbalanceOrig': new_balance_orig,
        'nameDest': receivers,
        'oldbalanceDest': old_balance_dest,
        'newbalanceDest': new_balance_dest,
        'isFraud': is_fraud,
        'isFlaggedFraud': np.zeros(n, dtype=int)
    })

    print(f"[INFO] Generated {len(df):,} synthetic transactions | "
          f"Fraud rate: {df['isFraud'].mean():.2%} | "
          f"Mule accounts: {n_mules}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from raw PaySim data.
    Returns enriched DataFrame ready for graph construction.
    """
    print("[INFO] Preprocessing transactions...")
    df = df.copy()

    # --- Encode transaction type ---
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    # --- Temporal features (step = 1 hour in PaySim) ---
    df['hour_of_day'] = df['step'] % 24
    df['day_of_month'] = (df['step'] // 24) % 30 + 1
    df['is_odd_hour'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)

    # --- Balance-based fraud indicators ---
    df['balance_drained'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
    df['amount_to_balance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    ).clip(0, 1)
    df['dest_balance_unchanged'] = (
        (df['oldbalanceDest'] == df['newbalanceDest']) & (df['amount'] > 0)
    ).astype(int)

    # --- Normalize amount for ML ---
    scaler = MinMaxScaler()
    df['amount_normalized'] = scaler.fit_transform(df[['amount']])

    # --- Flag high-risk transaction types ---
    df['is_high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)

    print(f"[INFO] Feature engineering complete. Shape: {df.shape}")
    return df


def get_account_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-account behavioral features.
    These become node features in the transaction graph.
    """
    print("[INFO] Computing account-level features...")

    sent = df.groupby('nameOrig').agg(
        sent_count=('amount', 'count'),
        sent_total=('amount', 'sum'),
        sent_avg=('amount', 'mean'),
        sent_max=('amount', 'max'),
        sent_fraud_count=('isFraud', 'sum'),
        balance_drained_count=('balance_drained', 'sum'),
        high_risk_sent=('is_high_risk_type', 'sum'),
    ).reset_index().rename(columns={'nameOrig': 'account'})

    received = df.groupby('nameDest').agg(
        recv_count=('amount', 'count'),
        recv_total=('amount', 'sum'),
        recv_avg=('amount', 'mean'),
        recv_max=('amount', 'max'),
    ).reset_index().rename(columns={'nameDest': 'account'})

    # Merge sent and received features
    account_features = pd.merge(sent, received, on='account', how='outer').fillna(0)

    # Derived graph-level behavioral features
    account_features['fan_in_fan_out_ratio'] = np.where(
        account_features['recv_count'] > 0,
        account_features['sent_count'] / account_features['recv_count'],
        account_features['sent_count']
    )
    account_features['is_mule_candidate'] = (
        (account_features['recv_count'] > 2) &
        (account_features['sent_count'] > 2) &
        (account_features['balance_drained_count'] > 0)
    ).astype(int)

    # Ground truth: account is fraudulent if it sent any fraud transaction
    account_features['is_fraud'] = (account_features['sent_fraud_count'] > 0).astype(int)

    print(f"[INFO] Account features computed for {len(account_features):,} unique accounts.")
    print(f"[INFO] Mule candidates: {account_features['is_mule_candidate'].sum():,}")
    return account_features


if __name__ == "__main__":
    # --- Run with synthetic data (no download needed) ---
    df_raw = generate_synthetic_paysim(n_transactions=5000)
    df = preprocess(df_raw)
    account_features = get_account_level_features(df)
    print("\n[SAMPLE] Account features (top 5):")
    print(account_features.head())

    # Save for use in next module
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/transactions_processed.csv", index=False)
    account_features.to_csv("output/account_features.csv", index=False)
    print("\n[INFO] Saved processed data to output/")
