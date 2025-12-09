"""
Data loading and preprocessing module.
Supports paysim.csv and momtsim.csv; falls back to synthetic data.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

def load_and_preprocess_data(paysim_path: str = 'data/paysim.csv', momtsim_path: str = 'data/momtsim.csv') -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    needed_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    if not os.path.exists(paysim_path):
        print(f"Note: '{paysim_path}' not found — creating small synthetic demo dataset.")
        n = 2000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'step': rng.integers(1, 100, size=n),
            'type': rng.choice(['CASH_OUT','PAYMENT','CASH_IN','TRANSFER','DEBIT'], size=n),
            'amount': rng.lognormal(mean=6, sigma=1, size=n),
            'nameOrig': [f'O{i}' for i in range(n)],
            'oldbalanceOrg': rng.uniform(0, 1e5, size=n),
            'newbalanceOrig': np.nan,
            'nameDest': [f'D{i}' for i in range(n)],
            'oldbalanceDest': rng.uniform(0, 1e5, size=n),
            'newbalanceDest': np.nan,
            'isFraud': 0
        })
        # Ensure numeric dtypes for all balance/amount columns before arithmetic
        for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float64)
            else:
                df[col] = np.zeros(len(df), dtype=np.float64)
        
        fraud_idx = rng.choice(n, size=int(n*0.05), replace=False).tolist()
        df.loc[fraud_idx, 'isFraud'] = 1
        # Use .iloc with integer indexing to ensure Series type for multiplication
        fraud_idx_int = [int(i) for i in fraud_idx]
        df.loc[fraud_idx_int, 'amount'] = df.iloc[fraud_idx_int]['amount'].values * 10.0  # type: ignore
        df['newbalanceOrig'] = df['oldbalanceOrg'] - df['amount']
        df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
        df['type'] = df['type'].map({'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}).fillna(0)
    else:
        df = pd.read_csv(paysim_path)
        if len(df) > 100000:
            df = df.sample(frac=0.1, random_state=42)
        df['type'] = df['type'].map({'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}).fillna(0)

    X = df[features].values.astype(np.float32)
    y = df['isFraud'].values.astype(np.float32)
    df = df[needed_cols].copy()

    if momtsim_path and os.path.exists(momtsim_path):
        df_m = pd.read_csv(momtsim_path)
        for col in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            if col not in df_m.columns:
                if 'Org' in col:
                    df_m[col] = df_m.get('oldbalanceOrg', 0) if 'new' in col else 0
                else:
                    df_m[col] = 0
        df_m['type'] = df_m.get('transaction_type', pd.Series([1]*len(df_m))).map(
            {'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}
        ).fillna(0)
        if 'is_fraud' in df_m.columns:
            df_m = df_m.rename(columns={'is_fraud': 'isFraud'})
        X_m = df_m[features].values.astype(np.float32)
        y_m = df_m['isFraud'].values.astype(np.float32)
        X = np.vstack([X, X_m])
        y = np.hstack([y, y_m])
        df_m_needed = df_m[needed_cols].copy()
        df = pd.concat([df, df_m_needed], ignore_index=True)

    # Normalise
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    return df.reset_index(drop=True), X, y