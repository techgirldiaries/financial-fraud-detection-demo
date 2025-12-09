"""
Model training module: PyTorch NN or Sklearn LogisticRegression fallback.
"""

import numpy as np
import pandas as pd
from typing import Any, List, Tuple, TYPE_CHECKING

# Optional imports with proper typing
USE_TORCH = False
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
else:
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    accuracy_score = None  # type: ignore
    precision_score = None  # type: ignore
    recall_score = None  # type: ignore
    f1_score = None  # type: ignore
    
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    USE_TORCH = True
except (ImportError, OSError) as e:
    # Catch both ImportError (missing package) and OSError (DLL initialization failures on Windows)
    print(f"Note: PyTorch not available ({type(e).__name__}), using sklearn fallback.")
    pass

HAS_SKLEARN = False
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore
    HAS_SKLEARN = True
except (ImportError, OSError) as e:
    print(f"Warning: sklearn metrics not available ({type(e).__name__}).")
    pass

if USE_TORCH:
    class TorchFraudClassifier(nn.Module):  # type: ignore
        def __init__(self, input_size: int):
            super().__init__()
            self.fclayer1 = nn.Linear(input_size, 64)  # type: ignore
            self.fclayer2 = nn.Linear(64, 32)  # type: ignore
            self.fclayer3 = nn.Linear(32, 1)  # type: ignore
            self.sigmoid = nn.Sigmoid()  # type: ignore

        def forward(self, x):
            x = torch.relu(self.fclayer1(x))  # type: ignore
            x = torch.relu(self.fclayer2(x))  # type: ignore
            x = self.sigmoid(self.fclayer3(x))  # type: ignore
            return x
else:
    class TorchFraudClassifier(object):  # type: ignore
        def __init__(self, input_size: int):
            # Minimal fallback class used when torch is not available.
            self.input_size = input_size

        def forward(self, x):
            raise RuntimeError("Torch is not available in this environment; TorchFraudClassifier cannot run.")


def train_classifier(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 1024) -> Tuple[Any, np.ndarray, List[str]]:
    fraud_mask = (y == 1)
    if np.sum(fraud_mask) < 10:
        print('Warning: Few fraud examples; using top amounts as synthetic fraud.')
        orig_amounts = np.asarray(df['amount'].values, dtype=np.float64)  # Ensure numpy array for argsort
        top_idx = np.argsort(orig_amounts)[-50:]
        fraud_mask = np.zeros_like(y, dtype=bool)
        fraud_mask[top_idx] = True

    # NumPy train/test split
    random_number_gen = np.random.default_rng(42)
    indices = np.arange(len(X))
    random_number_gen.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if USE_TORCH:
        X_train_t = torch.from_numpy(X_train).float()  # type: ignore
        y_train_t = torch.from_numpy(y_train).float()  # type: ignore
        train_dataset = TensorDataset(X_train_t, y_train_t)  # type: ignore
        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(X_train)), shuffle=True)  # type: ignore

        model = TorchFraudClassifier(X.shape[1])
        criterion = nn.BCELoss()  # type: ignore
        optimiser = optim.Adam(model.parameters(), lr=0.001)  # type: ignore

        model.train()  # type: ignore
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:  # type: ignore
                optimiser.zero_grad()  # type: ignore
                outputs = model(batch_x).squeeze()  # type: ignore
                loss = criterion(outputs, batch_y)  # type: ignore
                loss.backward()  # type: ignore
                optimiser.step()  # type: ignore
                epoch_loss += float(loss.item())  # type: ignore
            print(f'Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}')

        model.eval()  # type: ignore
        with torch.no_grad():  # type: ignore
            test_t = torch.from_numpy(X_test).float()  # type: ignore
            preds = (model(test_t).squeeze().numpy() > 0.5).astype(int)  # type: ignore
        acc = accuracy_score(y_test, preds) if HAS_SKLEARN else 0.0
        prec = precision_score(y_test, preds, zero_division=0) if HAS_SKLEARN else 0.0
        rec = recall_score(y_test, preds, zero_division=0) if HAS_SKLEARN else 0.0
        f1 = f1_score(y_test, preds, zero_division=0) if HAS_SKLEARN else 0.0
        print(f"Torch Metrics: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

    else:
        if not HAS_SKLEARN:
            raise RuntimeError('No ML backend available.')
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200, class_weight='balanced')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"Sklearn Metrics: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

    fraud_vectors = X[fraud_mask]
    if len(fraud_vectors) == 0:
        fallback_amounts = np.asarray(df['amount'].values, dtype=np.float64)
        idx_top = np.argsort(fallback_amounts)[-50:]
        fraud_vectors = X[idx_top]

    fraud_idx = np.where(fraud_mask)[0]
    fraud_df = df.iloc[fraud_idx]
    amounts = np.asarray(fraud_df['amount'].values, dtype=np.float64)
    balance_changes = np.abs(
        np.asarray(fraud_df['oldbalanceOrg'].values, dtype=np.float64) - 
        np.asarray(fraud_df['newbalanceOrig'].values, dtype=np.float64)
    )
    fraud_texts = [f"High amount transfer: {amt:.0f}, balance change: {bc:.0f}" 
                for amt, bc in zip(amounts[:1000], balance_changes[:1000])]
    if len(fraud_texts) == 0:
        fraud_texts = [f"Pattern {i}" for i in range(len(fraud_vectors))]

    return model, fraud_vectors[:1000], fraud_texts