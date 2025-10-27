#!/usr/bin/env python3
"""
Stage 3A: Neural Models with Tabular Features Only (Production-Ready)

This experiment tests if neural networks can beat XGBoost when using the SAME
33 engineered features (no categorical embeddings).

Key Design Decision:
- Uses aggregate features only (venue_play_rate, country_play_rate, etc.)
- NO raw categorical IDs (venue_id, tour_name)
- Production-ready: works for any future show (new venues, new tours)
- Fair comparison: same feature set as Stage 2 XGBoost

Models:
- Simple MLP (baseline neural model)
- DeepFM (without embeddings - uses tabular features only)

Comparison Target:
- Stage 2 XGBoost (tuned)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.dataio import load_all_data, save_model
from src.features.engineer_features import engineer_all_features
from src.utils.constants import (
    SPECIALIZED_SHOW_DATES,
    ORCHESTRA_SHOW_DATES,
    RAVE_SHOW_DATES,
)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import json
from src.utils.model_config import model_config

print("=" * 80)
print("STAGE 3A: NEURAL MODELS - TABULAR FEATURES ONLY")
print("=" * 80)
print("\nProduction-Ready Approach:")
print("  - Uses 33 engineered features (same as XGBoost)")
print("  - NO categorical embeddings (venue_id, tour_name)")
print("  - Works for new/unseen venues and tours")
print("\nGoal: Can neural nets beat XGBoost on same features?")
print("  XGBoost (Stage 2): 23.52% Recall@15")
print()

# Configuration
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 15

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print()

# Set random seeds
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Load data
print("Loading data...")
shows, songs, setlists, song_tags = load_all_data()
shows["date"] = pd.to_datetime(shows["date"])
shows = shows.sort_values("date")

# Use best configuration from Stage 1/2
print("Applying best configuration: recent (2022+), filtered")
dataset_shows = shows[shows["date"] >= "2022-01-01"].copy()
specialized_dates = set(SPECIALIZED_SHOW_DATES + ORCHESTRA_SHOW_DATES + RAVE_SHOW_DATES)
specialized_dates = [pd.to_datetime(d) for d in specialized_dates]
dataset_shows = dataset_shows[~dataset_shows["date"].isin(specialized_dates)]

print(f"Dataset: {len(dataset_shows)} shows")

# Temporal split
n_train = int(len(dataset_shows) * 0.70)
n_val = int(len(dataset_shows) * 0.85)
train_shows = dataset_shows.iloc[:n_train]
val_shows = dataset_shows.iloc[n_train:n_val]
test_shows = dataset_shows.iloc[n_val:]
train_show_ids = set(train_shows["show_id"].values)

print(f"Train: {len(train_shows)} shows")
print(f"Val: {len(val_shows)} shows")
print(f"Test: {len(test_shows)} shows")

# Engineer features
print("\nEngineering features...")
df, feature_cols = engineer_all_features(
    shows=dataset_shows,
    songs=songs,
    setlists=setlists,
    song_tags=song_tags,
    train_show_ids=train_show_ids,
    include_residency_features=True,
)

print(f"Total examples: {len(df):,}")
print(f"Features: {len(feature_cols)}")

# Split data
val_show_ids = set(val_shows["show_id"].values)
test_show_ids = set(test_shows["show_id"].values)

df_train = df[df["show_id"].isin(train_show_ids)].copy()
df_val = df[df["show_id"].isin(val_show_ids)].copy()
df_test = df[df["show_id"].isin(test_show_ids)].copy()

print(f"\nTrain examples: {len(df_train):,}")
print(f"Val examples: {len(df_val):,}")
print(f"Test examples: {len(df_test):,}")
print(f'Positive rate: {df_train["label"].mean():.4f}')

# Standardize features (important for neural networks)
print("\nStandardizing features...")
scaler = StandardScaler()
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
df_val[feature_cols] = scaler.transform(df_val[feature_cols])
df_test[feature_cols] = scaler.transform(df_test[feature_cols])


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular features only."""

    def __init__(self, df, feature_cols):
        self.features = torch.FloatTensor(df[feature_cols].values)
        self.labels = torch.FloatTensor(df["label"].values)
        self.show_ids = df["show_id"].values
        self.song_ids = df["song_id"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Create datasets
train_dataset = TabularDataset(df_train, feature_cols)
val_dataset = TabularDataset(df_val, feature_cols)
test_dataset = TabularDataset(df_test, feature_cols)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


class SimpleMLP(nn.Module):
    """Simple MLP for tabular data (production-ready baseline)."""

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x)
        return torch.sigmoid(logits).squeeze(-1)


class TabularDeepFM(nn.Module):
    """
    DeepFM adapted for tabular features only (no categorical embeddings).

    Combines:
    1. FM component: 2nd-order feature interactions
    2. Deep component: MLP for complex patterns
    """

    def __init__(self, input_dim, fm_k=10, deep_layers=(256, 128, 64), dropout=0.3):
        super().__init__()

        self.input_dim = input_dim

        # Linear component (1st order)
        self.linear = nn.Linear(input_dim, 1)

        # FM component (2nd order interactions)
        self.V = nn.Parameter(torch.randn(input_dim, fm_k) * 0.01)

        # Deep component (MLP)
        deep_modules = []
        prev_dim = input_dim

        for hidden_dim in deep_layers:
            deep_modules.append(nn.Linear(prev_dim, hidden_dim))
            deep_modules.append(nn.BatchNorm1d(hidden_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        deep_modules.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_modules)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Linear component (1st order)
        linear_out = self.linear(x)

        # FM component (2nd order interactions)
        # Compute: 0.5 * sum[(sum x_i * v_i)^2 - sum (x_i^2 * v_i^2)]
        square_of_sum = torch.pow(torch.matmul(x, self.V), 2)
        sum_of_square = torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2))
        fm_out = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        # Deep component
        deep_out = self.deep(x)

        # Combine all components
        logit = linear_out + fm_out + deep_out + self.bias

        return torch.sigmoid(logit).squeeze(-1)


def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    """Train a neural model with early stopping on validation set."""
    print(f'\n{"="*80}')
    print(f"TRAINING {model_name.upper()}")
    print(f'{"="*80}')

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, test_loader, df_test, dataset_shows, model_name):
    """Evaluate model and compute Recall@K."""
    print(f'\n{"="*80}')
    print(f"EVALUATING {model_name.upper()}")
    print(f'{"="*80}')

    model = model.to(DEVICE)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy())

    # Add predictions to dataframe
    test_df = df_test.copy()
    test_df["y_pred"] = all_preds
    test_df["y_true"] = df_test["label"].values

    # Standard metrics
    auc = roc_auc_score(test_df["y_true"], test_df["y_pred"])
    logloss = log_loss(test_df["y_true"], test_df["y_pred"])

    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {logloss:.4f}")

    # Recall@K with marathon-aware K values
    # Identify marathon shows
    marathon_show_ids = set(
        dataset_shows[dataset_shows["is_marathon"] == 1]["show_id"].values
    )

    recalls = {}
    precisions = {}
    recalls_regular = {}
    recalls_marathon = {}

    for k in [5, 10, 15, 20, 24]:
        show_recalls = []
        show_precisions = []
        regular_recalls = []
        marathon_recalls = []

        for show_id in test_df["show_id"].unique():
            show_data = test_df[test_df["show_id"] == show_id]
            is_marathon = show_id in marathon_show_ids

            # Use dynamic K: 24 for marathon, 15 for regular (when computing @15)
            if k == 15:
                dynamic_k = 24 if is_marathon else 15
            else:
                dynamic_k = k

            top_k = show_data.nlargest(dynamic_k, "y_pred")["song_id"].values
            actual = show_data[show_data["y_true"] == 1]["song_id"].values

            if len(actual) > 0:
                hits = len(set(top_k) & set(actual))
                recall = hits / len(actual)
                show_recalls.append(recall)
                show_precisions.append(hits / dynamic_k)

                # Track separately for regular vs marathon
                if k == 15:
                    if is_marathon:
                        marathon_recalls.append(recall)
                    else:
                        regular_recalls.append(recall)

        recalls[k] = np.mean(show_recalls) if show_recalls else 0.0
        precisions[k] = np.mean(show_precisions) if show_precisions else 0.0

        if k == 15:
            recalls_regular[15] = np.mean(regular_recalls) if regular_recalls else 0.0
            recalls_marathon[24] = (
                np.mean(marathon_recalls) if marathon_recalls else 0.0
            )

    print("\nRecall@K:")
    for k in [5, 10, 15, 20]:
        print(f"  Recall@{k}: {recalls[k]:.4f} ({recalls[k]*100:.2f}%)")
    print(f"\nMarathon-aware breakdown (Recall@15):")
    print(
        f"  Regular shows (K=15): {recalls_regular[15]:.4f} ({recalls_regular[15]*100:.2f}%)"
    )
    print(
        f"  Marathon shows (K=24): {recalls_marathon[24]:.4f} ({recalls_marathon[24]*100:.2f}%)"
    )

    print("\nPrecision@K:")
    for k in [5, 10, 15, 20]:
        print(f"  Precision@{k}: {precisions[k]:.4f}")

    f1_15 = (
        (2 * precisions[15] * recalls[15]) / (precisions[15] + recalls[15])
        if (precisions[15] + recalls[15]) > 0
        else 0.0
    )
    print(f"\nF1@15: {f1_15:.4f}")

    return {
        "auc": auc,
        "logloss": logloss,
        "recall@5": recalls[5],
        "recall@10": recalls[10],
        "recall@15": recalls[15],
        "recall@20": recalls[20],
        "precision@15": precisions[15],
        "f1@15": f1_15,
    }


# Train and evaluate models
results = {}
output_dir = Path("output")
models_dir = output_dir / "models" / "stage3"
models_dir.mkdir(parents=True, exist_ok=True)

# 1. Simple MLP
print("\n" + "=" * 80)
print("MODEL 1: SIMPLE MLP (Baseline Neural Network)")
print("=" * 80)

mlp = SimpleMLP(input_dim=len(feature_cols), hidden_dims=(256, 128, 64), dropout=0.3)

start_time = time.time()
mlp = train_model(mlp, train_loader, val_loader, EPOCHS, LEARNING_RATE, "MLP")
mlp_train_time = time.time() - start_time

mlp_metrics = evaluate_model(mlp, test_loader, df_test, dataset_shows, "MLP")
mlp_metrics["train_time"] = mlp_train_time
results["mlp_tabular"] = mlp_metrics

# Save model
torch.save(mlp.state_dict(), models_dir / "mlp_tabular_stage3a.pt")

# 2. Tabular DeepFM
print("\n" + "=" * 80)
print("MODEL 2: DEEPFM (Tabular Features Only)")
print("=" * 80)

deepfm = TabularDeepFM(
    input_dim=len(feature_cols), fm_k=10, deep_layers=(256, 128, 64), dropout=0.3
)

start_time = time.time()
deepfm = train_model(deepfm, train_loader, val_loader, EPOCHS, LEARNING_RATE, "DeepFM")
deepfm_train_time = time.time() - start_time

deepfm_metrics = evaluate_model(deepfm, test_loader, df_test, dataset_shows, "DeepFM")
deepfm_metrics["train_time"] = deepfm_train_time
results["deepfm_tabular"] = deepfm_metrics

# Save model
torch.save(deepfm.state_dict(), models_dir / "deepfm_tabular_stage3a.pt")

# 3. TabNet
print("\n" + "=" * 80)
print("MODEL 3: TABNET (Attentive Interpretable Tabular Learning)")
print("=" * 80)
print("NOTE: TabNet implementation requires categorical embeddings.")
print("Skipping TabNet for tabular-only experiment (Stage 3A).")
print("TabNet will be tested in Stage 3B with full embeddings.")
print("=" * 80)

# TabNet requires categorical embeddings (song_id, venue_id, etc.)
# This experiment (3A) uses only tabular features (production-ready)
# Skip TabNet - it's not compatible with tabular-only approach

# Save results
print("\n" + "=" * 80)
print("STAGE 3A RESULTS SUMMARY")
print("=" * 80)

reports_dir = output_dir / "reports" / "stage3"
reports_dir.mkdir(parents=True, exist_ok=True)

results_path = reports_dir / "stage3a_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_path}")

# Comparison table
print("\n" + "=" * 80)
print("MODEL COMPARISON (Same 33 Features)")
print("=" * 80)
print(f'\n{"Model":<25} {"Recall@15":<12} {"AUC":<8} {"Train Time":<12}')
print("-" * 65)
print(f'{"XGBoost (Stage 2)":<25} {"23.52%":<12} {"0.7495":<8} {"~1 min":<12}')
print(f'{"Logistic (Stage 1)":<25} {"21.97%":<12} {"0.7377":<8} {"~2 sec":<12}')

for model_name, metrics in results.items():
    name = model_name.replace("_", " ").title()
    recall = f"{metrics['recall@15']*100:.2f}%"
    auc = f"{metrics['auc']:.4f}"
    train_time = f"{metrics['train_time']:.1f}s"
    print(f"{name:<25} {recall:<12} {auc:<8} {train_time:<12}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("Production-Ready: All models work for new venues/tours")
print("Feature Set: Identical 33 features used by XGBoost")
print("Comparison: Neural vs Tree-based on same feature engineering")
print()
print("=" * 80)
print("STAGE 3A COMPLETED")
print("=" * 80)
print()
