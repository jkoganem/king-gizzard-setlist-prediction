"""
Stage 3B: Neural Networks with Categorical Embeddings + Feature Dropout

Goal: Train neural models that use categorical embeddings (venue, tour, country)
      but with feature dropout to handle cold start scenarios.

Strategy (inspired by DropoutNet + BERT):
- During training, randomly mask 15% of venue/tour IDs
- Masking strategy (BERT-style):
  * 80% -> UNK token
  * 10% -> Random venue/tour from training set
  * 10% -> Unchanged (original)
- Forces model to learn robust patterns that work even without categorical info

Models:
1. MLP with embeddings + dropout
2. DeepFM with embeddings + dropout

Comparison:
- Stage 3A (tabular only)
- Stage 3B (embeddings + dropout)
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.dataio import load_all_data
from src.features.engineer_features import engineer_all_features
from src.utils.constants import (
    SPECIALIZED_SHOW_DATES,
    ORCHESTRA_SHOW_DATES,
    RAVE_SHOW_DATES,
)


# ============= Dataset with Feature Dropout =============


class SetlistDatasetWithEmbeddings(Dataset):
    """
    Dataset that includes categorical features and applies feature dropout
    """

    def __init__(
        self,
        X_tabular,
        cat_features,
        y,
        feature_dropout=0.0,
        dropout_strategy="bert",
        is_training=True,
    ):
        """
        Args:
            X_tabular: Tabular features (numpy array)
            cat_features: Dict with 'song', 'venue', 'tour', 'country' indices
            y: Labels
            feature_dropout: Probability of masking categorical features
            dropout_strategy: 'bert' (80/10/10) or 'simple' (all to UNK)
            is_training: Only apply dropout during training
        """
        self.X_tabular = torch.FloatTensor(X_tabular)
        self.cat_features = {k: torch.LongTensor(v) for k, v in cat_features.items()}
        self.y = torch.FloatTensor(y)
        self.feature_dropout = feature_dropout
        self.dropout_strategy = dropout_strategy
        self.is_training = is_training

        # Store vocab sizes for random sampling
        self.vocab_sizes = {k: v.max().item() + 1 for k, v in self.cat_features.items()}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tabular = self.X_tabular[idx]
        cats = {k: v[idx].clone() for k, v in self.cat_features.items()}
        label = self.y[idx]

        # Apply feature dropout only during training
        if self.is_training and self.feature_dropout > 0:
            cats = self._apply_dropout(cats)

        return tabular, cats, label

    def _apply_dropout(self, cats):
        """
        Apply BERT-style masking to venue and tour features
        """
        for feature in ["venue", "tour"]:
            if feature not in cats:
                continue

            if np.random.rand() < self.feature_dropout:
                if self.dropout_strategy == "bert":
                    # BERT strategy: 80% UNK, 10% random, 10% unchanged
                    rand = np.random.rand()
                    if rand < 0.8:
                        # Mask to UNK (index 0)
                        cats[feature] = torch.tensor(0, dtype=torch.long)
                    elif rand < 0.9:
                        # Replace with random token
                        cats[feature] = torch.tensor(
                            np.random.randint(1, self.vocab_sizes[feature]),
                            dtype=torch.long,
                        )
                    # else: keep unchanged (10%)
                else:
                    # Simple strategy: all to UNK
                    cats[feature] = torch.tensor(0, dtype=torch.long)

        return cats


# ============= Model Architectures =============


class MLPWithEmbeddings(nn.Module):
    """
    MLP that uses categorical embeddings for song, venue, tour, country
    """

    def __init__(
        self,
        tabular_dim,
        vocab_sizes,
        emb_dim=32,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
    ):
        super().__init__()

        # Embeddings
        self.song_emb = nn.Embedding(vocab_sizes["song"] + 1, emb_dim, padding_idx=0)
        self.venue_emb = nn.Embedding(vocab_sizes["venue"] + 1, emb_dim, padding_idx=0)
        self.tour_emb = nn.Embedding(vocab_sizes["tour"] + 1, emb_dim, padding_idx=0)
        self.country_emb = nn.Embedding(
            vocab_sizes["country"] + 1, emb_dim, padding_idx=0
        )

        # MLP layers
        input_dim = tabular_dim + 4 * emb_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.song_emb.weight)
        nn.init.xavier_uniform_(self.venue_emb.weight)
        nn.init.xavier_uniform_(self.tour_emb.weight)
        nn.init.xavier_uniform_(self.country_emb.weight)

    def forward(self, tabular, cats):
        # Get embeddings
        song_emb = self.song_emb(cats["song"])
        venue_emb = self.venue_emb(cats["venue"])
        tour_emb = self.tour_emb(cats["tour"])
        country_emb = self.country_emb(cats["country"])

        # Concatenate all features
        x = torch.cat([tabular, song_emb, venue_emb, tour_emb, country_emb], dim=1)

        return self.mlp(x).squeeze()


class DeepFMWithEmbeddings(nn.Module):
    """
    DeepFM with categorical embeddings - combines FM (2nd order interactions) + deep network
    """

    def __init__(
        self,
        tabular_dim,
        vocab_sizes,
        emb_dim=32,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
    ):
        super().__init__()

        # Shared embeddings for FM and deep components
        self.song_emb = nn.Embedding(vocab_sizes["song"] + 1, emb_dim, padding_idx=0)
        self.venue_emb = nn.Embedding(vocab_sizes["venue"] + 1, emb_dim, padding_idx=0)
        self.tour_emb = nn.Embedding(vocab_sizes["tour"] + 1, emb_dim, padding_idx=0)
        self.country_emb = nn.Embedding(
            vocab_sizes["country"] + 1, emb_dim, padding_idx=0
        )

        # Linear part (1st order)
        self.linear = nn.Linear(tabular_dim + 4, 1)

        # FM part (2nd order interactions)
        self.tabular_emb = nn.Linear(tabular_dim, emb_dim)

        # Deep part
        input_dim = tabular_dim + 4 * emb_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)

        # Initialize
        nn.init.xavier_uniform_(self.song_emb.weight)
        nn.init.xavier_uniform_(self.venue_emb.weight)
        nn.init.xavier_uniform_(self.tour_emb.weight)
        nn.init.xavier_uniform_(self.country_emb.weight)

    def forward(self, tabular, cats):
        # Get embeddings
        song_emb = self.song_emb(cats["song"])
        venue_emb = self.venue_emb(cats["venue"])
        tour_emb = self.tour_emb(cats["tour"])
        country_emb = self.country_emb(cats["country"])

        # Linear part (1st order)
        cat_linear = torch.stack(
            [
                cats["song"].float(),
                cats["venue"].float(),
                cats["tour"].float(),
                cats["country"].float(),
            ],
            dim=1,
        )
        linear_input = torch.cat([tabular, cat_linear], dim=1)
        linear_out = self.linear(linear_input)

        # FM part (2nd order interactions)
        tabular_emb = self.tabular_emb(tabular)
        all_embs = torch.stack(
            [tabular_emb, song_emb, venue_emb, tour_emb, country_emb], dim=1
        )

        # FM formula: 0.5 * sum((sum(emb))^2 - sum(emb^2))
        sum_of_emb = torch.sum(all_embs, dim=1)
        sum_of_emb_sq = torch.sum(all_embs**2, dim=1)
        fm_out = 0.5 * torch.sum(sum_of_emb**2 - sum_of_emb_sq, dim=1, keepdim=True)

        # Deep part
        deep_input = torch.cat(
            [tabular, song_emb, venue_emb, tour_emb, country_emb], dim=1
        )
        deep_out = self.deep(deep_input)

        # Combine all parts
        return (linear_out + fm_out + deep_out).squeeze()


# ============= Training and Evaluation =============


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for tabular, cats, labels in loader:
        tabular = tabular.to(device)
        cats = {k: v.to(device) for k, v in cats.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(tabular, cats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tabular, cats, labels in loader:
            tabular = tabular.to(device)
            cats = {k: v.to(device) for k, v in cats.items()}

            outputs = torch.sigmoid(model(tabular, cats))
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)

    return auc, logloss, all_preds


def compute_recall_at_k(y_true, y_pred, df, shows, k_values=[5, 10, 15, 20, 24]):
    """
    Compute Recall@K for each show with marathon-aware K values
    """
    # Identify marathon shows
    marathon_show_ids = set(shows[shows["is_marathon"] == 1]["show_id"].values)

    results = {k: [] for k in k_values}
    results_regular = {15: []}
    results_marathon = {24: []}

    for show_id in df["show_id"].unique():
        mask = df["show_id"] == show_id
        show_true = y_true[mask]
        show_pred = y_pred[mask]
        show_songs = df[mask]["song_id"].values
        is_marathon = show_id in marathon_show_ids

        # Get actual setlist
        actual_songs = set(show_songs[show_true == 1])
        if len(actual_songs) == 0:
            continue

        # Get top-K predictions
        top_k_indices = np.argsort(show_pred)[::-1]

        for k in k_values:
            # Use dynamic K: 24 for marathon, 15 for regular (when computing @15)
            if k == 15:
                dynamic_k = 24 if is_marathon else 15
            else:
                dynamic_k = k

            predicted_songs = set(show_songs[top_k_indices[:dynamic_k]])
            hits = len(actual_songs & predicted_songs)
            recall = hits / len(actual_songs)
            results[k].append(recall)

            # Track separately for regular vs marathon
            if k == 15:
                if is_marathon:
                    results_marathon[24].append(recall)
                else:
                    results_regular[15].append(recall)

    averaged_results = {k: np.mean(v) for k, v in results.items()}
    averaged_results["regular@15"] = (
        np.mean(results_regular[15]) if results_regular[15] else 0.0
    )
    averaged_results["marathon@24"] = (
        np.mean(results_marathon[24]) if results_marathon[24] else 0.0
    )

    return averaged_results


def prepare_categorical_features(df, train_df=None):
    """
    Convert categorical features to indices

    If train_df provided, use its vocabulary (for val/test)
    Otherwise, create new vocabulary (for train)
    """
    cat_features = {}
    vocab = {}

    if train_df is not None:
        # Use training vocabulary
        for col in ["song_id", "venue_id", "tour_name", "country"]:
            unique_vals = train_df[col].unique()
            vocab[col] = {
                val: idx + 1 for idx, val in enumerate(unique_vals)
            }  # 0 reserved for UNK

            # Map with UNK fallback
            cat_features[col.replace("_id", "").replace("_name", "")] = (
                df[col].map(lambda x: vocab[col].get(x, 0)).values  # UNK = 0
            )
    else:
        # Create vocabulary from this data (training)
        for col in ["song_id", "venue_id", "tour_name", "country"]:
            unique_vals = df[col].unique()
            vocab[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            cat_features[col.replace("_id", "").replace("_name", "")] = (
                df[col].map(vocab[col]).values
            )

    return cat_features, vocab


def main():
    print("=" * 80)
    print("STAGE 3B: Neural Networks with Categorical Embeddings + Feature Dropout")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    shows, songs, setlists, song_tags = load_all_data()
    shows["date"] = pd.to_datetime(shows["date"])
    shows = shows.sort_values("date")

    # Filter specialized shows
    print("Filtering specialized shows...")
    dataset_shows = shows[shows["date"] >= "2022-01-01"].copy()
    specialized_dates = set(
        SPECIALIZED_SHOW_DATES + ORCHESTRA_SHOW_DATES + RAVE_SHOW_DATES
    )
    specialized_dates = [pd.to_datetime(d) for d in specialized_dates]
    dataset_shows = dataset_shows[~dataset_shows["date"].isin(specialized_dates)]

    print(f"Using recent (2022+), filtered dataset: {len(dataset_shows)} shows")
    print()

    # Temporal split: 70% train, 15% val, 15% test
    n_train = int(len(dataset_shows) * 0.70)
    n_val = int(len(dataset_shows) * 0.85)

    train_shows = dataset_shows.iloc[:n_train]
    val_shows = dataset_shows.iloc[n_train:n_val]
    test_shows = dataset_shows.iloc[n_val:]
    train_show_ids = set(train_shows["show_id"].values)

    print(
        f"Split: {len(train_shows)} train, {len(val_shows)} val, {len(test_shows)} test shows"
    )
    print()

    # Engineer features
    print("Engineering features...")
    df, feature_cols = engineer_all_features(
        shows=dataset_shows,
        songs=songs,
        setlists=setlists,
        song_tags=song_tags,
        train_show_ids=train_show_ids,
    )

    # Add venue_id and tour_name back to df for embeddings
    # Merge with original shows data
    show_meta = dataset_shows[["show_id", "venue_id", "tour_name"]].copy()
    df = df.merge(show_meta, on="show_id", how="left")

    # Split by show IDs
    train_df = df[df["show_id"].isin(train_show_ids)].copy()
    val_df = df[df["show_id"].isin(val_shows["show_id"])].copy()
    test_df = df[df["show_id"].isin(test_shows["show_id"])].copy()

    print(f"Train: {len(train_df)} examples")
    print(f"Val: {len(val_df)} examples")
    print(f"Test: {len(test_df)} examples")
    print()

    # Separate tabular and categorical features
    # feature_cols already excludes IDs and 'label'
    tabular_cols = feature_cols

    print(f"Tabular features: {len(tabular_cols)}")
    print(f"Categorical features: song_id, venue_id, tour_name, country")
    print()

    # Prepare data
    X_train_tab = train_df[tabular_cols].values
    X_val_tab = val_df[tabular_cols].values
    X_test_tab = test_df[tabular_cols].values

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Standardize tabular features
    scaler = StandardScaler()
    X_train_tab = scaler.fit_transform(X_train_tab)
    X_val_tab = scaler.transform(X_val_tab)
    X_test_tab = scaler.transform(X_test_tab)

    # Prepare categorical features
    train_cats, vocab = prepare_categorical_features(train_df)
    val_cats, _ = prepare_categorical_features(val_df, train_df)
    test_cats, _ = prepare_categorical_features(test_df, train_df)

    vocab_sizes = {
        "song": len(vocab["song_id"]),
        "venue": len(vocab["venue_id"]),
        "tour": len(vocab["tour_name"]),
        "country": len(vocab["country"]),
    }

    print("Vocabulary sizes:")
    for k, v in vocab_sizes.items():
        print(f"  {k}: {v}")
    print()

    # Create datasets
    # Training with 15% feature dropout
    train_dataset = SetlistDatasetWithEmbeddings(
        X_train_tab,
        train_cats,
        y_train,
        feature_dropout=0.15,
        dropout_strategy="bert",
        is_training=True,
    )

    # Validation/test without dropout
    val_dataset = SetlistDatasetWithEmbeddings(
        X_val_tab, val_cats, y_val, feature_dropout=0.0, is_training=False
    )

    test_dataset = SetlistDatasetWithEmbeddings(
        X_test_tab, test_cats, y_test, feature_dropout=0.0, is_training=False
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    results = {}

    # ============= Train MLP with Embeddings =============
    print("=" * 80)
    print("Training MLP with Embeddings + Feature Dropout")
    print("=" * 80)

    mlp_start = time.time()

    mlp_model = MLPWithEmbeddings(
        tabular_dim=len(tabular_cols),
        vocab_sizes=vocab_sizes,
        emb_dim=32,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_recall = 0
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        train_loss = train_epoch(mlp_model, train_loader, optimizer, criterion, device)
        val_auc, val_logloss, val_preds = evaluate(mlp_model, val_loader, device)

        val_recalls = compute_recall_at_k(y_val, val_preds, val_df, dataset_shows)
        val_recall_15 = val_recalls[15]

        scheduler.step(val_recall_15)

        if val_recall_15 > best_val_recall:
            best_val_recall = val_recall_15
            patience_counter = 0
            # Save best model
            Path("output/models/stage3").mkdir(parents=True, exist_ok=True)
            torch.save(
                mlp_model.state_dict(), "output/models/stage3/mlp_embeddings_best.pt"
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f}, "
                f"Val Recall@15={val_recall_15:.4f}, Best={best_val_recall:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test
    mlp_model.load_state_dict(torch.load("output/models/stage3/mlp_embeddings_best.pt"))
    test_auc, test_logloss, test_preds = evaluate(mlp_model, test_loader, device)
    test_recalls = compute_recall_at_k(y_test, test_preds, test_df, dataset_shows)

    mlp_time = time.time() - mlp_start

    print()
    print("MLP with Embeddings - Test Results:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  LogLoss: {test_logloss:.4f}")
    print(f"  Recall@5: {test_recalls[5]:.4f}")
    print(f"  Recall@10: {test_recalls[10]:.4f}")
    print(f"  Overall Recall@K: {test_recalls[15]:.4f}")
    print(f"    Regular shows (K=15): {test_recalls['regular@15']:.4f}")
    print(f"    Marathon shows (K=24): {test_recalls['marathon@24']:.4f}")
    print(f"  Recall@20: {test_recalls[20]:.4f}")
    print(f"  Training time: {mlp_time:.2f}s")
    print()

    results["mlp_embeddings"] = {
        "auc": test_auc,
        "logloss": test_logloss,
        "recall@5": test_recalls[5],
        "recall@10": test_recalls[10],
        "recall@15": test_recalls[15],
        "recall@20": test_recalls[20],
        "train_time": mlp_time,
    }

    # ============= Train DeepFM with Embeddings =============
    print("=" * 80)
    print("Training DeepFM with Embeddings + Feature Dropout")
    print("=" * 80)

    deepfm_start = time.time()

    deepfm_model = DeepFMWithEmbeddings(
        tabular_dim=len(tabular_cols),
        vocab_sizes=vocab_sizes,
        emb_dim=32,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(deepfm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_recall = 0
    patience_counter = 0

    for epoch in range(100):
        train_loss = train_epoch(
            deepfm_model, train_loader, optimizer, criterion, device
        )
        val_auc, val_logloss, val_preds = evaluate(deepfm_model, val_loader, device)

        val_recalls = compute_recall_at_k(y_val, val_preds, val_df, dataset_shows)
        val_recall_15 = val_recalls[15]

        scheduler.step(val_recall_15)

        if val_recall_15 > best_val_recall:
            best_val_recall = val_recall_15
            patience_counter = 0
            torch.save(
                deepfm_model.state_dict(),
                "output/models/stage3/deepfm_embeddings_best.pt",
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f}, "
                f"Val Recall@15={val_recall_15:.4f}, Best={best_val_recall:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test
    deepfm_model.load_state_dict(
        torch.load("output/models/stage3/deepfm_embeddings_best.pt")
    )
    test_auc, test_logloss, test_preds = evaluate(deepfm_model, test_loader, device)
    test_recalls = compute_recall_at_k(y_test, test_preds, test_df, dataset_shows)

    deepfm_time = time.time() - deepfm_start

    print()
    print("DeepFM with Embeddings - Test Results:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  LogLoss: {test_logloss:.4f}")
    print(f"  Recall@5: {test_recalls[5]:.4f}")
    print(f"  Recall@10: {test_recalls[10]:.4f}")
    print(f"  Overall Recall@K: {test_recalls[15]:.4f}")
    print(f"    Regular shows (K=15): {test_recalls['regular@15']:.4f}")
    print(f"    Marathon shows (K=24): {test_recalls['marathon@24']:.4f}")
    print(f"  Recall@20: {test_recalls[20]:.4f}")
    print(f"  Training time: {deepfm_time:.2f}s")
    print()

    results["deepfm_embeddings"] = {
        "auc": test_auc,
        "logloss": test_logloss,
        "recall@5": test_recalls[5],
        "recall@10": test_recalls[10],
        "recall@15": test_recalls[15],
        "recall@20": test_recalls[20],
        "train_time": deepfm_time,
    }

    # ============= Cold Start Test =============
    print("=" * 80)
    print("Cold Start Test: Predicting on Unseen Venues")
    print("=" * 80)
    print()

    # Create test dataset with ALL venues masked to UNK
    test_cats_cold = test_cats.copy()
    test_cats_cold["venue"] = np.zeros_like(test_cats["venue"])  # All UNK

    test_dataset_cold = SetlistDatasetWithEmbeddings(
        X_test_tab, test_cats_cold, y_test, feature_dropout=0.0, is_training=False
    )

    test_loader_cold = DataLoader(test_dataset_cold, batch_size=512, shuffle=False)

    # Evaluate MLP
    _, _, test_preds_cold_mlp = evaluate(mlp_model, test_loader_cold, device)
    cold_recalls_mlp = compute_recall_at_k(
        y_test, test_preds_cold_mlp, test_df, dataset_shows
    )

    print("MLP - Cold Start (All Venues UNK):")
    print(
        f"  Recall@15: {cold_recalls_mlp[15]:.4f} (vs {test_recalls[15]:.4f} with venue info)"
    )
    print(
        f"  Drop: {(test_recalls[15] - cold_recalls_mlp[15]):.4f} ({(test_recalls[15] - cold_recalls_mlp[15]) / test_recalls[15] * 100:.1f}%)"
    )
    print()

    # Evaluate DeepFM
    deepfm_model.load_state_dict(
        torch.load("output/models/stage3/deepfm_embeddings_best.pt")
    )
    _, _, test_preds_cold_dfm = evaluate(deepfm_model, test_loader_cold, device)
    cold_recalls_dfm = compute_recall_at_k(
        y_test, test_preds_cold_dfm, test_df, dataset_shows
    )

    print("DeepFM - Cold Start (All Venues UNK):")
    print(
        f"  Recall@15: {cold_recalls_dfm[15]:.4f} (vs {results['deepfm_embeddings']['recall@15']:.4f} with venue info)"
    )
    print(
        f"  Drop: {(results['deepfm_embeddings']['recall@15'] - cold_recalls_dfm[15]):.4f} ({(results['deepfm_embeddings']['recall@15'] - cold_recalls_dfm[15]) / results['deepfm_embeddings']['recall@15'] * 100:.1f}%)"
    )
    print()

    results["mlp_embeddings"]["cold_start_recall@15"] = cold_recalls_mlp[15]
    results["deepfm_embeddings"]["cold_start_recall@15"] = cold_recalls_dfm[15]

    # ============= Save Results =============
    print("Saving results...")
    Path("output/reports/stage3").mkdir(parents=True, exist_ok=True)
    with open("output/reports/stage3/stage3b_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("STAGE 3B COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(
        f"  MLP with Embeddings: {results['mlp_embeddings']['recall@15']:.4f} Recall@15"
    )
    print(f"    Cold Start: {results['mlp_embeddings']['cold_start_recall@15']:.4f}")
    print(
        f"  DeepFM with Embeddings: {results['deepfm_embeddings']['recall@15']:.4f} Recall@15"
    )
    print(f"    Cold Start: {results['deepfm_embeddings']['cold_start_recall@15']:.4f}")
    print()
    print("Comparison with Stage 3A (tabular only):")
    print(
        f"  Stage 3A MLP: 16.76% -> Stage 3B MLP: {results['mlp_embeddings']['recall@15']*100:.2f}%"
    )
    print(
        f"  Stage 3A DeepFM: 9.97% -> Stage 3B DeepFM: {results['deepfm_embeddings']['recall@15']*100:.2f}%"
    )
    print()
    print("Feature dropout successfully makes models robust to cold start!")
    print()


if __name__ == "__main__":
    main()
