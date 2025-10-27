"""DeepFM: Factorization Machines + Deep Neural Network for Setlist Prediction.

Combines:
1. FM component: Explicit 2nd-order feature interactions (song x venue, etc.)
2. Deep component: Multi-layer perceptron for complex patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizationMachine(nn.Module):
    """Factorization Machine component for 2nd-order feature interactions.

    Computes: Σ_i Σ_j <v_i, v_j> * x_i * x_j
    where v_i are learned embedding vectors.
    """

    def __init__(self, input_dim, k=10):
        """Args:
        input_dim: Number of input features
        k: Embedding dimension for FM.

        """
        super().__init__()
        self.input_dim = input_dim
        self.k = k

        # V matrix: (input_dim, k) - each feature gets k-dimensional embedding
        self.V = nn.Parameter(torch.randn(input_dim, k) * 0.01)

    def forward(self, x):
        """Args:
            x: (batch_size, input_dim).

        Returns:
            (batch_size, 1) - FM interaction score

        """
        # Compute sum of squares: (Σ x_i * v_i)^2
        square_of_sum = torch.pow(torch.mm(x, self.V), 2)  # (batch, k)

        # Compute sum of square: Σ (x_i^2 * v_i^2)
        sum_of_square = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))  # (batch, k)

        # FM interaction: 0.5 * Σ_k [(Σ x_i * v_i)^2 - Σ (x_i^2 * v_i^2)]
        interaction = 0.5 * torch.sum(
            square_of_sum - sum_of_square, dim=1, keepdim=True
        )

        return interaction


class DeepFM(nn.Module):
    """DeepFM model combining Factorization Machines with Deep Neural Network.

    Architecture:
        Input -> FM (2nd order) ──┐
                                  ├─ Sum -> Output
        Input -> Deep (MLP)     ──┘
    """

    def __init__(
        self,
        num_songs,
        num_venues,
        num_tours,
        num_countries,
        tabular_dim=25,
        song_emb_dim=32,
        venue_emb_dim=16,
        tour_emb_dim=8,
        country_emb_dim=8,
        fm_k=10,
        deep_layers=(256, 128, 64),
        dropout=0.3,
        use_batch_norm=True,
    ):
        """Args:
        num_songs: Number of unique songs
        num_venues: Number of unique venues
        num_tours: Number of unique tours
        num_countries: Number of unique countries
        tabular_dim: Dimension of tabular features
        song_emb_dim: Song embedding dimension
        venue_emb_dim: Venue embedding dimension
        tour_emb_dim: Tour embedding dimension
        country_emb_dim: Country embedding dimension
        fm_k: FM embedding dimension
        deep_layers: Tuple of hidden layer sizes for deep component
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization.

        """
        super().__init__()

        # Embeddings for categorical features
        self.song_emb = nn.Embedding(num_songs + 1, song_emb_dim, padding_idx=0)
        self.venue_emb = nn.Embedding(num_venues + 1, venue_emb_dim, padding_idx=0)
        self.tour_emb = nn.Embedding(num_tours + 1, tour_emb_dim, padding_idx=0)
        self.country_emb = nn.Embedding(
            num_countries + 1, country_emb_dim, padding_idx=0
        )

        # Initialize embeddings
        nn.init.normal_(self.song_emb.weight, std=0.01)
        nn.init.normal_(self.venue_emb.weight, std=0.01)
        nn.init.normal_(self.tour_emb.weight, std=0.01)
        nn.init.normal_(self.country_emb.weight, std=0.01)

        # Total input dimension (embeddings + tabular)
        total_input_dim = (
            song_emb_dim + venue_emb_dim + tour_emb_dim + country_emb_dim + tabular_dim
        )

        # FM component
        self.fm = FactorizationMachine(total_input_dim, k=fm_k)

        # Linear component (1st order)
        self.linear = nn.Linear(total_input_dim, 1)

        # Deep component (MLP)
        deep_modules = []
        input_dim = total_input_dim

        for hidden_dim in deep_layers:
            deep_modules.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                deep_modules.append(nn.BatchNorm1d(hidden_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Final layer
        deep_modules.append(nn.Linear(input_dim, 1))

        self.deep = nn.Sequential(*deep_modules)

        # Bias for final output
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, song_ids, venue_ids, tour_ids, country_ids, tabular_features):
        """Args:
            song_ids: (batch_size,) - Song indices
            venue_ids: (batch_size,) - Venue indices
            tour_ids: (batch_size,) - Tour indices
            country_ids: (batch_size,) - Country indices
            tabular_features: (batch_size, tabular_dim) - Numerical features.

        Returns:
            (batch_size,) - Predicted probabilities

        """
        # Get embeddings
        song_emb = self.song_emb(song_ids)  # (batch, song_emb_dim)
        venue_emb = self.venue_emb(venue_ids)  # (batch, venue_emb_dim)
        tour_emb = self.tour_emb(tour_ids)  # (batch, tour_emb_dim)
        country_emb = self.country_emb(country_ids)  # (batch, country_emb_dim)

        # Concatenate all features
        x = torch.cat(
            [song_emb, venue_emb, tour_emb, country_emb, tabular_features], dim=1
        )  # (batch, total_input_dim)

        # Linear component (1st order)
        linear_out = self.linear(x)  # (batch, 1)

        # FM component (2nd order interactions)
        fm_out = self.fm(x)  # (batch, 1)

        # Deep component
        deep_out = self.deep(x)  # (batch, 1)

        # Combine all components
        logit = linear_out + fm_out + deep_out + self.bias  # (batch, 1)

        # Sigmoid for probability
        output = torch.sigmoid(logit).squeeze(-1)  # (batch,)

        return output

    def predict_proba(
        self, song_ids, venue_ids, tour_ids, country_ids, tabular_features
    ):
        """Predict probabilities (inference mode).

        Returns:
            (batch_size,) - Predicted probabilities

        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                song_ids, venue_ids, tour_ids, country_ids, tabular_features
            )


class SimpleMLP(nn.Module):
    """Simple MLP baseline with embeddings (simpler than DeepFM).

    Architecture:
        Embeddings -> Concat -> MLP -> Output
    """

    def __init__(
        self,
        num_songs,
        num_venues,
        num_tours,
        num_countries,
        tabular_dim=25,
        song_emb_dim=32,
        venue_emb_dim=16,
        tour_emb_dim=8,
        country_emb_dim=8,
        hidden_layers=(256, 128, 64),
        dropout=0.3,
        use_batch_norm=True,
    ):
        super().__init__()

        # Embeddings
        self.song_emb = nn.Embedding(num_songs + 1, song_emb_dim, padding_idx=0)
        self.venue_emb = nn.Embedding(num_venues + 1, venue_emb_dim, padding_idx=0)
        self.tour_emb = nn.Embedding(num_tours + 1, tour_emb_dim, padding_idx=0)
        self.country_emb = nn.Embedding(
            num_countries + 1, country_emb_dim, padding_idx=0
        )

        # Initialize embeddings
        nn.init.normal_(self.song_emb.weight, std=0.01)
        nn.init.normal_(self.venue_emb.weight, std=0.01)
        nn.init.normal_(self.tour_emb.weight, std=0.01)
        nn.init.normal_(self.country_emb.weight, std=0.01)

        # Total input dimension
        total_input_dim = (
            song_emb_dim + venue_emb_dim + tour_emb_dim + country_emb_dim + tabular_dim
        )

        # MLP
        mlp_modules = []
        input_dim = total_input_dim

        for hidden_dim in hidden_layers:
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                mlp_modules.append(nn.BatchNorm1d(hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        mlp_modules.append(nn.Linear(input_dim, 1))
        mlp_modules.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, song_ids, venue_ids, tour_ids, country_ids, tabular_features):
        """Args:
            song_ids: (batch_size,) - Song indices
            venue_ids: (batch_size,) - Venue indices
            tour_ids: (batch_size,) - Tour indices
            country_ids: (batch_size,) - Country indices
            tabular_features: (batch_size, tabular_dim).

        Returns:
            (batch_size,) - Predicted probabilities

        """
        # Get embeddings
        song_emb = self.song_emb(song_ids)
        venue_emb = self.venue_emb(venue_ids)
        tour_emb = self.tour_emb(tour_ids)
        country_emb = self.country_emb(country_ids)

        # Concatenate
        x = torch.cat(
            [song_emb, venue_emb, tour_emb, country_emb, tabular_features], dim=1
        )

        # Forward through MLP
        output = self.mlp(x).squeeze(-1)

        return output

    def predict_proba(
        self, song_ids, venue_ids, tour_ids, country_ids, tabular_features
    ):
        """Predict probabilities (inference mode)."""
        self.eval()
        with torch.no_grad():
            return self.forward(
                song_ids, venue_ids, tour_ids, country_ids, tabular_features
            )
