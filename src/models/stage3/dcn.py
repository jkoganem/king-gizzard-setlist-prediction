"""DCN V2 (Deep & Cross Network Version 2) for Setlist Prediction.

Based on "DCN V2: Improved Deep & Cross Network and Practical Lessons
for Web-scale Learning to Rank Systems" (Wang et al., KDD 2021)

Key innovation: Cross network with weight MATRIX (not vector) for
explicit bounded-degree feature interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetworkV2(nn.Module):
    """Cross Network V2 with weight matrices for bit-wise interactions.

    Models explicit feature interactions up to degree (num_layers + 1).
    More expressive than DCN V1 which used weight vectors.
    """

    def __init__(self, input_dim, num_layers=3, low_rank=None):
        """Args:
        input_dim: Dimension of input features
        num_layers: Number of cross layers (degree of polynomial)
        low_rank: If not None, use low-rank approximation (W = U · V^T)
                 Reduces parameters from d^2 to 2*d*r.

        """
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.cross_layers = nn.ModuleList()

        for _ in range(num_layers):
            if low_rank is None:
                # Full-rank weight matrix: d x d
                self.cross_layers.append(nn.Linear(input_dim, input_dim, bias=True))
            else:
                # Low-rank approximation: W = U · V^T where U: d x r, V: d x r
                self.cross_layers.append(LowRankCrossLayer(input_dim, low_rank))

    def forward(self, x0):
        """Args:
            x0: (batch, input_dim) - Input features.

        Returns:
            (batch, input_dim) - Cross network output

        Computes:
            x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
        where ⊙ is element-wise product

        """
        x_l = x0  # x_0

        for cross_layer in self.cross_layers:
            # x_{l+1} = x_0 ⊙ (W · x_l + b) + x_l
            xl_transformed = cross_layer(x_l)  # W · x_l + b
            x_l = x0 * xl_transformed + x_l  # Element-wise product + residual

        return x_l


class LowRankCrossLayer(nn.Module):
    """Low-rank cross layer: W = U · V^T.

    Reduces parameters from d^2 to 2*d*r while maintaining expressiveness.
    """

    def __init__(self, input_dim, rank):
        super().__init__()
        self.U = nn.Linear(input_dim, rank, bias=False)
        self.V = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        """Args:
            x: (batch, input_dim).

        Returns:
            (batch, input_dim) - W · x + b where W = U · V^T

        """
        return self.V(self.U(x)) + self.bias


class DCNV2(nn.Module):
    """Deep & Cross Network V2 for setlist prediction.

    Architecture:
        Input -> Cross Network (explicit interactions) ──┐
                                                        ├─-> Concat -> Output
        Input -> Deep Network (implicit interactions) ──┘
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
        cross_layers=3,
        deep_layers=(256, 128, 64),
        low_rank=32,
        dropout=0.3,
        use_batch_norm=True,
        structure="parallel",  # 'parallel' or 'stacked'
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
        cross_layers: Number of cross layers (polynomial degree - 1)
        deep_layers: Tuple of hidden layer sizes for deep component
        low_rank: Rank for low-rank approximation (None for full-rank)
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        structure: 'parallel' (cross and deep in parallel) or
                  'stacked' (deep on top of cross).

        """
        super().__init__()

        self.structure = structure

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

        # Total input dimension
        total_input_dim = (
            song_emb_dim + venue_emb_dim + tour_emb_dim + country_emb_dim + tabular_dim
        )

        # Cross Network
        self.cross_network = CrossNetworkV2(
            input_dim=total_input_dim, num_layers=cross_layers, low_rank=low_rank
        )

        # Deep Network
        deep_modules = []
        input_dim = total_input_dim

        for hidden_dim in deep_layers:
            deep_modules.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                deep_modules.append(nn.BatchNorm1d(hidden_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.deep_network = nn.Sequential(*deep_modules)

        # Final output layer
        if structure == "parallel":
            # Concatenate cross and deep outputs
            final_input_dim = total_input_dim + deep_layers[-1]
        else:  # stacked
            # Only deep output (which takes cross output as input)
            final_input_dim = deep_layers[-1]

        self.final_layer = nn.Linear(final_input_dim, 1)

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
        song_emb = self.song_emb(song_ids)
        venue_emb = self.venue_emb(venue_ids)
        tour_emb = self.tour_emb(tour_ids)
        country_emb = self.country_emb(country_ids)

        # Concatenate all features
        x = torch.cat(
            [song_emb, venue_emb, tour_emb, country_emb, tabular_features], dim=1
        )  # (batch, total_input_dim)

        if self.structure == "parallel":
            # Parallel: Cross and Deep process input independently
            cross_out = self.cross_network(x)  # (batch, total_input_dim)
            deep_out = self.deep_network(x)  # (batch, deep_layers[-1])

            # Concatenate cross and deep outputs
            combined = torch.cat([cross_out, deep_out], dim=1)

        else:  # stacked
            # Stacked: Deep network on top of cross network
            cross_out = self.cross_network(x)
            deep_out = self.deep_network(cross_out)
            combined = deep_out

        # Final prediction
        logit = self.final_layer(combined)  # (batch, 1)
        output = torch.sigmoid(logit).squeeze(-1)  # (batch,)

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


class WideAndDeep(nn.Module):
    """Wide & Deep Learning (simpler baseline compared to DCN V2).

    Architecture:
        Wide (linear) ──┐
                       ├─-> Output
        Deep (MLP)    ──┘
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
        deep_layers=(256, 128, 64),
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

        # Wide component (linear)
        self.wide = nn.Linear(total_input_dim, 1)

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

        # Final layer for deep
        deep_modules.append(nn.Linear(input_dim, 1))

        self.deep = nn.Sequential(*deep_modules)

    def forward(self, song_ids, venue_ids, tour_ids, country_ids, tabular_features):
        """Args:
            song_ids: (batch_size,)
            venue_ids: (batch_size,)
            tour_ids: (batch_size,)
            country_ids: (batch_size,)
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

        # Wide and Deep
        wide_out = self.wide(x)  # (batch, 1)
        deep_out = self.deep(x)  # (batch, 1)

        # Combine
        logit = wide_out + deep_out  # (batch, 1)
        output = torch.sigmoid(logit).squeeze(-1)  # (batch,)

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
