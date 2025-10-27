"""Temporal Sets GNN with Frequency/Recency Priors (Stage 5A).

Extends the base Temporal Sets GNN (Stage 4) with SAFERec-style priors:
- Frequency prior: Songs played often are more likely
- Recency prior: Songs played recently are penalized (avoid repeats)

Based on:
- DNNTSP (KDD 2020) for base architecture
- SAFERec (2024) for frequency/recency priors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.stage4.temporal_sets import (
    GraphConvLayer,
    TemporalAttention,
    TemporalSetsGNN,
)


class TemporalSetsGNNWithPriors(TemporalSetsGNN):
    """Temporal Sets GNN augmented with frequency/recency priors.

    Inherits from TemporalSetsGNN and adds:
    - Learnable α (alpha) weight for frequency prior
    - Learnable β (beta) weight for recency prior
    - Additive bias: logits' = logits + α·freq + β·recency

    Args:
        num_songs: Number of songs in vocabulary
        emb_dim: Embedding dimension (default: 64)
        gnn_layers: Number of GCN layers (default: 2)
        num_prev_shows: History length (default: 5)
        use_frequency_prior: Enable frequency prior (default: True)
        use_recency_prior: Enable recency prior (default: True)
        initial_alpha: Initial frequency weight (default: 0.5)
        initial_beta: Initial recency weight (default: 0.5)

    """

    def __init__(
        self,
        num_songs: int,
        emb_dim: int = 64,
        gnn_layers: int = 2,
        num_prev_shows: int = 5,
        use_frequency_prior: bool = True,
        use_recency_prior: bool = True,
        initial_alpha: float = 0.5,
        initial_beta: float = 0.5,
    ):
        # Initialize base GNN
        super().__init__(
            num_songs=num_songs,
            emb_dim=emb_dim,
            gnn_layers=gnn_layers,
            num_prev_shows=num_prev_shows,
        )

        # Prior configuration
        self.use_frequency_prior = use_frequency_prior
        self.use_recency_prior = use_recency_prior

        # Learnable prior weights
        if use_frequency_prior:
            self.alpha = nn.Parameter(torch.tensor(initial_alpha))
            self.register_buffer("freq_bias", torch.zeros(num_songs + 1))
        else:
            self.alpha = None
            self.freq_bias = None

        if use_recency_prior:
            self.beta = nn.Parameter(torch.tensor(initial_beta))
        else:
            self.beta = None

    def set_frequency_prior(self, freq_scores: torch.Tensor) -> None:
        """Set global frequency prior (call once before training).

        Args:
            freq_scores: Tensor of shape [num_songs + 1] with frequency scores
                        Index 0 is for padding/UNK token

        """
        if not self.use_frequency_prior:
            return

        assert freq_scores.shape == (
            self.num_songs + 1,
        ), f"Expected shape ({self.num_songs + 1},), got {freq_scores.shape}"
        self.freq_bias = freq_scores.to(self.freq_bias.device)

    def forward(
        self,
        song_ids,
        prev_setlists,
        venue_ids,
        tour_ids,
        country_ids,
        is_festival,
        is_marathon,
        recency_scores=None,
    ):
        """Forward pass with priors.

        Args:
            song_ids: [batch] - candidate songs
            prev_setlists: [batch, num_prev_shows, max_songs] - previous shows
            venue_ids: [batch] - venue IDs
            tour_ids: [batch] - tour IDs
            country_ids: [batch] - country IDs
            is_festival: [batch] - festival flag (0 or 1)
            is_marathon: [batch] - marathon flag (0 or 1)
            recency_scores: [batch, num_songs+1] - recency penalty scores (optional)

        Returns:
            [batch] - probability of each song being in setlist

        """
        batch_size = song_ids.size(0)

        # 1. Build co-occurrence graph from previous shows
        adj_matrix = self.build_cooccurrence_graph(prev_setlists, self.num_songs)

        # 2. Initialize song embeddings
        all_song_ids = torch.arange(self.num_songs + 1, device=song_ids.device)
        song_embs = self.song_embedding(all_song_ids)  # [num_songs+1, emb_dim]

        # Expand for batch
        song_embs = song_embs.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_songs+1, emb_dim]

        # 3. Apply GNN to get context-aware song embeddings
        for gnn_layer in self.gnn_layers:
            song_embs = gnn_layer(song_embs, adj_matrix)

        # 4. Get embeddings for previous shows
        prev_show_embs = self.get_show_embeddings(prev_setlists, song_embs)

        # 5. Encode current show context (venue, tour, country, is_festival, is_marathon)
        venue_emb = self.venue_embedding(venue_ids)
        tour_emb = self.tour_embedding(tour_ids)
        country_emb = self.country_embedding(country_ids)
        festival_emb = self.festival_embedding(is_festival)
        marathon_emb = self.marathon_embedding(is_marathon)
        context_emb = torch.cat(
            [venue_emb, tour_emb, country_emb, festival_emb, marathon_emb], dim=-1
        )
        context_emb = self.context_proj(context_emb)

        # 6. Temporal attention over previous shows
        temporal_context, _ = self.temporal_attention(context_emb, prev_show_embs)

        # 7. Get candidate song embeddings (after GNN)
        candidate_song_embs = song_embs[torch.arange(batch_size), song_ids]

        # 8. Combine all features and predict
        combined = torch.cat(
            [candidate_song_embs, context_emb, temporal_context], dim=-1
        )
        logits = self.predictor(combined).squeeze(-1)

        # 9. Add frequency/recency priors
        if self.use_frequency_prior:
            # Global frequency bias (same for all songs in batch)
            freq_bias = self.freq_bias[song_ids]  # [batch]
            logits = logits + self.alpha * freq_bias

        if self.use_recency_prior and recency_scores is not None:
            # Song-specific recency penalty (different per example)
            rec_bias = recency_scores[torch.arange(batch_size), song_ids]  # [batch]
            logits = logits + self.beta * rec_bias

        return torch.sigmoid(logits)

    def get_prior_weights(self) -> dict:
        """Get current prior weights for logging.

        Returns:
            Dictionary with 'alpha' and 'beta' values (or None if disabled)

        """
        return {
            "alpha": self.alpha.item() if self.use_frequency_prior else None,
            "beta": self.beta.item() if self.use_recency_prior else None,
        }
