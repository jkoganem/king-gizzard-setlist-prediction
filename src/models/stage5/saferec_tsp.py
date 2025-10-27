"""SAFERec-TSP: Self-Attention and Frequency Enriched Model for Temporal Set Prediction.

Based on "Self-Attention and Frequency Enriched Model for Next Basket Recommendation" (2024).

Key Innovation:
- Transformer encoder on show history (basket-level attention)
- Explicit frequency/recency priors added to logits
- Set-level prediction (vs item-level in original SAFERec)

Architecture:
    Show Encoding: Encode each show as set of song embeddings
    Transformer: Multi-head self-attention over show sequence
    Prediction: Linear projection + frequency/recency priors

Formula:
    logits_final = logits_transformer + α·log(1 + freq) + β·recency
"""

import torch
import torch.nn as nn
import math


class ShowSetEncoder(nn.Module):
    """Encodes a show (set of songs) into a single vector.

    Uses mean pooling over song embeddings.
    """

    def __init__(self, num_songs, emb_dim=64):
        super().__init__()
        self.song_embedding = nn.Embedding(num_songs + 1, emb_dim, padding_idx=0)

    def forward(self, song_ids):
        """Encode a show as mean of song embeddings.

        Args:
            song_ids: [batch_size, max_songs_per_show] with padding_idx=0

        Returns:
            show_embedding: [batch_size, emb_dim]

        """
        song_embs = self.song_embedding(song_ids)  # [batch_size, max_songs, emb_dim]

        # Create mask for non-padding songs
        mask = (song_ids != 0).unsqueeze(-1).float()  # [batch_size, max_songs, 1]

        # Mean pooling with mask
        masked_embs = song_embs * mask
        show_emb = masked_embs.sum(dim=1) / (
            mask.sum(dim=1) + 1e-8
        )  # [batch_size, emb_dim]

        return show_emb


class SAFERecTSP(nn.Module):
    """SAFERec adapted for Temporal Set Prediction.

    Args:
        num_songs: Number of songs in vocabulary
        emb_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum number of previous shows to consider
        use_frequency_prior: Whether to use frequency prior
        use_recency_prior: Whether to use recency prior
        initial_alpha: Initial frequency weight
        initial_beta: Initial recency weight
        dropout: Dropout rate

    """

    def __init__(
        self,
        num_songs: int,
        emb_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 10,
        use_frequency_prior: bool = True,
        use_recency_prior: bool = True,
        initial_alpha: float = 0.5,
        initial_beta: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_songs = num_songs
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.use_frequency_prior = use_frequency_prior
        self.use_recency_prior = use_recency_prior

        # Show encoder (encodes set of songs -> show embedding)
        self.show_encoder = ShowSetEncoder(num_songs, emb_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, emb_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Song prediction head
        self.song_query = nn.Embedding(num_songs + 1, emb_dim)
        self.predictor = nn.Linear(emb_dim, emb_dim)

        # Frequency prior (learnable per-song bias)
        if use_frequency_prior:
            self.alpha = nn.Parameter(torch.tensor(initial_alpha))
            self.register_buffer("freq_bias", torch.zeros(num_songs + 1))

        # Recency prior (learnable weight)
        if use_recency_prior:
            self.beta = nn.Parameter(torch.tensor(initial_beta))

    def encode_show_history(self, prev_shows):
        """Encode sequence of previous shows with transformer.

        Args:
            prev_shows: [batch_size, seq_len, max_songs_per_show]

        Returns:
            history_encoding: [batch_size, emb_dim]

        """
        batch_size, seq_len, max_songs = prev_shows.shape

        # Encode each show in the sequence
        # Reshape to [batch_size * seq_len, max_songs]
        prev_shows_flat = prev_shows.view(batch_size * seq_len, max_songs)
        show_embs = self.show_encoder(
            prev_shows_flat
        )  # [batch_size * seq_len, emb_dim]
        # [batch_size, seq_len, emb_dim]
        show_embs = show_embs.view(batch_size, seq_len, self.emb_dim)

        # Add positional encoding
        show_embs = show_embs + self.pos_encoding[:seq_len]

        # Transformer encoding
        history_encoding = self.transformer(show_embs)  # [batch_size, seq_len, emb_dim]

        # Use last position as summary
        summary = history_encoding[:, -1, :]  # [batch_size, emb_dim]

        return summary

    def forward(
        self,
        prev_shows: torch.Tensor,  # [batch_size, seq_len, max_songs_per_show]
        song_ids: torch.Tensor,  # [batch_size, num_candidates]
        recency_scores: torch.Tensor = None,  # [batch_size, num_candidates]
    ):
        """Forward pass.

        Args:
            prev_shows: Previous show sequences
            song_ids: Candidate songs to score
            recency_scores: Optional recency penalty scores

        Returns:
            probs: [batch_size, num_candidates]

        """
        batch_size = prev_shows.size(0)
        num_candidates = song_ids.size(1)

        # Encode show history
        history = self.encode_show_history(prev_shows)  # [batch_size, emb_dim]

        # Transform history for prediction
        history_proj = self.predictor(history)  # [batch_size, emb_dim]

        # Get song query embeddings
        song_queries = self.song_query(
            song_ids
        )  # [batch_size, num_candidates, emb_dim]

        # Compute base logits (dot product)
        logits = torch.bmm(
            song_queries,  # [batch_size, num_candidates, emb_dim]
            history_proj.unsqueeze(-1),  # [batch_size, emb_dim, 1]
        ).squeeze(
            -1
        )  # [batch_size, num_candidates]

        # Add frequency prior
        if self.use_frequency_prior:
            freq_prior = self.freq_bias[song_ids]  # [batch_size, num_candidates]
            logits = logits + self.alpha * freq_prior

        # Add recency prior
        if self.use_recency_prior and recency_scores is not None:
            logits = logits + self.beta * recency_scores

        return torch.sigmoid(logits)

    def set_frequency_bias(self, freq_scores):
        """Set frequency bias from precomputed scores.

        Args:
            freq_scores: [num_songs] tensor of frequency scores

        """
        if self.use_frequency_prior:
            self.freq_bias[: len(freq_scores)] = freq_scores
