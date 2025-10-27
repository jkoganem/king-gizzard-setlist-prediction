"""Temporal Sets GNN for setlist prediction.

Based on "Predicting Temporal Sets with Deep Neural Networks" (KDD 2020)
Simplified implementation focused on key components:
1. Dynamic co-occurrence graph
2. GNN to learn song embeddings
3. Temporal attention over previous shows
4. Set-level prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Simple Graph Convolutional Layer."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node_features, adj_matrix):
        """Args:
            node_features: [num_nodes, in_dim]
            adj_matrix: [num_nodes, num_nodes] (adjacency matrix).

        Returns:
            [num_nodes, out_dim]

        """
        # Aggregate neighbor features
        aggregated = torch.matmul(adj_matrix, node_features)

        # Apply linear transformation
        return F.relu(self.linear(aggregated))


class TemporalAttention(nn.Module):
    """Attention over sequence of previous shows."""

    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

    def forward(self, context_emb, show_embs):
        """Args:
            context_emb: [batch, emb_dim] - current show context
            show_embs: [batch, num_prev_shows, emb_dim] - previous shows.

        Returns:
            [batch, emb_dim] - attended context

        """
        # Compute attention scores
        Q = self.query(context_emb).unsqueeze(1)  # [batch, 1, emb_dim]
        K = self.key(show_embs)  # [batch, num_prev_shows, emb_dim]
        V = self.value(show_embs)  # [batch, num_prev_shows, emb_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)  # [batch, 1, num_prev_shows]

        # Weighted sum
        attended = torch.matmul(attention_weights, V).squeeze(1)  # [batch, emb_dim]

        return attended, attention_weights.squeeze(1)


class TemporalSetsGNN(nn.Module):
    """Temporal Sets model for setlist prediction.

    Simplified version that focuses on core innovations:
    - Dynamic co-occurrence graphs
    - GNN-based song embeddings
    - Temporal attention over previous shows
    """

    def __init__(self, num_songs, emb_dim=64, gnn_layers=2, num_prev_shows=5):
        super().__init__()

        self.num_songs = num_songs
        self.emb_dim = emb_dim
        self.num_prev_shows = num_prev_shows

        # Song embedding layer
        self.song_embedding = nn.Embedding(num_songs + 1, emb_dim, padding_idx=0)

        # GNN layers for co-occurrence graph
        self.gnn_layers = nn.ModuleList(
            [GraphConvLayer(emb_dim, emb_dim) for _ in range(gnn_layers)]
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(emb_dim)

        # Context encoding (venue, tour, country, is_festival, is_marathon)
        self.venue_embedding = nn.Embedding(200, emb_dim // 4)  # Max 200 venues
        self.tour_embedding = nn.Embedding(20, emb_dim // 4)  # Max 20 tours
        self.country_embedding = nn.Embedding(50, emb_dim // 8)  # Max 50 countries
        self.festival_embedding = nn.Embedding(2, emb_dim // 8)  # Boolean: 0 or 1
        self.marathon_embedding = nn.Embedding(2, emb_dim // 8)  # Boolean: 0 or 1

        # Final prediction layers
        # Context size: emb_dim//4 + emb_dim//4 + emb_dim//8 + emb_dim//8 + emb_dim//8
        #             = emb_dim//2 + emb_dim//8 + emb_dim//8 + emb_dim//8
        #             = emb_dim//2 + 3*emb_dim//8 = 7*emb_dim//8
        context_size = emb_dim // 2 + 3 * (emb_dim // 8)
        self.context_proj = nn.Linear(context_size, emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),  # song_emb + context + temporal
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, 1),
        )

    def build_cooccurrence_graph(self, prev_setlists, num_songs):
        """Build co-occurrence adjacency matrix from previous setlists.

        Args:
            prev_setlists: [batch, num_prev_shows, max_songs_per_show]
            num_songs: total number of songs

        Returns:
            adj_matrix: [batch, num_songs, num_songs]

        """
        batch_size = prev_setlists.size(0)
        adj = torch.zeros(
            batch_size, num_songs + 1, num_songs + 1, device=prev_setlists.device
        )

        for b in range(batch_size):
            for show_idx in range(prev_setlists.size(1)):
                # Get songs in this show (non-zero entries)
                songs = prev_setlists[b, show_idx]
                songs = songs[songs > 0]

                # Add edges between all pairs
                for i in range(len(songs)):
                    for j in range(len(songs)):
                        if i != j:
                            adj[b, songs[i], songs[j]] += 1

        # Normalize by degree (row normalization)
        row_sum = adj.sum(dim=2, keepdim=True) + 1e-10
        adj = adj / row_sum

        return adj

    def get_show_embeddings(self, prev_setlists, song_embs):
        """Get embeddings for each previous show.

        Args:
            prev_setlists: [batch, num_prev_shows, max_songs_per_show]
            song_embs: [batch, num_songs, emb_dim]

        Returns:
            [batch, num_prev_shows, emb_dim]

        """
        batch_size, num_shows, max_songs = prev_setlists.shape

        show_embs = []
        for show_idx in range(num_shows):
            songs = prev_setlists[:, show_idx, :]  # [batch, max_songs]

            # Get embeddings for songs in this show
            song_vectors = []
            for b in range(batch_size):
                show_songs = songs[b][songs[b] > 0]
                if len(show_songs) > 0:
                    show_song_embs = song_embs[
                        b, show_songs
                    ]  # [num_songs_in_show, emb_dim]
                    show_emb = show_song_embs.mean(dim=0)  # Average pooling
                else:
                    show_emb = torch.zeros(self.emb_dim, device=song_embs.device)
                song_vectors.append(show_emb)

            show_embs.append(torch.stack(song_vectors))  # [batch, emb_dim]

        return torch.stack(show_embs, dim=1)  # [batch, num_shows, emb_dim]

    def forward(
        self,
        song_ids,
        prev_setlists,
        venue_ids,
        tour_ids,
        country_ids,
        is_festival,
        is_marathon,
    ):
        """Forward pass.

        Args:
            song_ids: [batch] - candidate songs to predict
            prev_setlists: [batch, num_prev_shows, max_songs_per_show] - previous shows
            venue_ids: [batch] - venue for current show
            tour_ids: [batch] - tour for current show
            country_ids: [batch] - country for current show
            is_festival: [batch] - festival flag (0 or 1)
            is_marathon: [batch] - marathon flag (0 or 1)

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
        # [batch, num_songs+1, emb_dim]
        song_embs = song_embs.unsqueeze(0).expand(batch_size, -1, -1)

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

        return torch.sigmoid(logits)
