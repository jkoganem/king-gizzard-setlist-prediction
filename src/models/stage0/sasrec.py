"""SASRec (Self-Attentive Sequential Recommendation) for setlist sequence prediction.

This model uses self-attention to predict the next song in a setlist based on
the sequence of songs played so far.

Reference: Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential
Recommendation. In ICDM 2018.
"""

import torch
import torch.nn as nn
import numpy as np


class PointWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # Back to (B, T, D)
        outputs += inputs  # Residual connection
        return outputs


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Predicts next song in a setlist based on the sequence of songs played so far.
    """

    def __init__(
        self,
        num_songs,
        max_seq_len=50,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        device="cpu",
    ):
        """Args:
        num_songs: Total number of unique songs in vocabulary
        max_seq_len: Maximum sequence length to consider
        hidden_units: Dimension of hidden units
        num_blocks: Number of self-attention blocks
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        device: 'cpu' or 'cuda'.

        """
        super(SASRec, self).__init__()

        self.num_songs = num_songs
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units
        self.device = device

        # Embedding layers
        self.item_emb = nn.Embedding(
            num_songs + 1, hidden_units, padding_idx=0
        )  # +1 for padding
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Self-attention blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    hidden_units, num_heads, dropout=dropout_rate, batch_first=True
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self.to(device)

    def log2feats(self, log_seqs):
        """Convert log sequences to feature representations.

        Args:
            log_seqs: (B, T) - batch of sequences (song IDs)

        Returns:
            (B, T, D) - sequence features

        """
        seqs = self.item_emb(log_seqs)  # (B, T, D)
        seqs *= self.hidden_units**0.5  # Scale embeddings

        # Add positional embeddings
        positions = (
            torch.arange(log_seqs.size(1), device=self.device)
            .unsqueeze(0)
            .expand_as(log_seqs)
        )
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        # Create attention mask (causal mask - can't look ahead)
        timeline_mask = log_seqs == 0  # Padding positions
        seqs *= ~timeline_mask.unsqueeze(-1)  # Zero out padding

        # Create causal attention mask
        tl = seqs.size(1)
        attention_mask = torch.triu(
            torch.ones((tl, tl), device=self.device), diagonal=1
        ).bool()

        # Apply self-attention blocks
        for i in range(len(self.attention_layers)):
            # Multi-head self-attention with causal mask
            seqs = self.attention_layernorms[i](seqs)
            Q = seqs
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_outputs

            # Feed-forward
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, log_seqs, pos_items, neg_items=None):
        """Forward pass.

        Args:
            log_seqs: (B, T) - input sequences (song IDs)
            pos_items: (B, T) - positive next items (for training)
            neg_items: (B, T) - negative samples (for training)

        Returns:
            pos_logits: (B, T) - scores for positive items
            neg_logits: (B, T) - scores for negative items (if provided)

        """
        log_feats = self.log2feats(log_seqs)  # (B, T, D)

        pos_embs = self.item_emb(pos_items)  # (B, T, D)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (B, T)

        if neg_items is not None:
            neg_embs = self.item_emb(neg_items)  # (B, T, D)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)  # (B, T)
            return pos_logits, neg_logits
        else:
            return pos_logits

    def predict(self, log_seqs, candidate_items=None):
        """Predict scores for next item.

        Args:
            log_seqs: (B, T) - input sequences
            candidate_items: (B, K) - candidate items to score. If None, score all items.

        Returns:
            scores: (B, K) or (B, num_songs) - prediction scores

        """
        log_feats = self.log2feats(log_seqs)  # (B, T, D)
        final_feat = log_feats[:, -1, :]  # (B, D) - take last position

        if candidate_items is not None:
            candidate_embs = self.item_emb(candidate_items)  # (B, K, D)
            scores = torch.matmul(candidate_embs, final_feat.unsqueeze(-1)).squeeze(
                -1
            )  # (B, K)
        else:
            # Score all items
            item_embs = self.item_emb.weight  # (num_songs+1, D)
            scores = torch.matmul(final_feat, item_embs.t())  # (B, num_songs+1)

        return scores


def bpr_loss(pos_logits, neg_logits):
    """Bayesian Personalized Ranking loss.

    Encourages positive items to be ranked higher than negative items.
    """
    return -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-10).mean()


def train_sasrec_epoch(model, dataloader, optimizer, device):
    """Train SASRec for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        seqs, pos, neg = batch
        seqs = seqs.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        optimizer.zero_grad()
        pos_logits, neg_logits = model(seqs, pos, neg)

        # Only compute loss on non-padding positions
        loss_mask = pos != 0
        pos_logits = pos_logits[loss_mask]
        neg_logits = neg_logits[loss_mask]

        loss = bpr_loss(pos_logits, neg_logits)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_sasrec(model, dataloader, device, k=10):
    """Evaluate SASRec model.

    Returns:
        hit_rate: Hit rate @k
        ndcg: NDCG @k

    """
    model.eval()

    hits = []
    ndcgs = []

    for batch in dataloader:
        seqs, targets = batch  # targets is (B,) - single next item
        seqs = seqs.to(device)
        targets = targets.to(device)

        # Get predictions for all items
        scores = model.predict(seqs)  # (B, num_songs+1)

        # Get top-k predictions
        _, top_k_items = torch.topk(scores, k, dim=-1)  # (B, k)

        # Check if target is in top-k
        for i in range(targets.size(0)):
            target = targets[i].item()
            if target == 0:  # Skip padding
                continue

            top_k = top_k_items[i].cpu().numpy()

            if target in top_k:
                hits.append(1.0)
                # Calculate NDCG
                rank = np.where(top_k == target)[0][0] + 1
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                hits.append(0.0)
                ndcgs.append(0.0)

    hit_rate = np.mean(hits) if hits else 0.0
    ndcg = np.mean(ndcgs) if ndcgs else 0.0

    return hit_rate, ndcg
