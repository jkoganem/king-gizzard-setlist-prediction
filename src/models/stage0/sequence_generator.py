"""Full setlist sequence generation using SASRec with beam search and soft penalties.

This module combines:
1. SASRec for next-song prediction
2. Beam search for sequence generation
3. Soft rule penalties (no-repeat, venue constraints, style balance)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BeamSearchConfig:
    """Configuration for beam search decoding."""

    beam_width: int = 5
    max_length: int = 25

    # Soft penalty weights
    lambda_repeat_last_night: float = 0.8
    lambda_same_venue_repeat: float = 0.5
    lambda_style_runlength: float = 0.3

    # Diversity penalties
    lambda_diversity: float = 0.1

    # Temperature for sampling
    temperature: float = 1.0


@dataclass
class ShowContext:
    """Context information for a show."""

    venue_id: str
    date: str
    tour_name: str
    is_residency: bool = False
    residency_night: int = 0

    # Previous show information
    prev_show_songs: Optional[List[int]] = None
    prev_venue_songs: Optional[List[int]] = None

    # Song metadata (for style penalties)
    song_styles: Optional[Dict[int, str]] = None  # song_idx -> style


class SequenceGenerator:
    """Generate full setlist sequences using SASRec + beam search with soft penalties."""

    def __init__(self, sasrec_model, config: BeamSearchConfig, device="cpu"):
        """Args:
        sasrec_model: Trained SASRec model
        config: Beam search configuration
        device: 'cpu' or 'cuda'.

        """
        self.model = sasrec_model
        self.config = config
        self.device = device
        self.model.eval()

    def generate_sequence(
        self,
        context: ShowContext,
        opener_song: Optional[int] = None,
        candidate_songs: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate a setlist sequence using beam search.

        Args:
            context: Show context information
            opener_song: If provided, use as first song (otherwise predict)
            candidate_songs: List of candidate song indices (if None, use all)

        Returns:
            List of (song_idx, score) tuples representing the predicted setlist

        """
        # Initialize beam with opener
        if opener_song is not None:
            beams = [([opener_song], 0.0)]  # (sequence, cumulative_score)
        else:
            beams = [([], 0.0)]

        # Generate sequence step by step
        for step in range(self.config.max_length):
            new_beams = []

            for sequence, cum_score in beams:
                # Get next song candidates
                next_songs = self._get_next_candidates(
                    sequence, context, candidate_songs
                )

                for song_idx, log_prob in next_songs[: self.config.beam_width * 2]:
                    # Calculate penalties
                    penalty = self._calculate_penalties(sequence, song_idx, context)

                    # Combined score
                    score = log_prob - penalty

                    new_sequence = sequence + [song_idx]
                    new_score = cum_score + score

                    new_beams.append((new_sequence, new_score))

            # Keep top-k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[
                : self.config.beam_width
            ]

            # Early stopping if all beams have reached reasonable length
            min_len = min(len(seq) for seq, _ in beams)
            if min_len >= 15:  # Minimum reasonable setlist length
                # Check if we should stop
                avg_prob_recent = np.mean([score / len(seq) for seq, score in beams])
                if avg_prob_recent < -5.0:  # Very low probability, stop
                    break

        # Return best beam
        best_sequence, best_score = beams[0]

        # Convert to (song, score) format
        result = [
            (song_idx, best_score / len(best_sequence)) for song_idx in best_sequence
        ]

        return result

    def _get_next_candidates(
        self,
        current_sequence: List[int],
        context: ShowContext,
        candidate_songs: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Get next song candidates with probabilities.

        Returns:
            List of (song_idx, log_prob) sorted by probability

        """
        # Prepare input sequence for SASRec
        if len(current_sequence) == 0:
            # No sequence yet, use padding
            input_seq = torch.zeros(
                1, self.model.max_seq_len, dtype=torch.long, device=self.device
            )
        else:
            # Pad/truncate sequence
            if len(current_sequence) > self.model.max_seq_len:
                seq = current_sequence[-self.model.max_seq_len :]
            else:
                seq = [0] * (
                    self.model.max_seq_len - len(current_sequence)
                ) + current_sequence

            input_seq = torch.LongTensor([seq]).to(self.device)

        # Get predictions from SASRec
        with torch.no_grad():
            scores = self.model.predict(
                input_seq, candidate_items=None
            )  # (1, num_songs+1)
            scores = scores[0]  # (num_songs+1,)

            # Apply temperature
            scores = scores / self.config.temperature

            # Convert to log probabilities
            log_probs = torch.log_softmax(scores, dim=0)

        # Filter to candidate songs if provided
        if candidate_songs is not None:
            candidate_indices = torch.LongTensor(candidate_songs).to(self.device)
            log_probs = log_probs[candidate_indices]
            song_indices = candidate_songs
        else:
            song_indices = list(range(1, len(log_probs)))  # Skip padding (0)
            log_probs = log_probs[1:]

        # Sort by probability
        sorted_indices = torch.argsort(log_probs, descending=True)

        candidates = [
            (song_indices[idx], log_probs[idx].item()) for idx in sorted_indices
        ]

        return candidates

    def _calculate_penalties(
        self, sequence: List[int], candidate_song: int, context: ShowContext
    ) -> float:
        """Calculate soft penalties for a candidate song.

        Returns:
            Total penalty (higher = worse)

        """
        total_penalty = 0.0

        # 1. No-repeat from last night
        if context.prev_show_songs is not None:
            if candidate_song in context.prev_show_songs:
                total_penalty += self.config.lambda_repeat_last_night

        # 2. No-repeat from same venue
        if context.prev_venue_songs is not None:
            if candidate_song in context.prev_venue_songs:
                total_penalty += self.config.lambda_same_venue_repeat

        # 3. Style run-length penalty (avoid too many songs of same style in a row)
        if context.song_styles is not None and len(sequence) > 0:
            candidate_style = context.song_styles.get(candidate_song)
            if candidate_style is not None:
                # Check recent songs
                recent_styles = []
                for song in sequence[-3:]:  # Last 3 songs
                    style = context.song_styles.get(song)
                    if style is not None:
                        recent_styles.append(style)

                # Count consecutive occurrences of same style
                if len(recent_styles) > 0 and all(
                    s == candidate_style for s in recent_styles
                ):
                    # All recent songs are same style as candidate
                    run_length = len(recent_styles)
                    total_penalty += self.config.lambda_style_runlength * run_length

        # 4. Diversity penalty (avoid repeating songs in current setlist)
        if candidate_song in sequence:
            # Song already in setlist - strong penalty
            total_penalty += 10.0  # Very high penalty for exact repeats

        # 5. Recency penalty (slight preference for songs not played recently in sequence)
        if len(sequence) >= 5:
            recent_songs = set(sequence[-5:])
            if candidate_song in recent_songs:
                total_penalty += self.config.lambda_diversity

        return total_penalty

    def generate_with_constraints(
        self,
        context: ShowContext,
        opener_song: int,
        closer_song: int,
        target_length: int = 18,
        candidate_songs: Optional[List[int]] = None,
    ) -> List[int]:
        """Generate a setlist with fixed opener and closer.

        Args:
            context: Show context
            opener_song: Fixed opener
            closer_song: Fixed closer
            target_length: Target setlist length
            candidate_songs: Candidate songs to choose from

        Returns:
            Full setlist sequence [opener, ..., closer]

        """
        # Generate middle section
        middle_length = target_length - 2  # Exclude opener and closer

        # Start with opener
        sequence = [opener_song]

        # Generate middle songs
        for step in range(middle_length):
            # Get next candidates
            candidates = self._get_next_candidates(sequence, context, candidate_songs)

            # Find best candidate that isn't the closer (save for end)
            for song_idx, log_prob in candidates:
                if song_idx != closer_song and song_idx not in sequence:
                    # Calculate penalty
                    penalty = self._calculate_penalties(sequence, song_idx, context)

                    # Accept if reasonable
                    if log_prob - penalty > -10.0:  # Threshold
                        sequence.append(song_idx)
                        break

            # Fallback: just take first available
            if len(sequence) == step + 1:  # Didn't add anything
                for song_idx, _ in candidates:
                    if song_idx != closer_song and song_idx not in sequence:
                        sequence.append(song_idx)
                        break

        # Add closer
        sequence.append(closer_song)

        return sequence


def generate_setlist_with_model(
    sasrec_model,
    opener_model,
    closer_model,
    xgb_set_model,
    context: ShowContext,
    feature_dict: Dict,
    device="cpu",
) -> Dict:
    """Full pipeline: Predict setlist using all models.

    1. XGBoost predicts which songs (set)
    2. Opener/Closer models predict edges
    3. SASRec generates order with beam search

    Args:
        sasrec_model: Trained SASRec
        opener_model: Trained opener predictor
        closer_model: Trained closer predictor
        xgb_set_model: Trained XGBoost set model
        context: Show context
        feature_dict: Features for opener/closer/set models
        device: Device

    Returns:
        Dict with predictions:
        {
            'candidate_songs': List[int],  # From XGBoost
            'opener': int,
            'closer': int,
            'sequence': List[int],  # Full ordered setlist
            'scores': Dict  # Various model scores
        }

    """
    # 1. Get candidate songs from XGBoost (top-K by probability)
    xgb_probs = xgb_set_model.predict_proba(feature_dict["set_features"])[:, 1]
    top_k = 25  # Get more than needed
    top_indices = np.argsort(xgb_probs)[-top_k:]
    candidate_songs = top_indices.tolist()

    # 2. Predict opener
    opener_probs = opener_model.predict_proba(feature_dict["opener_features"])[:, 1]
    opener_idx = np.argmax(opener_probs)
    opener_song = (
        candidate_songs[opener_idx]
        if opener_idx < len(candidate_songs)
        else candidate_songs[0]
    )

    # 3. Predict closer
    closer_probs = closer_model.predict_proba(feature_dict["closer_features"])[:, 1]
    closer_idx = np.argmax(closer_probs)
    closer_song = (
        candidate_songs[closer_idx]
        if closer_idx < len(candidate_songs)
        else candidate_songs[-1]
    )

    # 4. Generate sequence with SASRec
    config = BeamSearchConfig(
        beam_width=5,
        max_length=20,
        lambda_repeat_last_night=0.8,
        lambda_same_venue_repeat=0.5,
        lambda_style_runlength=0.3,
    )

    generator = SequenceGenerator(sasrec_model, config, device)

    # Generate with constraints
    sequence = generator.generate_with_constraints(
        context=context,
        opener_song=opener_song,
        closer_song=closer_song,
        target_length=18,
        candidate_songs=candidate_songs,
    )

    return {
        "candidate_songs": candidate_songs,
        "opener": opener_song,
        "closer": closer_song,
        "sequence": sequence,
        "scores": {
            "xgb_probs": xgb_probs.tolist(),
            "opener_probs": opener_probs.tolist(),
            "closer_probs": closer_probs.tolist(),
        },
    }
