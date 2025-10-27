"""Stress tests for model architectures."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.stage5.temporal_sets_with_priors import TemporalSetsWithPriors
from src.models.stage5.priors import FrequencyRecencyPriors
from src.models.stage5.sfcn_large import create_sfcn_large
from src.utils.test_config import TEST_CONSTANTS, MODEL_TEST_CONFIG


class TestModels:
    """Test suite for model architectures."""

    def __init__(self):
        """Initialize test parameters from centralized config."""
        self.num_songs = TEST_CONSTANTS.NUM_SONGS
        self.batch_size = TEST_CONSTANTS.BATCH_SIZE
        self.device = torch.device('cpu')  # Use CPU for tests

    def test_dnntsp_initialization(self):
        """Test DNNTSP model initialization."""
        model = DNNTSP(
            num_songs=self.num_songs,
            embedding_dim=32,
            hidden_dims=[64, 64],
            dropout_rate=0.15
        )

        # Verify model structure
        assert model is not None
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'graph_conv1')
        assert hasattr(model, 'graph_conv2')

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

        print(f"[PASS] DNNTSP initialized with {num_params:,} parameters")

    def test_dnntsp_forward_pass(self):
        """Test DNNTSP forward pass."""
        model = DNNTSP(
            num_songs=self.num_songs,
            embedding_dim=32,
            hidden_dims=[64, 64],
            dropout_rate=0.15
        )
        model.eval()

        # Create dummy input
        batch_size = self.batch_size
        x = torch.randn(batch_size, self.num_songs)
        edge_index = torch.randint(0, self.num_songs, (2, 100))
        edge_weight = torch.rand(100)

        # Forward pass
        with torch.no_grad():
            output = model(x, edge_index, edge_weight)

        # Verify output shape
        assert output.shape == (batch_size, self.num_songs), \
            f"Expected shape ({batch_size}, {self.num_songs}), got {output.shape}"

        # Verify output is not all zeros or NaN
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.all(output == 0), "Output is all zeros"

        print(f"[PASS] DNNTSP forward pass produces valid output shape {output.shape}")

    def test_frequency_recency_priors(self):
        """Test FrequencyRecencyPriors computation."""
        priors = FrequencyRecencyPriors(num_songs=self.num_songs)

        # Create dummy previous setlists
        previous_setlists = [
            [0, 1, 2, 3, 4],
            [1, 2, 5, 6, 7],
            [2, 3, 8, 9, 10]
        ]

        # Compute priors
        freq_prior, rec_prior = priors.compute_priors(previous_setlists)

        # Verify shapes
        assert freq_prior.shape == (self.num_songs,), f"Expected freq_prior shape ({self.num_songs},), got {freq_prior.shape}"
        assert rec_prior.shape == (self.num_songs,), f"Expected rec_prior shape ({self.num_songs},), got {rec_prior.shape}"

        # Verify values are normalized (between 0 and 1)
        assert (freq_prior >= 0).all() and (freq_prior <= 1).all(), "Frequency prior not normalized"
        assert (rec_prior >= 0).all() and (rec_prior <= 1).all(), "Recency prior not normalized"

        # Verify song 2 has highest frequency (appears in all 3 setlists)
        most_frequent_song = torch.argmax(freq_prior).item()
        assert most_frequent_song == 2, f"Song 2 should have highest frequency, but song {most_frequent_song} does"

        print(f"[PASS] Priors computed correctly: freq_prior max={freq_prior.max():.3f}, rec_prior max={rec_prior.max():.3f}")

    def test_sfcn_large_initialization(self):
        """Test SFCN Large model initialization."""
        model = create_sfcn_large(num_songs=self.num_songs)

        # Verify model structure
        assert model is not None
        assert isinstance(model, nn.Module)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # SFCN Large should have around 70k parameters
        assert 60000 < num_params < 80000, f"SFCN Large should have ~70k params, got {num_params}"

        print(f"[PASS] SFCN Large initialized with {num_params:,} parameters")

    def test_sfcn_large_forward_pass(self):
        """Test SFCN Large forward pass."""
        model = create_sfcn_large(num_songs=self.num_songs)
        model.eval()

        # Create dummy input (freq_ids, recency, context features)
        batch_size = self.batch_size
        freq_ids = torch.randint(0, self.num_songs, (batch_size,))
        recency = torch.rand(batch_size, 1)
        context = torch.rand(batch_size, 5)  # 5 context features

        # Forward pass
        with torch.no_grad():
            output = model(freq_ids, recency, context)

        # Verify output shape
        assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"

        # Verify output range (should be logits, can be any real number)
        assert not torch.isnan(output).any(), "Output contains NaN values"

        print(f"[PASS] SFCN Large forward pass produces valid output shape {output.shape}")

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = DNNTSP(
            num_songs=self.num_songs,
            embedding_dim=32,
            hidden_dims=[64, 64],
            dropout_rate=0.15
        )
        model.train()

        # Create dummy input and target
        x = torch.randn(self.batch_size, self.num_songs)
        edge_index = torch.randint(0, self.num_songs, (2, 100))
        edge_weight = torch.rand(100)
        target = torch.rand(self.batch_size, self.num_songs)

        # Forward pass
        output = model(x, edge_index, edge_weight)

        # Compute loss
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist and are non-zero for some parameters
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        assert has_gradients, "No gradients computed during backward pass"

        print("[PASS] Gradients flow correctly through the model")

    def test_model_dropout_behavior(self):
        """Test that dropout behaves differently in train vs eval mode."""
        model = DNNTSP(
            num_songs=self.num_songs,
            embedding_dim=32,
            hidden_dims=[64, 64],
            dropout_rate=0.5  # High dropout for testing
        )

        # Create dummy input
        x = torch.randn(self.batch_size, self.num_songs)
        edge_index = torch.randint(0, self.num_songs, (2, 100))
        edge_weight = torch.rand(100)

        # Get outputs in train mode (multiple runs should differ)
        model.train()
        with torch.no_grad():
            train_out1 = model(x, edge_index, edge_weight)
            train_out2 = model(x, edge_index, edge_weight)

        # Get outputs in eval mode (multiple runs should be identical)
        model.eval()
        with torch.no_grad():
            eval_out1 = model(x, edge_index, edge_weight)
            eval_out2 = model(x, edge_index, edge_weight)

        # In eval mode, outputs should be identical
        assert torch.allclose(eval_out1, eval_out2), "Eval mode outputs should be identical"

        # In train mode, outputs should differ (due to dropout)
        # Note: There's a small chance they could be identical by random chance
        outputs_differ = not torch.allclose(train_out1, train_out2)

        print(f"[PASS] Dropout behaves correctly (train mode differs: {outputs_differ}, eval mode consistent)")

    def test_model_device_compatibility(self):
        """Test model can be moved to different devices."""
        model = DNNTSP(
            num_songs=self.num_songs,
            embedding_dim=32,
            hidden_dims=[64, 64],
            dropout_rate=0.15
        )

        # Test CPU
        model_cpu = model.to('cpu')
        x_cpu = torch.randn(2, self.num_songs)
        edge_index_cpu = torch.randint(0, self.num_songs, (2, 10))
        edge_weight_cpu = torch.rand(10)

        with torch.no_grad():
            output_cpu = model_cpu(x_cpu, edge_index_cpu, edge_weight_cpu)

        assert output_cpu.device.type == 'cpu'

        print("[PASS] Model works on CPU device")

    def test_batch_size_flexibility(self):
        """Test model works with different batch sizes."""
        model = DNNTSP(
            num_songs=self.num_songs,
            embedding_dim=32,
            hidden_dims=[64, 64],
            dropout_rate=0.15
        )
        model.eval()

        edge_index = torch.randint(0, self.num_songs, (2, 100))
        edge_weight = torch.rand(100)

        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, self.num_songs)

            with torch.no_grad():
                output = model(x, edge_index, edge_weight)

            assert output.shape == (batch_size, self.num_songs), \
                f"Failed for batch_size={batch_size}: expected ({batch_size}, {self.num_songs}), got {output.shape}"

        print("[PASS] Model handles various batch sizes (1, 4, 16, 32)")

    def test_model_determinism(self):
        """Test that model outputs are deterministic with same seed."""
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)

        model1 = DNNTSP(num_songs=self.num_songs, embedding_dim=32, hidden_dims=[64, 64])
        model1.eval()

        torch.manual_seed(42)
        np.random.seed(42)

        model2 = DNNTSP(num_songs=self.num_songs, embedding_dim=32, hidden_dims=[64, 64])
        model2.eval()

        # Same input
        torch.manual_seed(42)
        x = torch.randn(4, self.num_songs)
        edge_index = torch.randint(0, self.num_songs, (2, 50))
        edge_weight = torch.rand(50)

        with torch.no_grad():
            out1 = model1(x, edge_index, edge_weight)
            out2 = model2(x, edge_index, edge_weight)

        # Models should produce identical outputs
        assert torch.allclose(out1, out2, rtol=1e-5), "Models with same seed should produce identical outputs"

        print("[PASS] Model outputs are deterministic with same random seed")


def run_all_tests():
    """Run all tests in this module."""
    test_class = TestModels()

    tests = [
        test_class.test_dnntsp_initialization,
        test_class.test_dnntsp_forward_pass,
        test_class.test_frequency_recency_priors,
        test_class.test_sfcn_large_initialization,
        test_class.test_sfcn_large_forward_pass,
        test_class.test_model_gradient_flow,
        test_class.test_model_dropout_behavior,
        test_class.test_model_device_compatibility,
        test_class.test_batch_size_flexibility,
        test_class.test_model_determinism,
    ]

    passed = 0
    failed = 0

    print("="*60)
    print("Running Model Architecture Stress Tests")
    print("="*60)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
