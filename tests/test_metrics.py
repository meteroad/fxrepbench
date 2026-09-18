from __future__ import annotations

import unittest

from fxrepbench.metrics import fx_set_f1


class FxSetF1Tests(unittest.TestCase):
    def test_ignores_order(self) -> None:
        self.assertEqual(
            fx_set_f1(["reverb", "compressor"], ["compressor", "reverb"]),
            1.0,
        )

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(
            fx_set_f1(["reverb", "delay"], ["reverb", "compressor"]),
            0.5,
        )

    def test_empty_prediction(self) -> None:
        self.assertEqual(fx_set_f1([], ["reverb"]), 0.0)


try:
    import auraloss
    import torch
except ImportError:
    auraloss = None
    torch = None


@unittest.skipIf(torch is None or auraloss is None, "reference dependencies not installed")
class DistanceParityTests(unittest.TestCase):
    def test_matches_auraloss_default_reduction(self) -> None:
        from fxrepbench.metrics import MultiResolutionStftDistance

        generator = torch.Generator().manual_seed(7)
        target = torch.randn(1, 2, 8192, generator=generator) * 0.05
        candidates = target + torch.randn(3, 2, 8192, generator=generator) * 0.01
        ours = MultiResolutionStftDistance()(candidates, target, batch_size=2)
        reference = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512],
            hop_sizes=[120, 240, 50],
            win_lengths=[600, 1200, 240],
        )
        expected = torch.stack(
            [reference(candidate.unsqueeze(0), target) for candidate in candidates]
        ).reshape(-1)
        self.assertTrue(torch.allclose(ours, expected, atol=1e-6, rtol=1e-5))
