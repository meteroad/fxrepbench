from __future__ import annotations

import unittest

from fxrepbench.head_proposals import parameter_order


class ProposalSchemaTests(unittest.TestCase):
    def test_parameter_center_has_22_dimensions(self) -> None:
        self.assertEqual(len(parameter_order()), 22)

    def test_cmaes_layouts_match_paper_encoding(self) -> None:
        from fxrepbench.cmaes import make_layout

        known = make_layout("known", ["compressor", "reverb"])
        hidden = make_layout("hidden")
        self.assertEqual(known.dimension, 8)
        self.assertEqual(hidden.dimension, 38)


try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


@unittest.skipIf(np is None, "reference dependencies not installed")
class HeadProposalTests(unittest.TestCase):
    def test_hidden_random_draw_is_frozen(self) -> None:
        from fxrepbench.candidates import sample_hidden_candidate, stable_seed
        from fxrepbench.config import chain_config, effects_config

        seed = stable_seed(20270724, "fma_val_125159", "hidden")
        candidate = sample_hidden_candidate(
            effects_config(),
            chain_config(),
            np.random.default_rng(seed),
        )
        self.assertEqual(candidate.chain_order, ("peak_filter", "distortion"))
        self.assertAlmostEqual(
            candidate.physical_params["distortion"]["drive_db"],
            8.332989960919537,
        )

    def test_known_pool_starts_at_unperturbed_center(self) -> None:
        from fxrepbench.head_proposals import sample_head_candidates

        center = np.full(22, 0.5, dtype=np.float32)
        candidates = sample_head_candidates(
            2,
            center,
            topology="known",
            target_chain=["compressor", "reverb"],
            rng=np.random.default_rng(4),
        )
        self.assertEqual(candidates[0].chain_order, ("compressor", "reverb"))
        self.assertNotEqual(candidates[0].physical_params, candidates[1].physical_params)


@unittest.skipIf(torch is None, "reference dependencies not installed")
class RegistrationHeadTests(unittest.TestCase):
    def test_output_shapes(self) -> None:
        from fxrepbench.registration import build_proposal_head

        model = build_proposal_head(2048)
        usage, params, order = model(torch.zeros(3, 2048))
        self.assertEqual(tuple(usage.shape), (3, 8))
        self.assertEqual(tuple(params.shape), (3, 22))
        self.assertEqual(tuple(order.shape), (3, 4, 9))
