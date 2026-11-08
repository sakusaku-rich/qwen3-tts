# coding=utf-8
# Copyright 2026 sakusaku-rich
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check the guard, the cache behaviour and the precomputed values of graph_predictor.

The guard tests run without a GPU. The cache tests and the replay test run only
when CUDA is available. The precompute tests build a tiny real predictor and run
on any device, so they also run on CPU.
"""

import unittest
from types import SimpleNamespace

import torch

from qwen_tts.core.models import graph_predictor


class FakeConfig:
    """The attributes `is_supported` and `_StaticCache` read."""

    num_code_groups = 16
    num_hidden_layers = 2
    hidden_size = 8

    def get_text_config(self, decoder=False):
        """StaticCache asks the config for its text sub-config."""
        return self


class FakePredictor:
    """A stand-in for the code predictor that only carries a config."""

    def __init__(self):
        self.config = FakeConfig()


def make_kwargs(**overrides):
    """Build the kwargs the caller passes to the predictor's generate."""
    kwargs = {
        "inputs_embeds": torch.zeros(1, 2, 2048, dtype=torch.bfloat16),
        "max_new_tokens": 15,
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.9,
    }
    kwargs.update(overrides)
    return kwargs


def make_tiny_predictor(talker_hidden=32, hidden=16, vocab=24, num_code_groups=6,
                        layers=2, device="cpu", seed=0):
    """Build a small real code predictor that runs on the given device."""
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    )

    torch.manual_seed(seed)
    config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        num_code_groups=num_code_groups,
    )
    talker_config = SimpleNamespace(hidden_size=talker_hidden)
    predictor = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        config=config, talker_config=talker_config
    )
    return predictor.to(device=device, dtype=torch.bfloat16).eval()


def make_graph(predictor, talker_hidden, device):
    """Build a GraphCodePredictor for the tiny predictor."""
    embeds = torch.randn(1, graph_predictor.PREFILL_TOKENS, talker_hidden,
                         device=device, dtype=torch.bfloat16)
    kwargs = {
        "inputs_embeds": embeds,
        "max_new_tokens": predictor.config.num_code_groups - 1,
        "do_sample": True,
        "top_k": 5,
        "temperature": 0.9,
    }
    return graph_predictor.GraphCodePredictor(predictor, kwargs), kwargs


class IsSupportedTest(unittest.TestCase):
    def test_rejects_missing_inputs_embeds(self):
        """A call without inputs_embeds takes the eager path."""
        # Arrange
        predictor = FakePredictor()
        kwargs = make_kwargs(inputs_embeds=None)

        # Act
        supported = graph_predictor.is_supported(predictor, kwargs)

        # Assert
        self.assertFalse(supported)

    def test_rejects_cpu_inputs_embeds(self):
        """A CPU input takes the eager path."""
        # Arrange
        predictor = FakePredictor()
        kwargs = make_kwargs()

        # Act
        supported = graph_predictor.is_supported(predictor, kwargs)

        # Assert
        self.assertFalse(supported)

    def test_rejects_non_bf16_inputs_embeds(self):
        """A non-BF16 input takes the eager path."""
        # Arrange
        predictor = FakePredictor()
        kwargs = make_kwargs(inputs_embeds=torch.zeros(1, 2, 2048, dtype=torch.float32))

        # Act
        supported = graph_predictor.is_supported(predictor, kwargs)

        # Assert
        self.assertFalse(supported)

    def test_rejects_unexpected_prefill_length(self):
        """A prefill that is not two tokens takes the eager path."""
        # Arrange
        predictor = FakePredictor()
        kwargs = make_kwargs(inputs_embeds=torch.zeros(1, 3, 2048, dtype=torch.bfloat16))

        # Act
        supported = graph_predictor.is_supported(predictor, kwargs)

        # Assert
        self.assertFalse(supported)

    def test_rejects_unexpected_max_new_tokens(self):
        """A step count other than the configured one takes the eager path."""
        # Arrange
        predictor = FakePredictor()
        kwargs = make_kwargs(max_new_tokens=8)

        # Act
        supported = graph_predictor.is_supported(predictor, kwargs)

        # Assert
        self.assertFalse(supported)

    def test_rejects_non_sampling(self):
        """Greedy decoding takes the eager path."""
        # Arrange
        predictor = FakePredictor()
        kwargs = make_kwargs(do_sample=False)

        # Act
        supported = graph_predictor.is_supported(predictor, kwargs)

        # Assert
        self.assertFalse(supported)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_accepts_the_expected_frame(self):
        """A BF16 CUDA prefill of two tokens with the configured step count is accepted."""
        # Arrange
        predictor = FakePredictor()
        kwargs = make_kwargs(inputs_embeds=torch.zeros(1, 2, 2048, device="cuda",
                                                       dtype=torch.bfloat16))

        # Act
        supported = graph_predictor.is_supported(predictor, kwargs)

        # Assert
        self.assertTrue(supported)


class StaticCacheTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_appends_instead_of_overwriting(self):
        """Each update extends the cache, so a decode step does not overwrite the prefill."""
        # Arrange
        cache = graph_predictor._StaticCache(config=FakeConfig(), max_cache_len=17,
                                             device=torch.device("cuda"), dtype=torch.bfloat16)
        cache.reset()
        keys = torch.zeros(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
        values = torch.zeros(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)

        # Act
        first_keys, _ = cache.layers[0].update(keys, values, {"cache_position": torch.tensor([2])})
        step_keys = torch.ones(1, 4, 1, 8, device="cuda", dtype=torch.bfloat16)
        step_values = torch.ones(1, 4, 1, 8, device="cuda", dtype=torch.bfloat16)
        second_keys, _ = cache.layers[0].update(step_keys, step_values,
                                                {"cache_position": torch.tensor([1])})

        # Assert
        self.assertEqual(first_keys.shape[2], 2)
        self.assertEqual(second_keys.shape[2], 3)
        # The prefill slot is untouched and the new token is appended
        self.assertTrue(torch.equal(second_keys[:, :, :2], keys))
        self.assertTrue(torch.equal(second_keys[:, :, 2:], step_keys))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_reports_filled_length_before_first_update(self):
        """The mask size is the query length before any update, like DynamicCache."""
        # Arrange
        cache = graph_predictor._StaticCache(config=FakeConfig(), max_cache_len=17,
                                             device=torch.device("cuda"), dtype=torch.bfloat16)
        cache.reset()

        # Act
        length, offset = cache.get_mask_sizes(torch.tensor([0, 1], device="cuda"), 0)

        # Assert
        self.assertEqual(length, 2)
        self.assertEqual(offset, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_reset_clears_the_filled_length(self):
        """reset() lets the cache be reused for the next frame."""
        # Arrange
        cache = graph_predictor._StaticCache(config=FakeConfig(), max_cache_len=17,
                                             device=torch.device("cuda"), dtype=torch.bfloat16)
        cache.reset()
        keys = torch.zeros(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
        cache.layers[0].update(keys, keys, {"cache_position": torch.tensor([2])})

        # Act
        cache.reset()

        # Assert
        self.assertEqual(cache.layers[0]._filled, 0)


class PrecomputedPositionEmbeddingsTest(unittest.TestCase):
    """The rotary tables must equal what rotary_emb returns on each forward."""

    def test_prefill_table_matches_a_forward(self):
        """The prefill table is what rotary_emb returns for the prefill positions."""
        # Arrange
        predictor = make_tiny_predictor()
        graph, _ = make_graph(predictor, talker_hidden=32, device="cpu")
        rotary = predictor.model.rotary_emb
        hidden = torch.empty(1, dtype=torch.bfloat16)

        # Act
        expected = rotary(hidden, graph.prefill_position_ids)
        actual = graph.prefill_position_embeddings

        # Assert
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_step_tables_match_each_forward(self):
        """Every step table is what rotary_emb returns for that step's position."""
        # Arrange
        predictor = make_tiny_predictor()
        graph, _ = make_graph(predictor, talker_hidden=32, device="cpu")
        rotary = predictor.model.rotary_emb
        hidden = torch.empty(1, dtype=torch.bfloat16)

        # Act / Assert
        self.assertEqual(len(graph.step_position_embeddings), graph.num_steps - 1)
        for step in range(1, graph.num_steps):
            expected = rotary(hidden, graph.step_position_ids[step - 1])
            actual = graph.step_position_embeddings[step - 1]
            self.assertTrue(torch.equal(actual[0], expected[0]), f"cos mismatch at step {step}")
            self.assertTrue(torch.equal(actual[1], expected[1]), f"sin mismatch at step {step}")

    def test_forward_accepts_precomputed_tables(self):
        """A forward with the precomputed tables equals a forward without them."""
        # Arrange
        predictor = make_tiny_predictor()
        graph, _ = make_graph(predictor, talker_hidden=32, device="cpu")
        embeds = torch.randn(1, 2, 16, dtype=torch.bfloat16)
        position_ids = torch.tensor([[0, 1]])

        # Act
        without = predictor.model(
            inputs_embeds=embeds, position_ids=position_ids, use_cache=False)
        with_tables = predictor.model(
            inputs_embeds=embeds, position_ids=position_ids, use_cache=False,
            position_embeddings=graph.prefill_position_embeddings)

        # Assert
        self.assertTrue(torch.equal(with_tables.last_hidden_state,
                                   without.last_hidden_state))


class ProjectedEmbeddingsTest(unittest.TestCase):
    """The projected tables must equal the per-code lookup and projection."""

    def test_lookup_matches_embed_and_projection(self):
        """Every code of every step embeds and projects to the same values."""
        # Arrange
        predictor = make_tiny_predictor()
        graph, _ = make_graph(predictor, talker_hidden=32, device="cpu")
        embeddings = predictor.model.get_input_embeddings()
        vocab = predictor.config.vocab_size

        # Act / Assert
        self.assertIsNotNone(graph.projected_embeddings)
        for step in range(1, graph.num_steps):
            for code in range(vocab):
                code_tensor = torch.tensor([code])
                expected = predictor.small_to_mtp_projection(
                    embeddings[step - 1](code_tensor.view(1, 1)))
                actual = graph._embed(code_tensor, step)
                self.assertTrue(torch.equal(actual, expected),
                                f"mismatch at step={step} code={code}")

    def test_identity_projection_skips_precompute(self):
        """Without a projection there is nothing to precompute."""
        # Arrange
        predictor = make_tiny_predictor(talker_hidden=16, hidden=16)

        # Act
        graph, _ = make_graph(predictor, talker_hidden=16, device="cpu")

        # Assert
        self.assertIsNone(graph.projected_embeddings)

    def test_identity_lookup_matches_embed(self):
        """With the identity projection the lookup still equals the eager embed."""
        # Arrange
        predictor = make_tiny_predictor(talker_hidden=16, hidden=16)
        graph, _ = make_graph(predictor, talker_hidden=16, device="cpu")
        embeddings = predictor.model.get_input_embeddings()
        vocab = predictor.config.vocab_size

        # Act / Assert
        for step in range(1, graph.num_steps):
            for code in range(vocab):
                code_tensor = torch.tensor([code])
                expected = embeddings[step - 1](code_tensor.view(1, 1))
                actual = graph._embed(code_tensor, step)
                self.assertTrue(torch.equal(actual, expected),
                                f"mismatch at step={step} code={code}")


class GraphReplayTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_replay_matches_eager_generate_bitwise(self):
        """The captured graph returns the sequences the eager path returns.

        `predictor.generate` takes the graph path once the graph exists, so the
        eager reference is produced with `graph_predictor.is_supported` patched
        to False. Otherwise both sides would run the graph and the comparison
        would only check that two graph runs agree.
        """
        # Arrange
        device = "cuda"
        predictor = make_tiny_predictor(device=device)
        graph, kwargs = make_graph(predictor, talker_hidden=32, device=device)
        graph.capture(kwargs)

        # Act
        original = graph_predictor.is_supported
        graph_predictor.is_supported = lambda *args, **call_kwargs: False
        try:
            torch.manual_seed(1234)
            eager = predictor.generate(**kwargs, output_hidden_states=True,
                                       return_dict_in_generate=True)
            eager_rng = torch.cuda.get_rng_state().clone()
        finally:
            graph_predictor.is_supported = original
        torch.manual_seed(1234)
        replayed = graph.replay(kwargs)

        # Assert
        self.assertTrue(torch.equal(eager.sequences, replayed))
        # The graph consumes the RNG in the same order as the eager path
        self.assertTrue(torch.equal(eager_rng, torch.cuda.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_replay_is_deterministic_for_a_seed(self):
        """The same seed gives the same codes on every replay."""
        # Arrange
        device = "cuda"
        predictor = make_tiny_predictor(device=device)
        graph, kwargs = make_graph(predictor, talker_hidden=32, device=device)
        graph.capture(kwargs)

        # Act
        torch.manual_seed(7)
        first = graph.replay(kwargs)
        torch.manual_seed(7)
        second = graph.replay(kwargs)

        # Assert
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
