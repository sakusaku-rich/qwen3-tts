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
"""Check the talker CUDA Graph path.

The guard tests run without a GPU. The numerical tests run only when CUDA is
available and compare the graph path with the eager path after both graphs are
captured, because capturing a graph consumes the RNG.
"""

import unittest

import torch

from qwen_tts.core.models import graph_talker


class FakeConfig:
    """The attributes `is_supported` and `_GraphStaticCache` read."""

    num_code_groups = 16
    num_hidden_layers = 2
    hidden_size = 8

    def get_text_config(self, decoder=False):
        """StaticCache asks the config for its text sub-config."""
        return self


class FakeTalker:
    def __init__(self):
        self.config = FakeConfig()


def make_kwargs(**overrides):
    kwargs = {
        "inputs_embeds": torch.zeros(1, 7, 2048, dtype=torch.bfloat16),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "max_new_tokens": 8,
        "do_sample": True,
        "top_k": 5,
        "temperature": 0.9,
    }
    kwargs.update(overrides)
    return kwargs


class IsSupportedTest(unittest.TestCase):
    def test_rejects_missing_inputs_embeds(self):
        """A call without inputs_embeds takes the eager path."""
        self.assertFalse(graph_talker.is_supported(FakeTalker(), make_kwargs(inputs_embeds=None)))

    def test_rejects_cpu_inputs_embeds(self):
        """A CPU input takes the eager path."""
        self.assertFalse(graph_talker.is_supported(FakeTalker(), make_kwargs()))

    def test_rejects_non_bf16_inputs_embeds(self):
        """A non-BF16 input takes the eager path."""
        kwargs = make_kwargs(inputs_embeds=torch.zeros(1, 7, 2048, dtype=torch.float32))
        self.assertFalse(graph_talker.is_supported(FakeTalker(), kwargs))

    def test_rejects_batch_above_one(self):
        """A batch above one takes the eager path."""
        kwargs = make_kwargs(inputs_embeds=torch.zeros(2, 7, 2048, dtype=torch.bfloat16))
        self.assertFalse(graph_talker.is_supported(FakeTalker(), kwargs))

    def test_rejects_greedy(self):
        """Greedy decoding takes the eager path."""
        self.assertFalse(graph_talker.is_supported(FakeTalker(), make_kwargs(do_sample=False)))

    def test_rejects_beams(self):
        """Beam search takes the eager path."""
        self.assertFalse(graph_talker.is_supported(FakeTalker(), make_kwargs(num_beams=2)))

    def test_rejects_streamer(self):
        """A streamer takes the eager path."""
        self.assertFalse(graph_talker.is_supported(FakeTalker(), make_kwargs(streamer=object())))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_accepts_the_expected_call(self):
        """A BF16 CUDA batch-1 call is accepted."""
        kwargs = make_kwargs(inputs_embeds=torch.zeros(1, 7, 2048, device="cuda",
                                                       dtype=torch.bfloat16))
        self.assertTrue(graph_talker.is_supported(FakeTalker(), kwargs))


class StaticCacheTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_writes_at_the_set_offset(self):
        """A replay writes at the offset the caller set, not at a moving one."""
        # Arrange
        cache = graph_talker._GraphStaticCache(
            config=FakeConfig(), max_cache_len=16, device=torch.device("cuda"),
            dtype=torch.bfloat16)
        keys = torch.zeros(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
        values = torch.zeros(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
        cache.set_offset(5)

        # Act
        out_keys, _ = cache.layers[0].update(keys, values, {"cache_position": torch.tensor([0])})

        # Assert
        # The write lands at offset 5 with length 2, so the filled part is 7
        self.assertEqual(out_keys.shape[2], 7)
        # The offset does not move on update; the caller controls it
        self.assertEqual(cache.layers[0]._offset, 5)


class GraphTalkerGenerateTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_generate_matches_eager_warm_to_warm(self):
        """After both graphs are captured, the graph path matches the eager path."""
        # Arrange
        from test_talker_forward import make_talker_kwargs, make_tiny_talker

        talker = make_tiny_talker(device="cuda")
        kwargs = make_talker_kwargs(talker, "cuda")
        original = graph_talker.is_supported
        try:
            # Warm both paths so neither capture happens during the comparison
            graph_talker.is_supported = lambda *a, **k: False
            torch.manual_seed(1)
            talker.generate(**kwargs)
            graph_talker.is_supported = original
            torch.manual_seed(1)
            talker.generate(**kwargs)

            # Act
            graph_talker.is_supported = lambda *a, **k: False
            torch.manual_seed(1234)
            eager = talker.generate(**kwargs)
            eager_rng = torch.cuda.get_rng_state().clone()
            graph_talker.is_supported = original
            torch.manual_seed(1234)
            graph = talker.generate(**kwargs)
            graph_rng = torch.cuda.get_rng_state().clone()
        finally:
            graph_talker.is_supported = original

        # Assert
        self.assertTrue(torch.equal(eager.sequences, graph.sequences))
        self.assertTrue(torch.equal(eager_rng, graph_rng))
        self.assertEqual(len(eager.hidden_states), len(graph.hidden_states))
        for i, (e, g) in enumerate(zip(eager.hidden_states, graph.hidden_states)):
            self.assertTrue(torch.equal(e[0][-1], g[0][-1]), f"hidden mismatch at frame {i}")


if __name__ == "__main__":
    unittest.main()
