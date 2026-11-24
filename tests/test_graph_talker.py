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

CPU tests cover cache contents, growth and memory limits. CUDA tests also
check graph identity and compare warmed graph outputs with eager generation.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from qwen_tts.core.models import graph_talker
from tests.test_talker_forward import make_talker_kwargs, make_tiny_talker


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


class GraphBudgetTest(unittest.TestCase):
    def test_budget_reserves_headroom_on_small_and_large_gpus(self):
        runner = object.__new__(graph_talker.GraphTalkerForward)
        runner.device = torch.device("cuda:1")
        graph_bytes = graph_talker.GRAPH_BYTES
        cases = (
            # (device capacity, used bytes, additional graphs within the budget)
            (8 * 2**30, 8 * 2**30 - 12 * graph_bytes, 3),
            (24 * 2**30, 20_000_000_000 - 4 * graph_bytes, 1),
            (24 * 2**30, 20_000_000_000 - 4 * graph_bytes + 1, 0),
            (24 * 2**30, 20_000_000_001, 0),
        )
        for total, used, expected in cases:
            with self.subTest(total=total, used=used), patch.object(
                torch.cuda, "mem_get_info", return_value=(total - used, total)
            ) as memory_info:
                self.assertEqual(runner._graph_budget(), expected)
                memory_info.assert_called_once_with(runner.device)

    def test_existing_graph_replays_after_headroom_runs_out(self):
        total = 24 * 2**30
        with patch.object(torch.cuda, "mem_get_info", return_value=(total, total)):
            runner = graph_talker.GraphTalkerForward(make_tiny_talker(), 16)

        captured = Mock()
        runner.graphs[runner.position] = captured
        with patch.object(torch.cuda, "mem_get_info", return_value=(total - 20_000_000_000, total)) as memory_info:
            runner.decode(runner.inputs_embeds)
            captured.replay.assert_called_once_with()
            memory_info.assert_not_called()
            self.assertIsNone(runner._graph_for(runner.position))
            memory_info.assert_called_once_with(runner.device)


class GraphTalkerGenerateTest(unittest.TestCase):
    @torch.inference_mode()
    def test_static_cache_reuse_and_capacity_growth(self):
        """Exercise the real prefill and bounded eager fallback on CPU."""
        talker = make_tiny_talker()
        original = make_talker_kwargs(talker, "cpu")
        shorter = make_talker_kwargs(talker, "cpu")
        shorter["inputs_embeds"] = shorter["inputs_embeds"][:, :-1]
        shorter["attention_mask"] = shorter["attention_mask"][:, :-1]
        shorter["eos_token_id"] = None

        torch.manual_seed(1234)
        expected = talker.generate(**shorter)
        with patch.object(graph_talker, "is_supported", return_value=True), patch.object(
            graph_talker.GraphTalkerForward, "_graph_budget", return_value=0
        ):
            talker.generate(**original)
            runner = talker._graph_talker_forward
            for layer in runner.cache.layers:
                layer.keys.fill_(float("nan"))
                layer.values.fill_(float("nan"))

            torch.manual_seed(1234)
            actual = talker.generate(**shorter)
            self.assertIs(talker._graph_talker_forward, runner)
            self.assertEqual(runner.graphs, {})
            self.assertEqual(runner.position, shorter["inputs_embeds"].size(1) + actual.sequences.size(1) - 1)
            self.assert_generation_equal(expected, actual)

            # Reserve one slot beyond capacity, and stop at EOS after two tokens.
            larger = dict(original, max_new_tokens=runner.max_cache_len - original["inputs_embeds"].size(1) + 1)
            eos = torch.tensor([[original["eos_token_id"]]])
            with patch.object(talker, "_sample_code", return_value=eos):
                talker.generate(**larger)
            self.assertIsNot(talker._graph_talker_forward, runner)
            self.assertGreater(talker._graph_talker_forward.max_cache_len, runner.max_cache_len)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_replays_shared_positions_and_matches_eager_across_prefills(self):
        """Alternate requests while checking graph identities and exact outputs."""
        talker = make_tiny_talker(device="cuda")
        longer = make_talker_kwargs(talker, "cuda")
        shorter = make_talker_kwargs(talker, "cuda")
        shorter["inputs_embeds"] = shorter["inputs_embeds"][:, :-1]
        shorter["attention_mask"] = shorter["attention_mask"][:, :-1]
        for kwargs in (longer, shorter):
            # Reach the same number of decode positions on every run.
            kwargs["eos_token_id"] = None

        talker.generate(**longer)
        runner = talker._graph_talker_forward
        long_start = longer["inputs_embeds"].size(1)
        initial = dict(runner.graphs)
        self.assertIn(long_start, initial)

        talker.generate(**shorter)
        self.assertIs(talker._graph_talker_forward, runner)
        self.assertEqual(set(runner.graphs), set(initial) | {long_start - 1})
        for position, graph in initial.items():
            self.assertIs(runner.graphs[position], graph)
        warmed = dict(runner.graphs)

        for kwargs in (longer, shorter, longer):
            with self.subTest(prefill_len=kwargs["inputs_embeds"].size(1)):
                with patch.object(graph_talker, "is_supported", return_value=False):
                    torch.manual_seed(1234)
                    expected = talker.generate(**kwargs)
                    expected_rng = torch.cuda.get_rng_state()
                torch.manual_seed(1234)
                actual = talker.generate(**kwargs)
                self.assertTrue(torch.equal(expected_rng, torch.cuda.get_rng_state()))
                self.assert_generation_equal(expected, actual)
                self.assertIs(talker._graph_talker_forward, runner)
                self.assertEqual(runner.graphs, warmed)

    def assert_generation_equal(self, expected, actual):
        self.assertTrue(torch.equal(expected.sequences, actual.sequences))
        self.assertEqual(len(expected.hidden_states), len(actual.hidden_states))
        for i, (e, a) in enumerate(zip(expected.hidden_states, actual.hidden_states)):
            self.assertTrue(torch.equal(e[0][-1], a[0][-1]), f"hidden mismatch at frame {i}")
            if i:
                self.assertTrue(torch.equal(e[1], a[1]), f"codec mismatch at frame {i}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @torch.inference_mode()
    def test_full_cache_keeps_overlap_and_makes_room_for_new_prompt(self):
        talker = make_tiny_talker(device="cuda")
        kwargs = make_talker_kwargs(talker, "cuda")
        kwargs["eos_token_id"] = None
        with patch.object(graph_talker.GraphTalkerForward, "_graph_budget", return_value=2):
            talker.generate(**kwargs)
            runner = talker._graph_talker_forward
            start = runner.prefill_len
            self.assertEqual(set(runner.graphs), {start, start + 1})
            common_graph = runner.graphs[start]

            kwargs["inputs_embeds"] = kwargs["inputs_embeds"][:, :-1]
            kwargs["attention_mask"] = kwargs["attention_mask"][:, :-1]
            talker.generate(**kwargs)
            self.assertIs(talker._graph_talker_forward, runner)
            self.assertIs(runner.graphs[start], common_graph)
            self.assertEqual(set(runner.graphs), {start - 1, start})


if __name__ == "__main__":
    unittest.main()
