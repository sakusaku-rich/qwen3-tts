# Copyright 2026 sakusaku-rich
# SPDX-License-Identifier: Apache-2.0
"""Check the layout and sharing of the talker's multimodal rotary tables."""

import unittest
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache

from qwen_tts.core.models import modeling_qwen3_tts as modeling
from tests.test_talker_forward import make_tiny_talker


class TalkerMropeTest(unittest.TestCase):
    def test_layouts_select_the_original_channels(self):
        cos = torch.arange(24, dtype=torch.bfloat16).view(3, 1, 1, 8)
        sin = -cos
        for interleaved, expected in (
            (True, [0, 9, 18, 3, 0, 9, 18, 3]),
            (False, [0, 1, 10, 19, 4, 5, 14, 23]),
        ):
            with self.subTest(interleaved=interleaved):
                tables = modeling._prepare_mrope_tables(cos, sin, [2, 1, 1], interleaved)
                expected_cos = cos.new_tensor(expected).view(1, 1, 8)
                self.assertTrue(torch.equal(tables[0], expected_cos))
                self.assertTrue(torch.equal(tables[1], -expected_cos))

    @torch.inference_mode()
    def test_one_preparation_per_forward_preserves_raw_cache_tables(self):
        model = make_tiny_talker().model
        embeddings = torch.randn(1, 5, model.config.hidden_size, dtype=torch.bfloat16)
        positions = torch.arange(15).view(3, 1, 5)
        raw_tables = model.rotary_emb(embeddings, positions)
        cache = DynamicCache()
        with patch.object(modeling, "_prepare_mrope_tables", wraps=modeling._prepare_mrope_tables) as prepare, \
             patch.object(cache, "update", wraps=cache.update) as update:
            model(inputs_embeds=embeddings, position_ids=positions, position_embeddings=raw_tables,
                  past_key_values=cache, use_cache=True)
        prepare.assert_called_once()
        self.assertEqual(update.call_count, len(model.layers))
        for call in update.call_args_list:
            cache_kwargs = call.args[3]
            self.assertIs(cache_kwargs["cos"], raw_tables[0])
            self.assertIs(cache_kwargs["sin"], raw_tables[1])
