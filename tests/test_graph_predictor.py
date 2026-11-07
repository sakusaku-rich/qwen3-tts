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
"""Check the guard and the cache behaviour of graph_predictor.

The guard tests run without a GPU. The cache tests run only when CUDA is available.
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
