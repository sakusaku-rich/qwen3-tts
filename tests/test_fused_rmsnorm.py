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
"""Check the input guards and the numerical agreement of fused_rmsnorm.

The guard tests run without a GPU. The numerical tests run only when CUDA is available.
"""

import unittest

import torch

from qwen_tts.core.models import fused_rmsnorm


def reference_rms_norm(hidden_states, weight, eps):
    """Compute with the same operation order as `Qwen3TTSRMSNorm.forward`."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


class WeightIsSupportedTest(unittest.TestCase):
    def test_rejects_cpu_weight(self):
        """A CPU weight is rejected by the Triton kernel."""
        # Arrange
        weight = torch.randn(128, dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.weight_is_supported(weight)

        # Assert
        self.assertFalse(supported)

    def test_rejects_float32_weight(self):
        """A non-BF16 weight is rejected by the Triton kernel."""
        # Arrange
        weight = torch.randn(128, dtype=torch.float32)

        # Act
        supported = fused_rmsnorm.weight_is_supported(weight)

        # Assert
        self.assertFalse(supported)

    def test_rejects_unsupported_width(self):
        """A weight whose width is neither 128 nor 1024 is rejected."""
        # Arrange
        weight = torch.randn(256, dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.weight_is_supported(weight)

        # Assert
        self.assertFalse(supported)

    def test_rejects_two_dimensional_weight(self):
        """A weight that is not one dimensional is rejected."""
        # Arrange
        weight = torch.randn(2, 128, dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.weight_is_supported(weight)

        # Assert
        self.assertFalse(supported)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_accepts_bf16_cuda_weight(self):
        """A BF16 CUDA weight with a supported width is accepted."""
        # Arrange
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.weight_is_supported(weight)

        # Assert
        self.assertTrue(supported)


class InputIsSupportedTest(unittest.TestCase):
    def test_rejects_cpu_input(self):
        """A CPU input is rejected by the Triton kernel."""
        # Arrange
        hidden_states = torch.randn(1, 2, 8, 128, dtype=torch.bfloat16)
        weight = torch.randn(128, dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.input_is_supported(hidden_states, weight)

        # Assert
        self.assertFalse(supported)

    def test_rejects_float32_input(self):
        """A non-BF16 input is rejected by the Triton kernel."""
        # Arrange
        hidden_states = torch.randn(1, 2, 8, 128, dtype=torch.float32)
        weight = torch.randn(128, dtype=torch.float32)

        # Act
        supported = fused_rmsnorm.input_is_supported(hidden_states, weight)

        # Assert
        self.assertFalse(supported)

    def test_rejects_width_mismatch(self):
        """An input whose last dimension does not match the weight width is rejected."""
        # Arrange
        hidden_states = torch.randn(1, 2, 8, 128, dtype=torch.bfloat16)
        weight = torch.randn(1024, dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.input_is_supported(hidden_states, weight)

        # Assert
        self.assertFalse(supported)

    def test_rejects_non_contiguous_last_dimension(self):
        """An input whose last dimension is not contiguous is rejected."""
        # Arrange
        hidden_states = torch.randn(1, 8, 2, 128, dtype=torch.bfloat16).transpose(1, 2)
        weight = torch.randn(128, dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.input_is_supported(hidden_states, weight)

        # Assert
        self.assertFalse(supported)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_accepts_bf16_cuda_input(self):
        """A BF16 CUDA input that matches the weight width is accepted."""
        # Arrange
        hidden_states = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)

        # Act
        supported = fused_rmsnorm.input_is_supported(hidden_states, weight)

        # Assert
        self.assertTrue(supported)


class FusedRMSNormTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_matches_reference_bitwise(self):
        """The kernel matches the reference bit for bit on the shapes the code predictor uses."""
        # Arrange
        cases = [
            (1, 1, 8, 128),
            (1, 1, 16, 128),
            (1, 2, 8, 128),
            (1, 2, 16, 128),
            (1, 1, 1024),
            (1, 2, 1024),
        ]
        eps = 1e-6

        # Act
        results = []
        for shape in cases:
            hidden_states = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
            expected = reference_rms_norm(hidden_states, weight, eps)
            actual = fused_rmsnorm.fused_rms_norm(hidden_states, weight, eps)
            results.append((shape, torch.equal(actual, expected)))

        # Assert
        for shape, equal in results:
            self.assertTrue(equal, f"result mismatch for shape={shape}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_returns_same_shape_and_dtype(self):
        """The result has the same shape and dtype as the input."""
        # Arrange
        hidden_states = torch.randn(1, 2, 8, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)

        # Act
        actual = fused_rmsnorm.fused_rms_norm(hidden_states, weight, 1e-6)

        # Assert
        self.assertEqual(actual.shape, hidden_states.shape)
        self.assertEqual(actual.dtype, hidden_states.dtype)
        self.assertTrue(actual.is_contiguous())


if __name__ == "__main__":
    unittest.main()
