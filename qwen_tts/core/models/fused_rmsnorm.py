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
"""Compute BF16 RMSNorm with a single Triton kernel.

`Qwen3TTSRMSNorm.forward` casts to FP32 and then runs the square, mean, rsqrt and
weight multiply as separate PyTorch ops. A row is only 128 or 1024 elements, so on
a GPU the kernel launches and the round trips through intermediate buffers dominate.
This module fuses the ops after the mean into one kernel to cut both.

Two rules keep the output bit-identical to the original:

- The variance reduction stays in PyTorch (`pow(2).mean(-1)`). Changing the
  summation order changes the result.
- The normalized value is rounded to BF16 once before the weight multiply, which
  preserves where the original implementation rounds.

SM75 (Turing) has no BF16 arithmetic instructions, so BF16 is loaded as uint16,
computed in FP32 and stored back with round-to-nearest-even. The kernel launches
with `enable_fp_fusion=False` so FMA contraction cannot change the rounding.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # without Triton, every call falls back to the original path
    triton = None
    tl = None

# Q/K norm width (head_dim) and block norm width (hidden_size) in the code predictor
SUPPORTED_WIDTHS = (128, 1024)

if triton is not None:

    @triton.jit
    def _decode_bf16(bits):
        return (bits.to(tl.uint32) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _encode_bf16(value):
        bits = value.to(tl.uint32, bitcast=True)
        # round to nearest, ties to even. Canonicalize NaN so that a small NaN
        # payload does not round to infinity. No native BF16 arithmetic is emitted.
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).to(tl.uint16)
        return tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0x7FC0, rounded).to(tl.uint16)

    @triton.jit
    def _row_offset(row, ROW_SHAPE: tl.constexpr, ROW_STRIDES: tl.constexpr):
        offset = tl.full((), 0, tl.int32)
        remaining = row
        for axis in tl.static_range(len(ROW_SHAPE) - 1, -1, -1):
            offset += (remaining % ROW_SHAPE[axis]) * ROW_STRIDES[axis]
            remaining = remaining // ROW_SHAPE[axis]
        return offset

    @triton.jit
    def _rms_norm(X, W, V, Y, N: tl.constexpr, EPS: tl.constexpr,
                  ROW_SHAPE: tl.constexpr, ROW_STRIDES: tl.constexpr,
                  BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        offset = _row_offset(row, ROW_SHAPE, ROW_STRIDES)
        xp = X.to(tl.pointer_type(tl.uint16))
        wp = W.to(tl.pointer_type(tl.uint16))
        yp = Y.to(tl.pointer_type(tl.uint16))
        x = _decode_bf16(tl.load(xp + offset + col, mask=col < N, other=0))
        w = _decode_bf16(tl.load(wp + col, mask=col < N, other=0))
        variance = tl.load(V + row)
        normalized = x * tl.rsqrt(variance + EPS)
        normalized_bf16 = _decode_bf16(_encode_bf16(normalized))
        result = _encode_bf16(w * normalized_bf16)
        tl.store(yp + row * N + col, result, mask=col < N)


def weight_is_supported(weight: torch.Tensor) -> bool:
    """Return whether the weight can be used by the Triton kernel.

    The weight dtype, shape, stride and device do not change after the model is
    loaded, so call this once per module instead of on every forward. The input is
    assumed to share the weight device and dtype; `input_is_supported` checks it.
    """
    if triton is None:
        return False
    if not weight.is_cuda or weight.dtype != torch.bfloat16:
        return False
    if weight.ndim != 1 or weight.shape[0] not in SUPPORTED_WIDTHS:
        return False
    if weight.stride(0) != 1:
        return False
    return True


def input_is_supported(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether the input can be computed by the Triton kernel.

    The weight conditions are assumed to be checked by `weight_is_supported`. Only
    the parts that can change per call are checked: device, dtype, last dimension
    and stride.
    """
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        return False
    if hidden_states.ndim < 1 or hidden_states.shape[-1] != weight.shape[0]:
        return False
    if hidden_states.stride(-1) != 1:
        return False
    return True


def is_supported(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether the input and weight can be computed by the Triton kernel.

    Checks both sides. For a per-call check use `input_is_supported` instead.
    """
    return weight_is_supported(weight) and input_is_supported(hidden_states, weight)


# Keep one compiled launcher per (shape, stride). The code predictor calls the norm
# 315 times per frame, so rebuilding the launch arguments from the shape and stride
# on every call would eat up the savings from the kernel itself.
_launchers: dict[tuple, tuple] = {}


def _launcher_for(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float):
    """Return the launcher for this shape and stride, compiling it on first use."""
    key = (hidden_states.shape, hidden_states.stride(), float(eps))
    state = _launchers.get(key)
    if state is None:
        width = hidden_states.shape[-1]
        rows = math.prod(hidden_states.shape[:-1])
        const = (
            width,
            float(eps),
            tuple(hidden_states.shape[:-1]),
            tuple(hidden_states.stride()[:-1]),
            triton.next_power_of_2(width),
        )
        output = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        kernel = _rms_norm[(rows,)](
            hidden_states,
            weight,
            variance,
            output,
            N=const[0],
            EPS=const[1],
            ROW_SHAPE=const[2],
            ROW_STRIDES=const[3],
            BLOCK=const[4],
            num_warps=4,
            enable_fp_fusion=False,
        )
        state = (kernel[(rows, 1, 1)], const)
        _launchers[key] = state
    return state


def fused_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute a BF16 RMSNorm with the Triton kernel.

    Accepts only inputs where `weight_is_supported` and `input_is_supported` are
    both true, and returns the same values as `Qwen3TTSRMSNorm.forward`. The result
    has the same shape and dtype as the input.
    """
    rows = math.prod(hidden_states.shape[:-1])
    if rows == 0:
        return torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    runner, const = _launcher_for(hidden_states, weight, eps)
    # Keep the variance reduction in PyTorch so the summation order is unchanged
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    output = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    runner(hidden_states, weight, variance, output, *const)
    return output
