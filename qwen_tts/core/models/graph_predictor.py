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
"""Replay the code predictor's decode loop with a CUDA Graph.

The code predictor is CPU bound. One frame enqueues about 6,300 kernels for 15
decode steps, and the GPU runs for only about 23ms of the roughly 90ms the frame
takes. `torch.compile` with `reduce-overhead` does not help because the input shape
changes between the prefill and the decode steps, so it recaptures the graph.

This module writes the 15 steps out explicitly on fixed buffers and captures them
once in a CUDA Graph. A replay then runs the whole frame from the GPU side, so the
CPU launch cost mostly disappears.

The output must stay bit-identical to `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.generate`.
`DynamicCache` and `StaticCache` differ in three ways that change the numbers, and
`_StaticCache` below reproduces the `DynamicCache` behaviour:

- `DynamicCache` returns no attention mask and lets SDPA build the causal mask, so
  `get_mask_sizes` reports the filled length.
- `DynamicCache` appends each new chunk, while the eager generate passes
  `cache_position=[1]` for every decode step. Writing by that index would overwrite
  the prefill slot, so `_StaticLayer.update` appends at the filled offset.
- `DynamicCache` returns only the filled keys and values, so `_StaticLayer.update`
  slices the padded tail away.

Sampling follows `GenerationMixin.sample`: the logits are cast to FP32, divided by
the temperature, filtered by top-k, passed through softmax and drawn with
`torch.multinomial`. The RNG state is consumed in the same order as the eager path.
"""

from __future__ import annotations

import torch
from torch import nn

from transformers.cache_utils import StaticCache, StaticLayer

# The predictor always runs one frame at a time with 2 prefill tokens followed by
# num_code_groups - 1 decode steps.
PREFILL_TOKENS = 2


class _StaticLayer(StaticLayer):
    """A static cache layer that behaves like a `DynamicCache` layer."""

    def __init__(self, max_cache_len: int, positions: torch.Tensor):
        super().__init__(max_cache_len=max_cache_len)
        self._filled = 0
        # Preallocated so an update inside graph capture does not allocate
        self._positions = positions

    def update(self, key_states, value_states, cache_kwargs=None):
        length = key_states.shape[2]
        start = self._filled
        end = start + length
        # Append at the filled offset instead of writing at cache_position
        cache_kwargs = dict(cache_kwargs or {})
        cache_kwargs["cache_position"] = self._positions[start:end]
        keys, values = super().update(key_states, value_states, cache_kwargs)
        self._filled = end
        # Return only the filled part so attention sees the same length as DynamicCache
        return keys[:, :, :end], values[:, :, :end]

    def reset(self):
        self._filled = 0


class _StaticCache(StaticCache):
    """A `StaticCache` that produces the same attention inputs as `DynamicCache`."""

    # StaticCache sets this to True, which forces an explicit mask. DynamicCache
    # leaves it False, so SDPA builds the causal mask itself.
    is_compileable = False

    def __init__(self, config, max_cache_len: int, device: torch.device, dtype: torch.dtype):
        super().__init__(config=config, max_cache_len=max_cache_len)
        positions = torch.arange(max_cache_len, device=device)
        self.layers = [
            _StaticLayer(max_cache_len=max_cache_len, positions=positions)
            for _ in range(config.num_hidden_layers)
        ]

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def get_mask_sizes(self, cache_position, layer_idx=0):
        # DynamicCache reports the filled length, which makes create_causal_mask
        # return None. Before the first update the length is the query length.
        # Reading a device tensor here would sync and break graph capture.
        filled = self.layers[layer_idx]._filled
        return (filled if filled else cache_position.shape[0]), 0


def is_supported(predictor: nn.Module, kwargs: dict) -> bool:
    """Return whether the CUDA Graph path can run this call.

    The graph is captured once and replayed for every frame, so it needs a fixed
    shape: batch 1, a prefill of two tokens, and the decode steps the predictor
    always runs. Anything else takes the eager path.
    """
    if not torch.cuda.is_available():
        return False
    if kwargs.get("do_sample") is False:
        return False
    embeds = kwargs.get("inputs_embeds")
    if not isinstance(embeds, torch.Tensor):
        return False
    if not embeds.is_cuda or embeds.dtype != torch.bfloat16:
        return False
    if embeds.shape[0] != 1 or embeds.shape[1] != PREFILL_TOKENS:
        return False
    if kwargs.get("max_new_tokens") != predictor.config.num_code_groups - 1:
        return False
    return True


class GraphCodePredictor:
    """Run one frame's decode loop inside a CUDA Graph.

    Build it with a real frame's `generate` kwargs, call `capture()` once, then call
    `replay()` with the kwargs of each frame. `replay` copies the prefill embeddings
    into the captured buffer, so the graph itself stays fixed.
    """

    def __init__(self, predictor: nn.Module, sample_kwargs: dict):
        config = predictor.config
        self.predictor = predictor
        self.num_steps = config.num_code_groups - 1
        self.cache_len = PREFILL_TOKENS + self.num_steps
        self.top_k = sample_kwargs.get("top_k")
        self.temperature = sample_kwargs.get("temperature")
        self.device = next(predictor.parameters()).device
        self.cache = _StaticCache(config=config, max_cache_len=self.cache_len,
                                  device=self.device, dtype=torch.bfloat16)
        self.output = torch.zeros(1, self.num_steps, dtype=torch.long, device=self.device)
        # Fixed inputs, allocated once so capture never copies from the CPU.
        # capture() replaces prefill_embeds with a buffer of the right shape.
        self.prefill_embeds = None
        self.prefill_positions = torch.arange(PREFILL_TOKENS, device=self.device)
        self.prefill_position_ids = self.prefill_positions.unsqueeze(0)
        self.step_positions = [
            torch.tensor([1], device=self.device) for _ in range(self.num_steps)
        ]
        self.step_position_ids = [
            torch.tensor([[PREFILL_TOKENS + step]], device=self.device)
            for step in range(self.num_steps)
        ]
        self.graph: torch.cuda.CUDAGraph | None = None

    def _embed(self, code: torch.Tensor, step: int) -> torch.Tensor:
        """Embed one sampled code for the given decode step."""
        predictor = self.predictor
        embeds = predictor.model.get_input_embeddings()[step - 1](code.view(1, 1))
        return predictor.small_to_mtp_projection(embeds)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample one code the way `GenerationMixin.sample` does."""
        scores = logits.float()
        if self.temperature:
            scores = scores / self.temperature
        if self.top_k:
            top_k = min(self.top_k, scores.size(-1))
            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            scores = scores.masked_fill(indices_to_remove, float("-inf"))
        probs = torch.nn.functional.softmax(scores, dim=-1)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).squeeze(1)

    def _forward(self) -> torch.Tensor:
        """Run the prefill and the decode steps on the fixed buffers."""
        predictor = self.predictor
        outputs = predictor.model(
            input_ids=None,
            inputs_embeds=predictor.small_to_mtp_projection(self.prefill_embeds),
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.prefill_positions,
            position_ids=self.prefill_position_ids,
            attention_mask=None,
        )
        # The eager forward applies lm_head to the whole prefill tensor
        logits = predictor.lm_head[0](outputs.last_hidden_state)[:, -1:]
        code = self._sample(logits)
        self.output[:, 0] = code
        for step in range(1, self.num_steps):
            outputs = predictor.model(
                input_ids=None,
                inputs_embeds=self._embed(code, step),
                past_key_values=self.cache,
                use_cache=True,
                cache_position=self.step_positions[step - 1],
                position_ids=self.step_position_ids[step - 1],
                attention_mask=None,
            )
            logits = predictor.lm_head[step](outputs.last_hidden_state)
            code = self._sample(logits)
            self.output[:, step] = code
        return self.output

    def capture(self, sample_kwargs: dict) -> None:
        """Capture the decode loop. `sample_kwargs["inputs_embeds"]` seeds the buffers."""
        embeds = sample_kwargs["inputs_embeds"]
        self.prefill_embeds = torch.zeros_like(embeds)
        self.prefill_embeds.copy_(embeds)
        # Warm up on a side stream, as CUDA Graph capture requires
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.cache.reset()
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.cache.reset()
        with torch.cuda.graph(self.graph):
            self.output.copy_(self._forward())
        torch.cuda.synchronize()

    def replay(self, sample_kwargs: dict) -> torch.Tensor:
        """Replay the captured loop with a new frame's prefill embeddings."""
        if self.graph is None:
            raise RuntimeError("capture() must be called before replay()")
        self.prefill_embeds.copy_(sample_kwargs["inputs_embeds"])
        self.cache.reset()
        self.graph.replay()
        return self.output.clone()
