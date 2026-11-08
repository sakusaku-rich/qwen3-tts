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
"""Replay the talker's 28-layer decode forward with a CUDA Graph.

The talker spends most of a generation in its decode loop. Measured on an RTX
3090 with the 1.7B model, the talker frames are about 68% of a long generation,
the code predictor inside them is about 25%, and the 28-layer forward of one
frame is about 17ms.

`Qwen3TTSTalkerForConditionalGeneration` runs that forward through
`GenerationMixin.generate`, which appends to a `DynamicCache` with `torch.cat`
and rebuilds the attention mask on every step. This module runs the same
forward with a fixed cache and captures it in a CUDA Graph. The code predictor,
the embedding sum, the sampling and the EOS check stay on the Python side,
exactly as in the eager forward, so the captured step only replaces the
28-layer forward.

Measured on an RTX 3090 with the 1.7B model, one 28-layer decode step drops from
about 29.7ms to about 6.1ms when replayed from the graph.

The decode step passes `attention_mask=None`. With a 2D padding mask, SDPA's
mask builder calls `padding_mask.all()`, which synchronizes the CPU and aborts
the capture. Every position of a decode step is valid, so the causal mask alone
is enough and the values are unchanged.

The output must stay bit-identical to the eager loop. The details that make
that possible:

- `_GraphStaticLayer` writes at the offset the caller sets, so a replay always
  lands on the same slot, and returns only the filled part, so the attention
  sees the same keys and values as `DynamicCache`.
- The prefill stays outside the graph. It runs once per call with the same
  `DynamicCache` path as the eager loop, and its keys and values are copied into
  the static cache afterwards.
- The position is a Python int inside the captured step, so each decode position
  has its own graph. Graphs are captured lazily, on the first replay that needs
  them.

Capturing a graph runs the step a few times, which consumes the RNG. The first
call that captures therefore differs from a fully eager run with the same seed,
exactly as the code predictor's graph does. Once the graphs exist, a run with a
fixed seed matches the eager run bit for bit.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import StaticCache, StaticLayer


class _GraphStaticLayer(StaticLayer):
    """A static cache layer that writes at a Python offset.

    `graph_predictor._StaticLayer` advances its offset on every update, which a
    captured step cannot do. This layer writes at the offset the caller set, so
    a replay always lands on the same slot.
    """

    def __init__(self, max_cache_len: int, positions: torch.Tensor):
        super().__init__(max_cache_len=max_cache_len)
        self._positions = positions
        self._offset = 0

    def update(self, key_states, value_states, cache_kwargs=None):
        length = key_states.shape[2]
        start = self._offset
        end = start + length
        cache_kwargs = dict(cache_kwargs or {})
        cache_kwargs["cache_position"] = self._positions[start:end]
        keys, values = super().update(key_states, value_states, cache_kwargs)
        return keys[:, :, :end], values[:, :, :end]


class _GraphStaticCache(StaticCache):
    """A `StaticCache` whose write offset the caller controls."""

    is_compileable = False

    def __init__(self, config, max_cache_len: int, device: torch.device, dtype: torch.dtype):
        super().__init__(config=config, max_cache_len=max_cache_len)
        positions = torch.arange(max_cache_len, device=device)
        self.layers = [
            _GraphStaticLayer(max_cache_len=max_cache_len, positions=positions)
            for _ in range(config.num_hidden_layers)
        ]

    def set_offset(self, value: int) -> None:
        for layer in self.layers:
            layer._offset = value

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return cache_position.shape[0], 0


def is_supported(talker: nn.Module, kwargs: dict) -> bool:
    """Return whether the graph path can run this call."""
    if not torch.cuda.is_available():
        return False
    if kwargs.get("do_sample") is False:
        return False
    for name in ("num_beams", "num_return_sequences"):
        if kwargs.get(name, 1) not in (None, 1):
            return False
    for name in ("streamer", "assistant_model", "prompt_lookup_num_tokens",
                 "synced_gpus", "logits_processor", "stopping_criteria"):
        if kwargs.get(name) is not None:
            return False
    embeds = kwargs.get("inputs_embeds")
    if not isinstance(embeds, torch.Tensor):
        return False
    if not embeds.is_cuda or embeds.dtype != torch.bfloat16:
        return False
    if embeds.shape[0] != 1:
        return False
    return True


class GraphTalkerForward:
    """Run the talker's decode forwards with the 28-layer step captured.

    The prefill runs eagerly. Each decode forward copies its input embedding
    into a fixed buffer, replays the graph for its position and returns the
    final hidden state.
    """

    def __init__(self, talker: nn.Module, max_cache_len: int, prefill_len: int):
        self.talker = talker
        self.device = next(talker.parameters()).device
        self.max_cache_len = max_cache_len
        self.prefill_len = prefill_len
        self.cache = _GraphStaticCache(config=talker.config, max_cache_len=max_cache_len,
                                       device=self.device, dtype=torch.bfloat16)
        self.cache_position = torch.arange(max_cache_len, device=self.device)
        self.position_ids = torch.arange(max_cache_len, device=self.device).view(1, 1, -1)
        self.inputs_embeds = torch.zeros(1, 1, talker.config.hidden_size,
                                         device=self.device, dtype=torch.bfloat16)
        self.last_hidden = torch.zeros(1, 1, talker.config.hidden_size,
                                       device=self.device, dtype=torch.bfloat16)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.position = prefill_len

    def prefill(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Run the prefill eagerly and fill the static cache."""
        cache_position = torch.arange(self.prefill_len, device=self.device)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        outputs = self.talker.model(
            input_ids=None,
            attention_mask=attention_mask[:, :self.prefill_len],
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=cache_position,
        )
        self.cache.set_offset(0)
        for layer, (keys, values) in zip(self.cache.layers, outputs.past_key_values):
            if layer.keys is None:
                layer.lazy_initialization(keys)
            length = keys.shape[2]
            layer.keys[:, :, :length].copy_(keys)
            layer.values[:, :, :length].copy_(values)
        self.position = self.prefill_len
        return outputs.last_hidden_state

    def _decode_step(self, position: int) -> None:
        """Run one decode forward on the fixed buffers at `position`."""
        outputs = self.talker.model(
            input_ids=None,
            # A 2D padding mask makes SDPA's mask builder synchronize the CPU,
            # which aborts the capture. Every decode position is valid.
            attention_mask=None,
            position_ids=self.position_ids[:, :, position:position + 1].expand(3, 1, -1),
            past_key_values=self.cache,
            inputs_embeds=self.inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=self.cache_position[position:position + 1],
        )
        self.last_hidden.copy_(outputs.last_hidden_state)

    def _graph_for(self, position: int) -> torch.cuda.CUDAGraph:
        """Return the graph for `position`, capturing it on first use."""
        graph = self.graphs.get(position)
        if graph is not None:
            return graph
        # Warm up on a side stream, as CUDA Graph capture requires
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.cache.set_offset(position)
                self._decode_step(position)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        self.cache.set_offset(position)
        with torch.cuda.graph(graph):
            self._decode_step(position)
        torch.cuda.synchronize()
        self.graphs[position] = graph
        return graph

    def capture(self, num_steps: int) -> None:
        """Capture the graphs for the first `num_steps` decode positions."""
        for position in range(self.prefill_len, self.prefill_len + num_steps):
            self._graph_for(position)

    def decode(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Replay the decode forward for the next position."""
        graph = self._graph_for(self.position)
        self.inputs_embeds.copy_(inputs_embeds)
        self.cache.set_offset(self.position)
        graph.replay()
        self.position += 1
        return self.last_hidden
