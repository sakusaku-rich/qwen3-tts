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
"""Check the rope position branch of the talker forward.

`Qwen3TTSTalkerForConditionalGeneration.forward` picks the position ids from
`generation_step` instead of reading `cache_position[0]`, which would synchronize
the CPU with the GPU on every decode step. The tests run a tiny talker on any
device and check that the prefill recomputes the rope index while the decode
steps reuse `rope_deltas`. Both branches behave the same as before the change on
every path `generate` can take.
"""

import unittest

import torch

from qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerForConditionalGeneration,
)

TALKER_HIDDEN = 64
PREDICTOR_HIDDEN = 32
VOCAB = 24
CODE_GROUPS = 6
PROMPT_LEN = 7
FRAMES = 3


def make_tiny_talker(device="cpu", seed=0):
    """Build a small talker that runs a few frames quickly."""
    torch.manual_seed(seed)
    predictor_config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=VOCAB,
        hidden_size=PREDICTOR_HIDDEN,
        intermediate_size=PREDICTOR_HIDDEN * 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        num_code_groups=CODE_GROUPS,
    )
    # head_dim = 64 / 4 = 16, so the mrope sections sum to 8
    config = Qwen3TTSTalkerConfig(
        code_predictor_config=predictor_config,
        vocab_size=VOCAB,
        hidden_size=TALKER_HIDDEN,
        intermediate_size=TALKER_HIDDEN * 4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_code_groups=CODE_GROUPS,
        text_hidden_size=32,
        text_vocab_size=16,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "interleaved": False},
    )
    talker = Qwen3TTSTalkerForConditionalGeneration(config)
    return talker.to(device=device, dtype=torch.bfloat16).eval()


def make_talker_kwargs(talker, device):
    """Build the kwargs the outer generate passes to the talker's generate."""
    embeds = torch.randn(1, PROMPT_LEN, TALKER_HIDDEN, device=device, dtype=torch.bfloat16)
    return {
        "inputs_embeds": embeds,
        "attention_mask": torch.ones(1, PROMPT_LEN, device=device, dtype=torch.long),
        "trailing_text_hidden": torch.randn(1, 4, TALKER_HIDDEN, device=device, dtype=torch.bfloat16),
        "tts_pad_embed": torch.zeros(1, 1, TALKER_HIDDEN, device=device, dtype=torch.bfloat16),
        "max_new_tokens": FRAMES,
        "min_new_tokens": 2,
        "do_sample": True,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 0.9,
        "eos_token_id": VOCAB - 1,
        "repetition_penalty": 1.05,
        "suppress_tokens": [],
        "subtalker_dosample": True,
        "subtalker_top_k": 5,
        "subtalker_temperature": 0.9,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }


def record_model_position_ids(talker):
    """Return a list that collects the position ids of every model forward."""
    recorded = []
    original = talker.model.forward

    def recorder(*args, **call_kwargs):
        recorded.append(call_kwargs.get("position_ids"))
        return original(*args, **call_kwargs)

    talker.model.forward = recorder

    def restore():
        talker.model.forward = original

    return recorded, restore


class TalkerGenerateTest(unittest.TestCase):
    def test_generate_runs_and_records_positions(self):
        """The tiny talker generates, and every forward gets 3D position ids."""
        # Arrange
        talker = make_tiny_talker()
        kwargs = make_talker_kwargs(talker, "cpu")
        recorded, restore = record_model_position_ids(talker)

        # Act
        try:
            torch.manual_seed(1234)
            result = talker.generate(**kwargs)
        finally:
            restore()

        # Assert
        # The prefill sample is the first token, so FRAMES tokens take one
        # prefill forward and FRAMES - 1 decode forwards.
        self.assertEqual(result.sequences.shape, (1, FRAMES))
        self.assertEqual(len(recorded), FRAMES)
        self.assertEqual(tuple(recorded[0].shape), (3, 1, PROMPT_LEN))
        for position_ids in recorded[1:]:
            self.assertEqual(tuple(position_ids.shape), (3, 1, 1))

    def test_prefill_recomputes_the_rope_index(self):
        """The prefill position ids follow the attention mask cumsum."""
        # Arrange
        talker = make_tiny_talker()
        kwargs = make_talker_kwargs(talker, "cpu")
        recorded, restore = record_model_position_ids(talker)

        # Act
        try:
            torch.manual_seed(1234)
            talker.generate(**kwargs)
        finally:
            restore()

        # Assert
        prefill = recorded[0]
        expected = torch.arange(PROMPT_LEN).view(1, -1).expand(3, 1, -1)
        self.assertTrue(torch.equal(prefill.cpu(), expected))
        # An all-ones mask produces a zero delta
        self.assertIsNotNone(talker.rope_deltas)
        self.assertTrue(torch.all(talker.rope_deltas == 0))

    def test_decode_reuses_the_rope_delta(self):
        """Each decode step shifts by cache_position plus the stored delta."""
        # Arrange
        talker = make_tiny_talker()
        kwargs = make_talker_kwargs(talker, "cpu")
        recorded, restore = record_model_position_ids(talker)

        # Act
        try:
            torch.manual_seed(1234)
            talker.generate(**kwargs)
        finally:
            restore()

        # Assert
        for frame in range(FRAMES - 1):
            position_ids = recorded[1 + frame]
            expected = torch.full((3, 1, 1), PROMPT_LEN + frame)
            self.assertTrue(torch.equal(position_ids.cpu(), expected),
                            f"frame {frame} got {position_ids.flatten().tolist()}")


class TalkerDirectForwardTest(unittest.TestCase):
    def test_decode_step_uses_the_delta_path(self):
        """A decode step with a shifted cache_position reuses rope_deltas.

        Reading `cache_position[0]` on the device would both synchronize and take
        the prefill branch here when the position is 0. The Python int
        `generation_step` keeps this on the delta path for every decode step.
        """
        # Arrange
        talker = make_tiny_talker()
        device = "cpu"
        mask = torch.ones(1, PROMPT_LEN, dtype=torch.long, device=device)
        # A prefill sets rope_deltas
        torch.manual_seed(1234)
        talker(
            inputs_embeds=torch.randn(1, PROMPT_LEN, TALKER_HIDDEN, dtype=torch.bfloat16),
            attention_mask=mask,
            generation_step=-1,
            cache_position=torch.arange(PROMPT_LEN, device=device),
            use_cache=False,
        )
        recorded, restore = record_model_position_ids(talker)

        # Act
        try:
            talker(
                input_ids=torch.zeros(1, 1, dtype=torch.long, device=device),
                attention_mask=torch.ones(1, 1, dtype=torch.long, device=device),
                past_hidden=torch.randn(1, 1, TALKER_HIDDEN, dtype=torch.bfloat16),
                trailing_text_hidden=torch.randn(1, 4, TALKER_HIDDEN, dtype=torch.bfloat16),
                tts_pad_embed=torch.zeros(1, 1, TALKER_HIDDEN, dtype=torch.bfloat16),
                generation_step=0,
                cache_position=torch.tensor([PROMPT_LEN], device=device),
                use_cache=False,
            )
        finally:
            restore()

        # Assert
        expected = torch.full((3, 1, 1), PROMPT_LEN)
        self.assertTrue(torch.equal(recorded[0].cpu(), expected))


if __name__ == "__main__":
    unittest.main()
