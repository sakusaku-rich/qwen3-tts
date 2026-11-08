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
"""Check the hidden state handling of the outer generate.

The talker forward no longer collects every layer's activations: it reports
only the final hidden state, which is the only element the callers read. The
outer generate honours `output_hidden_states=False`, and the codec ids still
travel through the per-step outputs. The tests run a tiny model on any device.
"""

import functools
import unittest

import torch

from qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

TALKER_HIDDEN = 64
PREDICTOR_HIDDEN = 32
VOCAB = 1048  # keeps the suppress token range inside the vocabulary
TEXT_VOCAB = 16
CODE_GROUPS = 6
FRAMES = 4
TEXT_LEN = 12


def make_tiny_tts(device="cpu", seed=0):
    """Build a small `Qwen3TTSForConditionalGeneration` with a voice clone prompt."""
    torch.manual_seed(seed)
    talker_config = dict(
        code_predictor_config=Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=VOCAB,
            hidden_size=PREDICTOR_HIDDEN,
            intermediate_size=PREDICTOR_HIDDEN * 4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            num_code_groups=CODE_GROUPS,
        ),
        vocab_size=VOCAB,
        hidden_size=TALKER_HIDDEN,
        intermediate_size=TALKER_HIDDEN * 4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_code_groups=CODE_GROUPS,
        text_hidden_size=32,
        text_vocab_size=TEXT_VOCAB,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "interleaved": False},
        codec_eos_token_id=VOCAB - 1,
        codec_think_id=VOCAB - 2,
        codec_nothink_id=VOCAB - 3,
        codec_think_bos_id=VOCAB - 4,
        codec_think_eos_id=VOCAB - 5,
        codec_pad_id=VOCAB - 6,
        codec_bos_id=VOCAB - 7,
        spk_id={},
        spk_is_dialect={},
        codec_language_id={},
    )
    config = Qwen3TTSConfig(
        talker_config=talker_config,
        tts_model_type="custom_voice",  # the speaker encoder is not used below
        tts_pad_token_id=TEXT_VOCAB - 1,
        tts_bos_token_id=TEXT_VOCAB - 2,
        tts_eos_token_id=TEXT_VOCAB - 3,
    )
    model = Qwen3TTSForConditionalGeneration(config)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    prompt = {
        "ref_code": [None],
        "ref_spk_embedding": [torch.randn(TALKER_HIDDEN, dtype=torch.bfloat16)],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }
    input_ids = [torch.arange(TEXT_LEN, device=device).unsqueeze(0) % TEXT_VOCAB]
    return model, prompt, input_ids


def generate_kwargs():
    return dict(
        languages=["Auto"],
        max_new_tokens=FRAMES,
        do_sample=True,
        top_k=5,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=5,
        subtalker_temperature=0.9,
    )


def record_talker_requests(model):
    """Record the output_hidden_states the talker model forward receives."""
    recorded = []
    original = model.talker.model.forward

    def recorder(*args, **call_kwargs):
        recorded.append(call_kwargs.get("output_hidden_states"))
        return original(*args, **call_kwargs)

    model.talker.model.forward = recorder

    def restore():
        model.talker.model.forward = original

    return recorded, restore


def record_talker_past_hidden(model):
    """Record the past_hidden of every talker forward."""
    recorded = []
    original = model.talker.forward

    @functools.wraps(original)
    def recorder(*args, **call_kwargs):
        result = original(*args, **call_kwargs)
        recorded.append(result.past_hidden)
        return result

    model.talker.forward = recorder

    def restore():
        model.talker.forward = original

    return recorded, restore


class TtsGenerateTest(unittest.TestCase):
    def test_generate_returns_codes_and_hidden(self):
        """The codes and the per-frame final hidden states come back."""
        # Arrange
        model, prompt, input_ids = make_tiny_tts()

        # Act
        torch.manual_seed(1234)
        codes, hiddens = model.generate(input_ids=input_ids,
                                        voice_clone_prompt=prompt,
                                        **generate_kwargs())

        # Assert
        self.assertEqual(len(codes), 1)
        self.assertEqual(codes[0].shape[1], CODE_GROUPS)
        self.assertGreater(codes[0].shape[0], 0)
        self.assertEqual(len(hiddens), 1)
        self.assertEqual(hiddens[0].shape[-1], TALKER_HIDDEN)
        self.assertEqual(hiddens[0].shape[0], codes[0].shape[0])

    def test_hidden_states_are_the_final_layer(self):
        """The reported hidden states are each frame's final hidden state."""
        # Arrange
        model, prompt, input_ids = make_tiny_tts()
        recorded, restore = record_talker_past_hidden(model)

        # Act
        try:
            torch.manual_seed(1234)
            codes, hiddens = model.generate(input_ids=input_ids,
                                            voice_clone_prompt=prompt,
                                            **generate_kwargs())
        finally:
            restore()

        # Assert
        # The prefill's past_hidden is the first row; every decode frame adds
        # one row, and the outer drops the last one.
        self.assertEqual(hiddens[0].shape[0], codes[0].shape[0])
        expected = torch.cat([p.to(torch.float32) for p in recorded], dim=1)[:, :-1].squeeze(0)
        self.assertTrue(torch.equal(hiddens[0].to(torch.float32), expected))

    def test_output_hidden_states_false_is_honoured(self):
        """output_hidden_states=False skips the hidden states, not the codes."""
        # Arrange
        model, prompt, input_ids = make_tiny_tts()

        # Act
        torch.manual_seed(1234)
        codes_with, hiddens_with = model.generate(input_ids=input_ids,
                                                  voice_clone_prompt=prompt,
                                                  **generate_kwargs())
        torch.manual_seed(1234)
        codes_without, hiddens_without = model.generate(
            input_ids=input_ids, voice_clone_prompt=prompt,
            output_hidden_states=False, **generate_kwargs())

        # Assert
        self.assertTrue(torch.equal(codes_without[0], codes_with[0]))
        self.assertEqual(hiddens_without, [None])

    def test_talker_model_is_not_asked_for_layer_states(self):
        """The talker model forward never collects the per-layer states."""
        # Arrange
        model, prompt, input_ids = make_tiny_tts()
        recorded, restore = record_talker_requests(model)

        # Act
        try:
            torch.manual_seed(1234)
            model.generate(input_ids=input_ids, voice_clone_prompt=prompt,
                           **generate_kwargs())
        finally:
            restore()

        # Assert
        self.assertGreater(len(recorded), 0)
        for request in recorded:
            self.assertFalse(request)

    def test_direct_forward_reports_the_final_hidden(self):
        """A direct call reports a tuple whose last element is the final hidden.

        `finetuning/sft_12hz.py` reads `outputs.hidden_states[0][-1]`, which is
        the final hidden state under both the old and the new report.
        """
        # Arrange
        model, _, _ = make_tiny_tts()
        talker = model.talker
        embeds = torch.randn(1, 6, TALKER_HIDDEN, dtype=torch.bfloat16)

        # Act
        outputs = talker(
            inputs_embeds=embeds,
            attention_mask=torch.ones(1, 6, dtype=torch.long),
            output_hidden_states=True,
            use_cache=False,
        )

        # Assert
        final_hidden = outputs.hidden_states[0][-1]
        self.assertEqual(tuple(final_hidden.shape), (1, 6, TALKER_HIDDEN))
        self.assertTrue(torch.equal(final_hidden[:, -1:, :], outputs.past_hidden))
        self.assertIsNone(outputs.hidden_states[1])


if __name__ == "__main__":
    unittest.main()
