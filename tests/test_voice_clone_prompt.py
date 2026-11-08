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
"""Check when `create_voice_clone_prompt` runs the codec encoder.

In x-vector-only mode the codec codes are discarded, so the encoder call is
skipped. A mixed batch still encodes every reference: filtering the batch would
change the batch composition the encoder sees and therefore the codes of the
ICL items. These tests use a recording stand-in for the speech tokenizer and the
speaker encoder, so they run on any device without model weights.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

SAMPLE_RATE = 24000
WAV_LEN = 2400


class RecordingSpeechTokenizer:
    """A stand-in tokenizer that records the batches it encodes."""

    def __init__(self):
        self.encode_calls = []

    def encode(self, wavs, sr):
        if not isinstance(wavs, list):
            wavs = [wavs]
        self.encode_calls.append(([len(wav) for wav in wavs], sr))
        # The code of item i is a 4x1 tensor filled with i
        codes = [torch.full((4, 1), float(i)) for i in range(len(wavs))]
        return SimpleNamespace(audio_codes=codes)


class RecordingSpeakerEncoderModel:
    """The parts of the model `create_voice_clone_prompt` reads."""

    def __init__(self):
        self.tts_model_type = "base"
        self.tokenizer_type = "qwen3-tts-tokenizer-v2"
        self.tts_model_size = "1.7b"
        self.speaker_encoder_sample_rate = SAMPLE_RATE
        self.speech_tokenizer = RecordingSpeechTokenizer()
        self.embedding_calls = 0
        # `Qwen3TTSModel.__init__` reads the device from the parameters
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._parameter])

    def extract_speaker_embedding(self, audio, sr):
        assert sr == SAMPLE_RATE
        self.embedding_calls += 1
        return torch.ones(8)


def make_tts():
    model = RecordingSpeakerEncoderModel()
    return Qwen3TTSModel(model=model, processor=None), model


def make_wavs(count, sr=SAMPLE_RATE):
    return [(np.zeros(WAV_LEN, dtype=np.float32), sr) for _ in range(count)]


class CreateVoiceClonePromptTest(unittest.TestCase):
    def test_x_vector_only_skips_the_encoder(self):
        """An all x-vector-only batch never encodes the reference audio."""
        # Arrange
        tts, model = make_tts()

        # Act
        items = tts.create_voice_clone_prompt(
            ref_audio=make_wavs(1), ref_text="ref", x_vector_only_mode=True)

        # Assert
        self.assertEqual(model.speech_tokenizer.encode_calls, [])
        self.assertEqual(model.embedding_calls, 1)
        self.assertIsNone(items[0].ref_code)
        self.assertTrue(items[0].x_vector_only_mode)
        self.assertFalse(items[0].icl_mode)

    def test_icl_encodes_the_batch_once(self):
        """An all ICL batch encodes the references as one batch."""
        # Arrange
        tts, model = make_tts()

        # Act
        items = tts.create_voice_clone_prompt(
            ref_audio=make_wavs(2), ref_text=["a", "b"], x_vector_only_mode=False)

        # Assert
        self.assertEqual(model.speech_tokenizer.encode_calls,
                         [([WAV_LEN, WAV_LEN], SAMPLE_RATE)])
        self.assertEqual(model.embedding_calls, 2)
        for i, item in enumerate(items):
            self.assertIsNotNone(item.ref_code)
            self.assertTrue(item.icl_mode)
            self.assertTrue(torch.equal(item.ref_code, torch.full((4, 1), float(i))))

    def test_mixed_batch_encodes_every_reference(self):
        """A mixed batch keeps the encoder batch composition unchanged."""
        # Arrange
        tts, model = make_tts()

        # Act
        items = tts.create_voice_clone_prompt(
            ref_audio=make_wavs(2), ref_text=["a", "b"], x_vector_only_mode=[True, False])

        # Assert
        self.assertEqual(model.speech_tokenizer.encode_calls,
                         [([WAV_LEN, WAV_LEN], SAMPLE_RATE)])
        self.assertIsNone(items[0].ref_code)
        # The ICL item keeps the code of its position in the full batch
        self.assertTrue(torch.equal(items[1].ref_code, torch.full((4, 1), 1.0)))

    def test_mixed_batch_with_different_rates_encodes_each_reference(self):
        """Different sample rates keep the per-item encoder path."""
        # Arrange
        tts, model = make_tts()
        wavs = [(np.zeros(WAV_LEN, dtype=np.float32), SAMPLE_RATE),
                (np.zeros(WAV_LEN // 2, dtype=np.float32), 16000)]

        # Act
        items = tts.create_voice_clone_prompt(
            ref_audio=wavs, ref_text=["a", "b"], x_vector_only_mode=[True, False])

        # Assert
        self.assertEqual(model.speech_tokenizer.encode_calls,
                         [([WAV_LEN], SAMPLE_RATE), ([WAV_LEN // 2], 16000)])
        self.assertIsNone(items[0].ref_code)
        self.assertTrue(torch.equal(items[1].ref_code, torch.zeros(4, 1)))


if __name__ == "__main__":
    unittest.main()
