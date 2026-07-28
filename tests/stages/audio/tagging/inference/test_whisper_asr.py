# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from dataclasses import dataclass

import pytest

from nemo_curator.stages.audio.tagging.inference.whisper_asr import WhisperASRStage
from nemo_curator.tasks import AudioTask


@dataclass
class DummyWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class DummySegment:
    text: str
    words: list[DummyWord] | None = None


class DummyWhisperModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(self, audio: str, **kwargs: object) -> tuple[list[DummySegment], object]:
        self.calls.append((audio, kwargs))
        return (
            [
                DummySegment(" salut ", [DummyWord(" salut", 0.1, 0.4, 0.91)]),
                DummySegment("lume", [DummyWord(" lume", 0.5, 0.9, 0.82)]),
            ],
            object(),
        )


class TestWhisperASRStage:
    def test_process_segments_writes_text(self) -> None:
        model = DummyWhisperModel()
        stage = WhisperASRStage(language="ro", infer_segment_only=True, _asr_model=model)
        task = AudioTask(
            data={
                "segments": [
                    {
                        "start": 10.0,
                        "end": 12.0,
                        "resampled_audio_filepath": "seg.wav",
                    }
                ],
            }
        )

        results = stage.process_batch([task])

        segment = results[0].data["segments"][0]
        assert segment["text"] == "salut lume"
        assert "words" not in segment
        assert model.calls[0][0] == "seg.wav"
        assert model.calls[0][1]["language"] == "ro"
        assert model.calls[0][1]["task"] == "transcribe"
        assert model.calls[0][1]["condition_on_previous_text"] is False

    def test_process_segments_offsets_word_timestamps(self) -> None:
        model = DummyWhisperModel()
        stage = WhisperASRStage(
            language="ro",
            infer_segment_only=True,
            compute_timestamps=True,
            _asr_model=model,
        )
        task = AudioTask(
            data={
                "segments": [
                    {
                        "start": 10.0,
                        "end": 12.0,
                        "resampled_audio_filepath": "seg.wav",
                    }
                ],
            }
        )

        results = stage.process_batch([task])

        segment = results[0].data["segments"][0]
        assert segment["text"] == "salut lume"
        assert segment["words"] == [
            {"word": "salut", "start": 10.1, "end": 10.4, "confidence": 0.91},
            {"word": "lume", "start": 10.5, "end": 10.9, "confidence": 0.82},
        ]
        assert model.calls[0][1]["word_timestamps"] is True
        assert model.calls[0][1]["without_timestamps"] is False

    def test_process_full_audio_writes_split_metadata(self) -> None:
        model = DummyWhisperModel()
        stage = WhisperASRStage(language="ro", infer_segment_only=False, _asr_model=model)
        task = AudioTask(
            data={
                "split_filepaths": ["split.wav"],
                "split_metadata": [
                    {
                        "start": 0,
                        "end": 5,
                    }
                ],
            }
        )

        results = stage.process_batch([task])

        split = results[0].data["split_metadata"][0]
        assert split["text"] == "salut lume"
        assert model.calls[0][0] == "split.wav"

    def test_process_raises_not_implemented(self) -> None:
        stage = WhisperASRStage(model_name_or_path="large-v3", _asr_model=DummyWhisperModel())
        with pytest.raises(NotImplementedError, match="only supports process_batch"):
            stage.process(AudioTask(data={}))

    def test_post_init_requires_model_name_or_path(self) -> None:
        with pytest.raises(ValueError, match="model_name_or_path"):
            WhisperASRStage(model_name_or_path="")

    def test_post_init_validates_task(self) -> None:
        with pytest.raises(ValueError, match="task must be"):
            WhisperASRStage(task="detect")
