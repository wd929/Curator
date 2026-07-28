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

from nemo_curator.stages.audio.tagging.inference.canary_asr import CanaryASRStage
from nemo_curator.tasks import AudioTask


@dataclass
class DummyHypothesis:
    text: str
    timestamp: dict[str, list[dict[str, float | str]]] | None = None


class DummyCanaryModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, **kwargs: object) -> list[DummyHypothesis]:
        self.calls.append(kwargs)
        audio = kwargs["audio"]
        return [DummyHypothesis(f"text for {path}") for path in audio]


class DummyTimestampCanaryModel(DummyCanaryModel):
    def transcribe(self, **kwargs: object) -> list[DummyHypothesis]:
        self.calls.append(kwargs)
        return [
            DummyHypothesis(
                "salut lume",
                {
                    "word": [
                        {"word": "salut", "start": 0.1, "end": 0.4, "confidence": 0.91},
                        {"word": "lume", "start": 0.5, "end": 0.9, "confidence": 0.82},
                    ]
                },
            )
        ]


class TestCanaryASRStage:
    def test_process_segments_defaults_to_romanian_transcribe(self) -> None:
        model = DummyCanaryModel()
        stage = CanaryASRStage(infer_segment_only=True, _asr_model=model)
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
        assert segment["text"] == "text for seg.wav"
        assert model.calls[0]["audio"] == ["seg.wav"]
        assert model.calls[0]["source_lang"] == "ro"
        assert model.calls[0]["target_lang"] == "ro"
        assert model.calls[0]["pnc"] is True
        assert "task" not in model.calls[0]

    def test_process_segments_offsets_word_timestamps(self) -> None:
        model = DummyTimestampCanaryModel()
        stage = CanaryASRStage(infer_segment_only=True, compute_timestamps=True, _asr_model=model)
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
        assert model.calls[0]["timestamps"] is True

    def test_process_full_audio_writes_split_metadata(self) -> None:
        model = DummyCanaryModel()
        stage = CanaryASRStage(infer_segment_only=False, _asr_model=model)
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
        assert split["text"] == "text for split.wav"
        assert model.calls[0]["audio"] == ["split.wav"]

    def test_canary_v1_task_can_be_enabled(self) -> None:
        model = DummyCanaryModel()
        stage = CanaryASRStage(task="asr", _asr_model=model)
        task = AudioTask(
            data={
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "resampled_audio_filepath": "seg.wav",
                    }
                ],
            }
        )

        stage.process_batch([task])

        assert model.calls[0]["task"] == "asr"

    def test_process_raises_not_implemented(self) -> None:
        stage = CanaryASRStage(_asr_model=DummyCanaryModel())
        with pytest.raises(NotImplementedError, match="only supports process_batch"):
            stage.process(AudioTask(data={}))

    def test_post_init_validates_languages(self) -> None:
        with pytest.raises(ValueError, match="source_lang"):
            CanaryASRStage(source_lang="")
        with pytest.raises(ValueError, match="target_lang"):
            CanaryASRStage(target_lang="")

    def test_post_init_validates_task(self) -> None:
        with pytest.raises(ValueError, match="task must be"):
            CanaryASRStage(task="transcribe")
