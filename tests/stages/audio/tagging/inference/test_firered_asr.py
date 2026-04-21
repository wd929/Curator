# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

import pytest

from nemo_curator.stages.audio.tagging.inference.firered_asr import FireredASRStage
from nemo_curator.tasks import AudioTask


class _DummyFireRedModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, uttids: list[str], wav_paths: list[str]) -> list[dict[str, str]]:
        self.calls.append({"uttids": uttids, "wav_paths": wav_paths})
        return [{"text": f"decoded_{idx}"} for idx, _ in enumerate(wav_paths)]


class TestFireredASRStage:
    def test_process_segments_transcribes_each_segment_path(self) -> None:
        stage = FireredASRStage(infer_segment_only=True, model_path="/tmp/firered_model")
        dummy_model = _DummyFireRedModel()
        stage._asr_model = dummy_model

        tasks = [
            AudioTask(
                data={
                    "segments": [
                        {"start": 0.0, "end": 1.0, "resampled_audio_filepath": "/tmp/seg_0.wav"},
                        {"start": 1.0, "end": 2.0, "resampled_audio_filepath": "/tmp/seg_1.wav"},
                    ]
                }
            )
        ]

        results = stage.process_batch(tasks)

        assert len(results) == 1
        segments = results[0].data["segments"]
        assert segments[0]["text"] == "decoded_0"
        assert segments[1]["text"] == "decoded_1"
        assert len(dummy_model.calls) == 1
        assert dummy_model.calls[0]["wav_paths"] == ["/tmp/seg_0.wav", "/tmp/seg_1.wav"]
        assert dummy_model.calls[0]["uttids"] == ["utt_0", "utt_1"]
        assert stage._asr_config.aed_length_penalty == 0.6
        assert stage._asr_config.decode_max_len == 0

    def test_requires_model_path(self) -> None:
        with pytest.raises(ValueError, match="model_path must be provided"):
            FireredASRStage(model_path="")

    def test_need_space_adds_spaces_between_chinese_chars(self) -> None:
        class _ChineseDummyModel:
            def transcribe(self, _uttids: list[str], _wav_paths: list[str]) -> list[dict[str, str]]:
                return [{"text": "你好world测试"}]

        stage_with_space = FireredASRStage(model_path="/tmp/firered_model", need_space=True)
        stage_with_space._asr_model = _ChineseDummyModel()
        spaced_tasks = [
            AudioTask(data={"segments": [{"start": 0.0, "end": 1.0, "resampled_audio_filepath": "/tmp/seg.wav"}]})
        ]
        spaced_results = stage_with_space.process_batch(spaced_tasks)
        assert spaced_results[0].data["segments"][0]["text"] == "你 好world测 试"

        stage_without_space = FireredASRStage(model_path="/tmp/firered_model", need_space=False)
        stage_without_space._asr_model = _ChineseDummyModel()
        unspaced_tasks = [
            AudioTask(data={"segments": [{"start": 0.0, "end": 1.0, "resampled_audio_filepath": "/tmp/seg.wav"}]})
        ]
        unspaced_results = stage_without_space.process_batch(unspaced_tasks)
        assert unspaced_results[0].data["segments"][0]["text"] == "你好world测试"
