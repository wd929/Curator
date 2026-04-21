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

from nemo_curator.stages.audio.tagging.inference.firered_lid import FireredLIDStage
from nemo_curator.tasks import AudioTask


class _DummyFireRedLidModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process(self, uttids: list[str], wav_paths: list[str]) -> list[dict[str, Any]]:
        self.calls.append({"uttids": uttids, "wav_paths": wav_paths})
        return [{"lang": "en", "confidence": 0.99} for _ in wav_paths]


class TestFireredLIDStage:
    def test_process_segments_runs_lid_for_each_segment(self) -> None:
        stage = FireredLIDStage(infer_segment_only=True, model_path="/tmp/firered_lid_model")
        dummy_model = _DummyFireRedLidModel()
        stage._lid_model = dummy_model

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
        assert segments[0]["lang"] == "en"
        assert segments[0]["lang_confidence"] == 0.99
        assert segments[1]["lang"] == "en"
        assert segments[1]["lang_confidence"] == 0.99
        assert len(dummy_model.calls) == 1
        assert dummy_model.calls[0]["wav_paths"] == ["/tmp/seg_0.wav", "/tmp/seg_1.wav"]
        assert dummy_model.calls[0]["uttids"] == ["utt_0", "utt_1"]

    def test_requires_model_path(self) -> None:
        with pytest.raises(ValueError, match="model_path must be provided"):
            FireredLIDStage(model_path="")
