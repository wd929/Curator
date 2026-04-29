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

import pytest

from nemo_curator.stages.audio.tagging.inference.firered_asr import FireredASRStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


class TestFireredASRStage:
    @staticmethod
    def _single_segment_task() -> list[AudioTask]:
        return [
            AudioTask(
                data={
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.32,
                            "resampled_audio_filepath": (
                                "/FireRedASR2S/assets/hello_zh.wav"
                            ),
                        }
                    ]
                }
            )
        ]

    def test_single_audio_inference_real_model_cpu(self) -> None:
        """Smoke test FireRed ASR2-AED on one real audio segment (CPU)."""
        stage = FireredASRStage(
            infer_segment_only=True,
            model_path="/FireRedASR2S/pretrained_models/FireRedASR2-AED",
            resources=Resources(cpus=1.0),
        )
        results = stage.process_batch(self._single_segment_task())
        assert len(results) == 1
        segments = results[0].data["segments"]
        assert len(segments) == 1
        assert "text" in segments[0]
        assert isinstance(segments[0]["text"], str)
        assert segments[0]["text"] != ""

    @pytest.mark.gpu
    def test_single_audio_inference_real_model_gpu(self) -> None:
        """Smoke test FireRed ASR2-AED on one real audio segment (GPU)."""
        stage = FireredASRStage(
            infer_segment_only=True,
            model_path="/FireRedASR2S/pretrained_models/FireRedASR2-AED",
            resources=Resources(gpus=1.0),
        )
        results = stage.process_batch(self._single_segment_task())
        assert len(results) == 1
        segments = results[0].data["segments"]
        assert len(segments) == 1
        assert "text" in segments[0]
        assert isinstance(segments[0]["text"], str)
        assert segments[0]["text"] != ""
