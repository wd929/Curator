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

"""
FireRed ASR stage.

Runs FireRed ASR2 inference for tagging pipeline entries and supports
segment-level transcription.
"""

import time
from dataclasses import dataclass, field
import re
from typing import Any

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from loguru import logger

from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import BaseASRProcessorStage
from nemo_curator.tasks import AudioTask


@dataclass
class FireredASRStage(BaseASRProcessorStage):
    """
    FireRed ASR2-AED inference stage for audio tagging.

    This stage transcribes either split audio files or per-segment audio files
    using FireRed ASR2-AED and writes predicted text back to manifest entries.
    """

    model_path: str = "models/firered_asr2_aed"
    transcribe_batch_size: int = 8
    infer_segment_only: bool = True
    segments_key: str = "segments"
    text_key: str = "text"
    need_space: bool = False
    name: str = "FireredASR"
    use_half: bool = False
    return_timestamp: bool = False
    beam_size: int = 3
    nbest: int = 1
    decode_max_len: int = 0
    softmax_smoothing: float = 1.25
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0

    _asr_model: Any = field(default=None, repr=False)
    _asr_config: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model_path:
            msg = "model_path must be provided for FireredASRStage"
            raise ValueError(msg)
        self._asr_config = self._build_asr_config()

    def _build_asr_config(self) -> FireRedAsr2Config:
        return FireRedAsr2Config(
            use_gpu=self._device == "cuda",
            use_half=self.use_half,
            beam_size=self.beam_size,
            nbest=self.nbest,
            decode_max_len=self.decode_max_len,
            softmax_smoothing=self.softmax_smoothing,
            aed_length_penalty=self.aed_length_penalty,
            eos_penalty=self.eos_penalty,
            return_timestamp=self.return_timestamp,
        )

    def _format_text(self, text: str) -> str:
        if not self.need_space:
            return text
        # Insert a space only when two adjacent CJK ideographs touch.
        return re.sub(r"(?<=[\u4e00-\u9fff])(?=[\u4e00-\u9fff])", " ", text)

    def _ensure_model_loaded(self) -> None:
        """Lazy-load FireRed ASR2-AED model from local model_path."""
        if self._asr_model is not None:
            return
        self._asr_model = FireRedAsr2.from_pretrained("aed", self.model_path, self._asr_config)

        logger.info(f"[{self.name}] Initialized ASR model on {self._device}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "FireredASRStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Process a batch of AudioTasks for ASR transcription."""
        if len(tasks) == 0:
            return []
        self._ensure_model_loaded()
        t0 = time.perf_counter()
        results = self.process_segments(tasks) if self.infer_segment_only else self.process_full_audio(tasks)

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "entries_processed": len(tasks),
            }
        )
        return results

    def process_full_audio(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Transcribe split audio paths and populate split metadata text fields."""
        entries = [task.data for task in tasks]
        all_paths: list[str] = []
        path_to_entry_and_split: list[tuple[int, int]] = []
        for entry_idx, _ in enumerate(entries):
            meta_entry = entries[entry_idx]
            split_filepaths = meta_entry.get("split_filepaths")
            if not isinstance(split_filepaths, list) or not split_filepaths:
                logger.warning(f"[{self.name}] Entry at index {entry_idx} has no split_filepaths, skipping.")
                continue
            for split_idx, path in enumerate(split_filepaths):
                all_paths.append(path)
                path_to_entry_and_split.append((entry_idx, split_idx))

        if not all_paths:
            return tasks

        texts: list[str] = []
        batch_size = max(1, self.transcribe_batch_size)
        for start_idx in range(0, len(all_paths), batch_size):
            chunk_paths = all_paths[start_idx : start_idx + batch_size]
            chunk_uttids = [f"utt_{idx}" for idx in range(start_idx, start_idx + len(chunk_paths))]
            results = self._asr_model.transcribe(chunk_uttids, chunk_paths)
            for item in results:
                text = str(item.get("text", "")) if isinstance(item, dict) else str(item)
                texts.append(self._format_text(text))

        for path_idx, text in enumerate(texts):
            entry_idx, split_idx = path_to_entry_and_split[path_idx]
            meta_entry = entries[entry_idx]

            split_metadata = meta_entry.get("split_metadata")
            if split_metadata and split_idx < len(split_metadata):
                split_metadata[split_idx][self.text_key] = text
            else:
                meta_entry[self.text_key] = text

        return tasks

    def process_segments(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Process entries in segment-only mode (infer per segment)."""
        entries = [task.data for task in tasks]
        if not entries:
            return []

        segment_metadata_list = self._prepare_segment_batch_with_metadata(
            entries,
            cut_audio_segments=False,
            segments_key=self.segments_key,
        )
        all_segments = [seg["resampled_audio_filepath"] for seg in segment_metadata_list]

        if len(all_segments) == 0:
            return tasks

        texts: list[str] = []
        batch_size = max(1, self.transcribe_batch_size)
        for start_idx in range(0, len(all_segments), batch_size):
            chunk_paths = all_segments[start_idx : start_idx + batch_size]
            chunk_uttids = [f"utt_{idx}" for idx in range(start_idx, start_idx + len(chunk_paths))]
            results = self._asr_model.transcribe(chunk_uttids, chunk_paths)
            for item in results:
                text = str(item.get("text", "")) if isinstance(item, dict) else str(item)
                texts.append(self._format_text(text))

        for segment_metadata, text in zip(segment_metadata_list, texts, strict=True):
            metadata_idx = segment_metadata["metadata_idx"]
            segment_idx = segment_metadata["segment_idx"]
            segment = entries[metadata_idx][self.segments_key][segment_idx]
            segment[self.text_key] = text

        return tasks
