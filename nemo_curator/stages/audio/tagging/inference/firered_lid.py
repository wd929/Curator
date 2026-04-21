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
FireRed LID stage.

Runs FireRed LID inference for tagging pipeline entries and supports
segment-level language identification.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig
from loguru import logger

from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import BaseASRProcessorStage
from nemo_curator.tasks import AudioTask


@dataclass
class FireredLIDStage(BaseASRProcessorStage):
    """FireRed LID inference stage for audio tagging."""

    model_path: str = "models/firered_lid"
    inference_batch_size: int = 16
    infer_segment_only: bool = True
    segments_key: str = "segments"
    lang_key: str = "lang"
    lang_confidence_key: str = "lang_confidence"
    name: str = "FireredLID"
    use_half: bool = False

    _lid_model: Any = field(default=None, repr=False)
    _lid_config: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model_path:
            msg = "model_path must be provided for FireredLIDStage"
            raise ValueError(msg)
        self._lid_config = self._build_lid_config()

    def _build_lid_config(self) -> FireRedLidConfig:
        return FireRedLidConfig(
            use_gpu=self._device == "cuda",
            use_half=self.use_half,
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy-load FireRed LID model from local model_path."""
        if self._lid_model is not None:
            return
        self._lid_model = FireRedLid.from_pretrained(self.model_path, self._lid_config)
        logger.info(f"[{self.name}] Initialized LID model on {self._device}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "FireredLIDStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Process a batch of AudioTasks for LID."""
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
        """Run LID for split audio paths and write lang/confidence."""
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

        results: list[dict[str, Any]] = []
        batch_size = max(1, self.inference_batch_size)
        for start_idx in range(0, len(all_paths), batch_size):
            chunk_paths = all_paths[start_idx : start_idx + batch_size]
            chunk_uttids = [f"utt_{idx}" for idx in range(start_idx, start_idx + len(chunk_paths))]
            chunk_results = self._lid_model.process(chunk_uttids, chunk_paths)
            results.extend(chunk_results)

        for path_idx, result in enumerate(results):
            entry_idx, split_idx = path_to_entry_and_split[path_idx]
            meta_entry = entries[entry_idx]
            lang = str(result.get("lang", ""))
            confidence = result.get("confidence")

            split_metadata = meta_entry.get("split_metadata")
            if split_metadata and split_idx < len(split_metadata):
                split_metadata[split_idx][self.lang_key] = lang
                split_metadata[split_idx][self.lang_confidence_key] = confidence
            else:
                meta_entry[self.lang_key] = lang
                meta_entry[self.lang_confidence_key] = confidence

        return tasks

    def process_segments(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run LID for per-segment audio paths and write lang/confidence."""
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

        results: list[dict[str, Any]] = []
        batch_size = max(1, self.inference_batch_size)
        for start_idx in range(0, len(all_segments), batch_size):
            chunk_paths = all_segments[start_idx : start_idx + batch_size]
            chunk_uttids = [f"utt_{idx}" for idx in range(start_idx, start_idx + len(chunk_paths))]
            chunk_results = self._lid_model.process(chunk_uttids, chunk_paths)
            results.extend(chunk_results)

        for segment_metadata, result in zip(segment_metadata_list, results, strict=True):
            metadata_idx = segment_metadata["metadata_idx"]
            segment_idx = segment_metadata["segment_idx"]
            segment = entries[metadata_idx][self.segments_key][segment_idx]
            segment[self.lang_key] = str(result.get("lang", ""))
            segment[self.lang_confidence_key] = result.get("confidence")

        return tasks
