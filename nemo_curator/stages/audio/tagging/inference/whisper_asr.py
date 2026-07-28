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
Whisper ASR stage.

Runs faster-whisper inference for tagging pipeline entries and supports
segment-level transcription.
"""

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import BaseASRProcessorStage
from nemo_curator.tasks import AudioTask

WordAlignment = dict[str, float | str | None]


class _WhisperModelProtocol(Protocol):
    def transcribe(self, audio: str, **kwargs: object) -> tuple[Iterable[object], object]:
        """Transcribe one audio file."""


@dataclass
class WhisperASRStage(BaseASRProcessorStage):
    """
    Whisper inference stage for audio tagging.

    This stage transcribes either split audio files or per-segment audio files
    using faster-whisper and writes predicted text back to manifest entries.
    """

    model_name_or_path: str = "large-v3"
    transcribe_batch_size: int = 8
    infer_segment_only: bool = True
    segments_key: str = "segments"
    text_key: str = "text"

    language: str | None = None
    task: str = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    condition_on_previous_text: bool = False
    prompt_reset_on_temperature: float = 0.5
    initial_prompt: str | list[int] | None = None
    prefix: str | None = None
    suppress_blank: bool = True
    suppress_tokens: tuple[int, ...] | None = (-1,)
    without_timestamps: bool = True
    compute_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    multilingual: bool = False
    vad_filter: bool = False
    vad_parameters: dict[str, object] | None = None
    max_new_tokens: int | None = None
    chunk_length: int | None = None
    clip_timestamps: str | tuple[float, ...] = "0"
    hallucination_silence_threshold: float | None = None
    hotwords: str | None = None
    language_detection_threshold: float | None = 0.5
    language_detection_segments: int = 1

    compute_type: str = "default"
    cpu_threads: int = 0
    model_num_workers: int = 1
    device_index: int | list[int] = 0
    download_root: str | None = None
    local_files_only: bool = False
    revision: str | None = None
    use_auth_token: str | bool | None = None
    model_kwargs: dict[str, object] = field(default_factory=dict)
    fail_on_error: bool = False

    name: str = "WhisperASR"
    _asr_model: _WhisperModelProtocol | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            msg = "model_name_or_path must be provided for WhisperASRStage"
            raise ValueError(msg)
        if self.task not in {"transcribe", "translate"}:
            msg = f"task must be 'transcribe' or 'translate', got {self.task}"
            raise ValueError(msg)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Download model weights without loading the model into memory."""
        if self._asr_model is not None or os.path.isdir(self.model_name_or_path):
            return

        try:
            from faster_whisper.utils import download_model
        except ImportError as e:
            msg = "faster-whisper is required for WhisperASRStage. Install nemo_curator[audio_common]."
            raise RuntimeError(msg) from e

        try:
            download_model(
                self.model_name_or_path,
                cache_dir=self.download_root,
                local_files_only=self.local_files_only,
                revision=self.revision,
                use_auth_token=self.use_auth_token,
            )
        except Exception as e:
            msg = f"[{self.name}] Failed to download model {self.model_name_or_path}"
            raise RuntimeError(msg) from e

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Load the model to the configured device."""
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Lazy-load faster-whisper model."""
        if self._asr_model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            msg = "faster-whisper is required for WhisperASRStage. Install nemo_curator[audio_common]."
            raise RuntimeError(msg) from e

        loader_kwargs: dict[str, object] = {
            "device": self._device,
            "device_index": self.device_index,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "num_workers": self.model_num_workers,
            "download_root": self.download_root,
            "local_files_only": self.local_files_only,
            "revision": self.revision,
            "use_auth_token": self.use_auth_token,
        }
        loader_kwargs.update(self.model_kwargs)
        self._asr_model = WhisperModel(self.model_name_or_path, **loader_kwargs)

        logger.info(f"[{self.name}] Initialized ASR model on {self._device}")

    def _transcribe_kwargs(self) -> dict[str, object]:
        suppress_tokens = None if self.suppress_tokens is None else list(self.suppress_tokens)
        clip_timestamps = (
            list(self.clip_timestamps) if isinstance(self.clip_timestamps, tuple) else self.clip_timestamps
        )
        without_timestamps = False if self.compute_timestamps else self.without_timestamps

        kwargs: dict[str, object] = {
            "task": self.task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
            "prompt_reset_on_temperature": self.prompt_reset_on_temperature,
            "suppress_blank": self.suppress_blank,
            "suppress_tokens": suppress_tokens,
            "without_timestamps": without_timestamps,
            "max_initial_timestamp": self.max_initial_timestamp,
            "word_timestamps": self.compute_timestamps,
            "multilingual": self.multilingual,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters,
            "max_new_tokens": self.max_new_tokens,
            "chunk_length": self.chunk_length,
            "clip_timestamps": clip_timestamps,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "language_detection_threshold": self.language_detection_threshold,
            "language_detection_segments": self.language_detection_segments,
        }
        if self.language is not None:
            kwargs["language"] = self.language
        if self.initial_prompt is not None:
            kwargs["initial_prompt"] = self.initial_prompt
        if self.prefix is not None:
            kwargs["prefix"] = self.prefix
        if self.hotwords is not None:
            kwargs["hotwords"] = self.hotwords
        return kwargs

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "WhisperASRStage only supports process_batch"
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

    def _transcribe_one(self, audio_path: str) -> tuple[str, list[WordAlignment]]:
        if self._asr_model is None:
            msg = "Whisper ASR model has not been loaded"
            raise RuntimeError(msg)

        segments, _info = self._asr_model.transcribe(audio_path, **self._transcribe_kwargs())

        text_parts: list[str] = []
        words: list[WordAlignment] = []
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if text:
                text_parts.append(text)
            if self.compute_timestamps:
                words.extend(self._extract_words(segment))

        return " ".join(text_parts).strip(), words

    def _transcribe_paths(self, paths: list[str]) -> list[tuple[str, list[WordAlignment]]]:
        results: list[tuple[str, list[WordAlignment]]] = []
        batch_size = max(1, self.transcribe_batch_size)
        for start_idx in range(0, len(paths), batch_size):
            chunk_paths = paths[start_idx : start_idx + batch_size]
            for path in chunk_paths:
                try:
                    results.append(self._transcribe_one(path))
                except Exception as e:  # noqa: PERF203
                    if self.fail_on_error:
                        msg = f"[{self.name}] Exception for {path}"
                        raise RuntimeError(msg) from e
                    logger.error(f"[{self.name}] Exception for {path}: {e}")
                    results.append(("", []))
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

        transcriptions = self._transcribe_paths(all_paths)
        for path_idx, (text, words) in enumerate(transcriptions):
            entry_idx, split_idx = path_to_entry_and_split[path_idx]
            meta_entry = entries[entry_idx]

            split_metadata = meta_entry.get("split_metadata")
            if split_metadata and split_idx < len(split_metadata):
                split_metadata[split_idx][self.text_key] = text
                if self.compute_timestamps:
                    split_metadata[split_idx][self.words_key] = words
            else:
                meta_entry[self.text_key] = text
                if self.compute_timestamps:
                    meta_entry[self.words_key] = words

        return tasks

    def process_segments(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Process entries in segment-only mode."""
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

        transcriptions = self._transcribe_paths(all_segments)
        for segment_metadata, (text, words) in zip(segment_metadata_list, transcriptions, strict=True):
            metadata_idx = segment_metadata["metadata_idx"]
            segment_idx = segment_metadata["segment_idx"]
            segment = entries[metadata_idx][self.segments_key][segment_idx]
            segment[self.text_key] = text
            if self.compute_timestamps:
                segment[self.words_key] = self._offset_words(words, segment.get("start", 0.0))

        return tasks

    def _extract_words(self, segment: object) -> list[WordAlignment]:
        raw_words = getattr(segment, "words", None)
        if not raw_words:
            return []

        words: list[WordAlignment] = []
        for item in raw_words:
            word_text = str(getattr(item, "word", "")).strip()
            if not word_text:
                continue
            words.append(
                {
                    "word": word_text,
                    "start": self._round_timestamp(getattr(item, "start", None)),
                    "end": self._round_timestamp(getattr(item, "end", None)),
                    "confidence": self._round_confidence(getattr(item, "probability", None)),
                }
            )
        return words

    def _offset_words(self, words: list[WordAlignment], offset: object) -> list[WordAlignment]:
        try:
            offset_value = float(offset)
        except (TypeError, ValueError):
            offset_value = 0.0
        if offset_value == 0.0:
            return words

        offset_words: list[WordAlignment] = []
        for word in words:
            offset_word = dict(word)
            if offset_word["start"] is not None:
                offset_word["start"] = round(float(offset_word["start"]) + offset_value, 3)
            if offset_word["end"] is not None:
                offset_word["end"] = round(float(offset_word["end"]) + offset_value, 3)
            offset_words.append(offset_word)
        return offset_words

    @staticmethod
    def _round_timestamp(value: object) -> float | None:
        if value is None:
            return None
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _round_confidence(value: object) -> float | None:
        if value is None:
            return None
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None
