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
Canary ASR stage.

Runs NeMo Canary inference for tagging pipeline entries and defaults to
Romanian transcription with ``source_lang=target_lang="ro"``.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import BaseASRProcessorStage
from nemo_curator.tasks import AudioTask

WordAlignment = dict[str, float | str | None]


class _CanaryModelProtocol(Protocol):
    def transcribe(self, **kwargs: object) -> list[object] | tuple[list[object], object]:
        """Transcribe a batch of audio files."""


@dataclass
class CanaryASRStage(BaseASRProcessorStage):
    """
    Canary ASR inference stage for audio tagging.

    This stage transcribes either split audio files or per-segment audio files
    using a NeMo Canary model and writes predicted text back to manifest entries.
    By default it performs Romanian ASR with ``source_lang="ro"`` and
    ``target_lang="ro"`` for ``nvidia/canary-1b-v2``.
    """

    model_name: str = "nvidia/canary-1b-v2"
    model_path: str | None = None
    cache_dir: str | None = None

    transcribe_batch_size: int = 8
    infer_segment_only: bool = True
    segments_key: str = "segments"
    text_key: str = "text"
    compute_timestamps: bool = False

    source_lang: str = "ro"
    target_lang: str | None = None
    pnc: bool = True
    itn: bool | None = None
    diarize: bool | None = None
    decodercontext: str | None = None
    emotion: str | None = None
    task: str | None = None
    beam_size: int | None = 1
    transcribe_kwargs: dict[str, object] = field(default_factory=dict)
    fail_on_error: bool = False

    name: str = "CanaryASR"
    _asr_model: _CanaryModelProtocol | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model_name and not self.model_path and self._asr_model is None:
            msg = "model_name or model_path must be provided for CanaryASRStage"
            raise ValueError(msg)
        self.source_lang = self.source_lang.strip()
        if not self.source_lang:
            msg = "source_lang must be provided for CanaryASRStage"
            raise ValueError(msg)
        self.target_lang = self.source_lang if self.target_lang is None else self.target_lang.strip()
        if not self.target_lang:
            msg = "target_lang must be provided for CanaryASRStage"
            raise ValueError(msg)
        if self.task not in {None, "asr", "ast"}:
            msg = f"task must be None, 'asr', or 'ast', got {self.task}"
            raise ValueError(msg)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Download model weights without loading the model into memory."""
        if self._asr_model is not None or self.model_path or os.path.isdir(self.model_name):
            return

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            msg = "nemo_toolkit[asr] is required for CanaryASRStage. Install nemo_curator[audio_common]."
            raise RuntimeError(msg) from e

        try:
            nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name, return_model_file=True)
        except Exception as e:
            msg = f"[{self.name}] Failed to download model {self.model_name}"
            raise RuntimeError(msg) from e

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Load the model to the configured device."""
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Lazy-load NeMo Canary model."""
        if self._asr_model is not None:
            return

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            msg = "nemo_toolkit[asr] is required for CanaryASRStage. Install nemo_curator[audio_common]."
            raise RuntimeError(msg) from e

        try:
            if self.model_path:
                self._asr_model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=self.model_path,
                    map_location=torch.device(self._device),
                )
            else:
                self._asr_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name,
                    map_location=torch.device(self._device),
                )
        except Exception as e:
            model_id = self.model_path or self.model_name
            msg = f"[{self.name}] Failed to load model {model_id}"
            raise RuntimeError(msg) from e

        self._configure_model()
        logger.info(f"[{self.name}] Initialized ASR model on {self._device}")

    def _configure_model(self) -> None:
        if self._asr_model is None:
            return

        if hasattr(self._asr_model, "to"):
            self._asr_model.to(self._device)
        if hasattr(self._asr_model, "eval"):
            self._asr_model.eval()

        if self.beam_size is None:
            return

        cfg = getattr(self._asr_model, "cfg", None)
        decoding_cfg = getattr(cfg, "decoding", None)
        if decoding_cfg is None or not hasattr(self._asr_model, "change_decoding_strategy"):
            return

        try:
            decoding_cfg.beam.beam_size = self.beam_size
            self._asr_model.change_decoding_strategy(decoding_cfg)
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(f"[{self.name}] Could not set Canary beam_size={self.beam_size}: {e}")

    def _transcribe_kwargs(self, batch_size: int) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "pnc": self.pnc,
        }
        if self.compute_timestamps:
            kwargs["timestamps"] = True
        if self.itn is not None:
            kwargs["itn"] = self.itn
        if self.diarize is not None:
            kwargs["diarize"] = self.diarize
        if self.decodercontext is not None:
            kwargs["decodercontext"] = self.decodercontext
        if self.emotion is not None:
            kwargs["emotion"] = self.emotion
        if self.task is not None:
            kwargs["task"] = self.task
        kwargs.update(self.transcribe_kwargs)
        return kwargs

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["duration", self.segments_key, "split_filepaths", "split_metadata"]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "CanaryASRStage only supports process_batch"
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

    def _transcribe_paths(self, paths: list[str]) -> list[tuple[str, list[WordAlignment]]]:
        if self._asr_model is None:
            msg = "Canary ASR model has not been loaded"
            raise RuntimeError(msg)

        results: list[tuple[str, list[WordAlignment]]] = []
        batch_size = max(1, self.transcribe_batch_size)
        for start_idx in range(0, len(paths), batch_size):
            chunk_paths = paths[start_idx : start_idx + batch_size]
            try:
                raw_outputs = self._asr_model.transcribe(
                    audio=chunk_paths,
                    **self._transcribe_kwargs(batch_size=len(chunk_paths)),
                )
                outputs = self._normalize_outputs(raw_outputs)
            except Exception as e:
                if self.fail_on_error:
                    msg = f"[{self.name}] Exception for paths: {chunk_paths}"
                    raise RuntimeError(msg) from e
                logger.error(f"[{self.name}] Exception for paths {chunk_paths}: {e}")
                outputs = [None] * len(chunk_paths)

            outputs = outputs[: len(chunk_paths)]
            for output in outputs:
                text = self._extract_text(output)
                words = self._extract_words(output) if self.compute_timestamps else []
                results.append((text, words))

            if len(outputs) < len(chunk_paths):
                results.extend([("", [])] * (len(chunk_paths) - len(outputs)))

        return results

    @staticmethod
    def _normalize_outputs(outputs: object) -> list[object]:
        if isinstance(outputs, tuple) and outputs:
            outputs = outputs[0]
        if outputs is None:
            return []
        if isinstance(outputs, list):
            return outputs
        return [outputs]

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

    def _extract_text(self, output: object) -> str:
        candidate = self._first_candidate(output)
        if candidate is None:
            return ""
        if isinstance(candidate, str):
            return candidate.strip()
        if isinstance(candidate, dict):
            return str(candidate.get("text", candidate.get("pred_text", ""))).strip()
        text = getattr(candidate, "text", None)
        if text is not None:
            return str(text).strip()
        return str(candidate).strip()

    def _extract_words(self, output: object) -> list[WordAlignment]:
        candidate = self._first_candidate(output)
        if candidate is None:
            return []

        timestamp = (
            candidate.get("timestamp") if isinstance(candidate, dict) else getattr(candidate, "timestamp", None)
        )
        if not isinstance(timestamp, dict):
            return []

        raw_words = timestamp.get("word") or timestamp.get("words") or []
        words: list[WordAlignment] = []
        for stamp in raw_words:
            if not isinstance(stamp, dict):
                continue
            word_text = str(stamp.get("word", stamp.get("text", ""))).strip()
            if not word_text:
                continue
            words.append(
                {
                    "word": word_text,
                    "start": self._round_timestamp(stamp.get("start", stamp.get("start_time"))),
                    "end": self._round_timestamp(stamp.get("end", stamp.get("end_time"))),
                    "confidence": self._round_confidence(
                        stamp.get("confidence", stamp.get("probability", stamp.get("score")))
                    ),
                }
            )
        return words

    @staticmethod
    def _first_candidate(output: object) -> object | None:
        if isinstance(output, (list, tuple)):
            if not output:
                return None
            return output[0]
        return output

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
