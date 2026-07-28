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

import nemo_curator.stages.audio.metrics.get_wer as get_wer_module
from nemo_curator.stages.audio.metrics.get_wer import (
    ComputeNormalizedWERMetricsStage,
    GetPairwiseWerStage,
    get_cer,
    get_charrate,
    get_wer,
    get_wordrate,
)
from nemo_curator.tasks import AudioTask


def test_get_wer_basic() -> None:
    assert get_wer("a b c", "a x c") == 33.33


def test_get_cer_basic() -> None:
    assert get_cer("abc", "axc") == 33.33


def test_rates() -> None:
    assert get_charrate("abcd", 2.0) == 2.0
    assert get_wordrate("a b c d", 2.0) == 2.0


def test_pairwise_wer_validate_input_valid() -> None:
    stage = GetPairwiseWerStage()
    assert stage.validate_input(AudioTask(data={"text": "a b c", "pred_text": "a x c"})) is True


def test_pairwise_wer_validate_input_missing_text() -> None:
    stage = GetPairwiseWerStage()
    assert stage.validate_input(AudioTask(data={"pred_text": "a x c"})) is False


def test_pairwise_wer_validate_input_missing_pred_text() -> None:
    stage = GetPairwiseWerStage()
    assert stage.validate_input(AudioTask(data={"text": "a b c"})) is False


def test_pairwise_wer_process_batch_raises_on_missing_text() -> None:
    stage = GetPairwiseWerStage()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"pred_text": "a x c"})])


def test_pairwise_wer_process_batch_raises_on_missing_pred_text() -> None:
    stage = GetPairwiseWerStage()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "a b c"})])


def test_pairwise_wer_stage() -> None:
    stage = GetPairwiseWerStage()
    entry = AudioTask(data={"text": "a b c", "pred_text": "a x c"})
    result = stage.process(entry)
    assert isinstance(result, AudioTask)
    assert result.data["wer"] == 33.33


def test_normalized_wer_setup_falls_back_for_unsupported_language(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnsupportedNormalizer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            msg = "Language ro has not been supported yet."
            raise NotImplementedError(msg)

    monkeypatch.setattr(get_wer_module, "NEMO_TEXT_PROCESSING_AVAILABLE", True)
    monkeypatch.setattr(get_wer_module, "Normalizer", UnsupportedNormalizer)

    stage = ComputeNormalizedWERMetricsStage(
        language="ro",
        hypothesis_text_key="text_canary",
        reference_text_key="text_whisper",
    )
    stage.setup()
    assert stage.normalizer is None

    task = AudioTask(
        data={
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text_canary": "a b c",
                    "text_whisper": "a x c",
                }
            ]
        }
    )
    result = stage.process(task)
    assert "wer" in result.data["segments"][0]["metrics"]


def test_normalized_wer_can_disable_nemo_tn(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnexpectedNormalizer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            msg = "Normalizer should not be constructed"
            raise AssertionError(msg)

    monkeypatch.setattr(get_wer_module, "NEMO_TEXT_PROCESSING_AVAILABLE", True)
    monkeypatch.setattr(get_wer_module, "Normalizer", UnexpectedNormalizer)

    stage = ComputeNormalizedWERMetricsStage(language="ro", use_nemo_tn=False)
    stage.setup()

    assert stage.normalizer is None
    assert stage.normalize_text("A  <unk>  B | C") == "A B C"


def test_pnc_chars_can_include_regex_special_characters() -> None:
    stage = ComputeNormalizedWERMetricsStage(use_nemo_tn=False, pnc_chars=".,?!;:%/[](){}\"'-\u2013\u2014")

    assert stage.strip_spaces_before_punctuations("word ? next ]") == "word? next]"
    assert stage.clean_text("A [B] - C% D/E", retain_pncs=False) == "a b c d e"
