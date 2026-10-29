"""Tests for the MOSS-TTS backend's request shaping and output handling.

The real model needs Apple Silicon and multi-GB weights, so these drive
``MossTTSBackend.generate`` with a stand-in for the mlx-audio model and check
what reaches ``model.generate`` and what comes back to the caller.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.backends.moss_backend import MOSS_HF_REPOS, MossTTSBackend


class _FakeMossModel:
    sample_rate = 48000

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        for chunk in self._chunks:
            yield SimpleNamespace(audio=chunk)


def _backend_with(model) -> MossTTSBackend:
    backend = MossTTSBackend()
    backend.model = model
    backend.model_size = "4B"
    backend._current_model_size = "4B"
    return backend


@pytest.mark.asyncio
async def test_clones_with_transcript_in_continuation_mode(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"")
    model = _FakeMossModel([np.zeros((10, 1), dtype=np.float32)])
    backend = _backend_with(model)

    await backend.generate("Ahoj", {"ref_audio": str(ref), "ref_text": "Dobrý den"}, language="cs")

    _, kwargs = model.calls[0]
    assert kwargs["mode"] == "continuation"
    assert kwargs["ref_audio"] == str(ref)
    assert kwargs["ref_text"] == "Dobrý den"
    assert kwargs["language"] == "Czech"


@pytest.mark.asyncio
async def test_falls_back_to_generation_mode_without_transcript(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"")
    model = _FakeMossModel([np.zeros((10, 1), dtype=np.float32)])
    backend = _backend_with(model)

    await backend.generate("Ahoj", {"ref_audio": str(ref), "ref_text": "  "}, language="cs")

    _, kwargs = model.calls[0]
    assert kwargs["mode"] == "generation"
    assert "ref_text" not in kwargs


@pytest.mark.asyncio
async def test_missing_reference_audio_is_dropped():
    model = _FakeMossModel([np.zeros((10, 1), dtype=np.float32)])
    backend = _backend_with(model)

    await backend.generate("Ahoj", {"ref_audio": "/nonexistent.wav", "ref_text": "x"}, language="cs")

    _, kwargs = model.calls[0]
    assert "ref_audio" not in kwargs
    assert kwargs["mode"] == "generation"


@pytest.mark.asyncio
async def test_stereo_chunks_are_concatenated_and_downmixed():
    left = np.full((4, 1), 0.2, dtype=np.float32)
    stereo = np.concatenate([left, -left], axis=1)
    model = _FakeMossModel([stereo, stereo])
    backend = _backend_with(model)

    audio, sample_rate = await backend.generate("Ahoj", {}, language="cs")

    assert sample_rate == 48000
    assert audio.shape == (8,)
    assert np.allclose(audio, 0.0)


def test_unknown_size_falls_back_to_default_repos():
    backend = MossTTSBackend()
    assert backend._get_model_path("default") == MOSS_HF_REPOS["1.7B"][0]
