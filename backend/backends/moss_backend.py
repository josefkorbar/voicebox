"""
MOSS-TTS backend, MLX (Metal) only.

Wraps the OpenMOSS MOSS-TTS family through mlx-audio's ``moss_tts`` model:
zero-shot voice cloning in 31 languages (Czech, Hungarian, Romanian, Persian
and others that no other Voicebox engine covers). Apache-2.0 code and weights.

Three checkpoints, exposed as model sizes:

  - ``1.7B``: MOSS-TTS-Local-Transformer (Qwen3-1.7B backbone, 24 kHz)
  - ``4B``:   MOSS-TTS-Local-Transformer-v1.5 (48 kHz stereo, language tags)
  - ``8B``:   MOSS-TTS-v1.5 (MossTTSDelay, best speaker similarity)

Each checkpoint pulls its audio tokenizer from a second repo at first use
(v1.5 local uses MOSS-Audio-Tokenizer-v2), so a model only counts as
downloaded once both are cached. The tokenizer is loaded eagerly inside
``model_load_progress`` so its multi-GB download shows up as progress instead
of stalling the first generation.

The upstream PyTorch path needs transformers 5.x and a CUDA build, so there is
no PyTorch fallback here: the engine only loads on the MLX backend.

Cloning modes: with a reference transcript the backend uses mlx-audio's
``continuation`` mode (reference audio + its text as a prefix), which holds
the speaker's voice more closely than ``generation`` mode (audio only).
"""

import logging
from pathlib import Path

import numpy as np

from ..utils.platform_detect import get_backend_type
from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    is_model_cached,
    model_load_progress,
)

logger = logging.getLogger(__name__)

MOSS_AUDIO_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
MOSS_AUDIO_TOKENIZER_V2_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"

# model_size -> (model repo, audio tokenizer repo)
MOSS_HF_REPOS = {
    "1.7B": ("OpenMOSS-Team/MOSS-TTS-Local-Transformer", MOSS_AUDIO_TOKENIZER_REPO),
    "4B": ("OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", MOSS_AUDIO_TOKENIZER_V2_REPO),
    "8B": ("OpenMOSS-Team/MOSS-TTS-v1.5", MOSS_AUDIO_TOKENIZER_REPO),
}
DEFAULT_MODEL_SIZE = "1.7B"

# MOSS takes the language as an English name in its prompt template.
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

# Upper bound on generated audio frames per request. MOSS runs at 12.5 frames
# per second, so this caps one chunk at ~2 minutes; the generation service
# already splits long text into shorter chunks.
MAX_TOKENS = 1500


def model_name_for_size(model_size: str) -> str:
    return f"moss-tts-{model_size}"


def _repos_for_size(model_size: str) -> tuple[str, str]:
    return MOSS_HF_REPOS.get(model_size, MOSS_HF_REPOS[DEFAULT_MODEL_SIZE])


class MossTTSBackend:
    """MOSS-TTS zero-shot voice cloning on MLX/Metal."""

    def __init__(self):
        self.model = None
        self.model_size = DEFAULT_MODEL_SIZE
        self._current_model_size = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = DEFAULT_MODEL_SIZE) -> str:
        return _repos_for_size(model_size)[0]

    def _is_model_cached(self, model_size: str = DEFAULT_MODEL_SIZE) -> bool:
        model_repo, tokenizer_repo = _repos_for_size(model_size)
        return is_model_cached(model_repo) and is_model_cached(tokenizer_repo)

    async def load_model(self, model_size: str = DEFAULT_MODEL_SIZE) -> None:
        """Load a MOSS-TTS checkpoint, swapping out a different size if loaded."""
        if model_size not in MOSS_HF_REPOS:
            model_size = DEFAULT_MODEL_SIZE
        if self.model is not None and self._current_model_size == model_size:
            return
        if get_backend_type() != "mlx":
            raise RuntimeError("MOSS-TTS is only available on Apple Silicon (MLX backend).")

        from .mlx_backend import _run_on_mlx_thread

        # MLX streams are thread-local: load and generate must share the
        # single MLX worker thread (see mlx_backend, issue #699).
        await _run_on_mlx_thread(self._reload_sync, model_size)

    def _reload_sync(self, model_size: str) -> None:
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()
        if self.model is None:
            self._load_model_sync(model_size)

    def _load_model_sync(self, model_size: str) -> None:
        model_repo, _ = _repos_for_size(model_size)
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name_for_size(model_size), is_cached):
            from mlx_audio.tts import load  # lazy: heavy import

            logger.info("Loading MOSS-TTS %s on MLX (Metal)...", model_size)
            model = load(model_repo, lazy=True)
            # Pull the separate audio tokenizer repo now rather than on the
            # first generate() call.
            model._ensure_audio_tokenizer()

        self.model = model
        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MOSS-TTS %s loaded successfully", model_size)

    def unload_model(self) -> None:
        if self.model is None:
            return

        del self.model
        self.model = None
        self._current_model_size = None
        try:
            import mlx.core as mx  # lazy: heavy import

            mx.clear_cache()
        except Exception:
            logger.debug("mlx cache not cleared", exc_info=True)
        logger.info("MOSS-TTS unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """MOSS encodes the reference at generation time, so store the path."""
        return {"ref_audio": str(audio_path), "ref_text": reference_text}, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio with MOSS-TTS.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ref_audio path and ref_text
            language: Voicebox two-letter code, passed to MOSS as a language name
            seed: Random seed for reproducibility
            instruct: Unused (protocol compatibility)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        from .mlx_backend import _run_on_mlx_thread

        model_size = self.model_size
        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("Reference audio not found: %s", ref_audio)
            ref_audio = None
        ref_text = (voice_prompt.get("ref_text") or "").strip() or None
        mode = "continuation" if ref_audio and ref_text else "generation"
        language_name = LANGUAGE_NAMES.get(language)

        def _generate_sync():
            import mlx.core as mx  # lazy: heavy import

            # Load (if needed) and generate as one worker submission so an
            # unload cannot land between the two.
            if self.model is None:
                self._load_model_sync(model_size)
            model = self.model

            if seed is not None:
                mx.random.seed(seed)

            logger.info("[MOSS-TTS] Generating: size=%s lang=%s mode=%s", model_size, language_name, mode)

            kwargs = {"max_tokens": MAX_TOKENS, "mode": mode, "verbose": False}
            if ref_audio:
                kwargs["ref_audio"] = ref_audio
                if ref_text:
                    kwargs["ref_text"] = ref_text
            if language_name:
                kwargs["language"] = language_name

            chunks = [np.asarray(result.audio, dtype=np.float32) for result in model.generate(text, **kwargs)]
            if not chunks:
                return np.zeros(0, dtype=np.float32), int(model.sample_rate)

            audio = np.concatenate(chunks, axis=0)
            # The audio tokenizer decodes to (samples, channels): one channel
            # for 1.7B/8B, 48 kHz stereo for v1.5 local. Voicebox works in mono.
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), int(model.sample_rate)

        return await _run_on_mlx_thread(_generate_sync)
