import inspect
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from core.utils import except_handler, load_key, rprint

_PIPELINE_CACHE = {}
_PIPELINE_LOCK = threading.Lock()
_GENERATE_LOCK = threading.Lock()


def _safe_load_key(key: str, default):
    try:
        return load_key(key)
    except Exception:
        return default


def _to_float32_array(audio):
    if audio is None:
        return np.array([], dtype=np.float32)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        audio = audio.cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _extract_audio_from_item(item):
    # KPipeline.Result (newer Kokoro) exposes `.audio` directly.
    for attr in ("audio", "wav", "waveform", "samples"):
        if hasattr(item, attr):
            return _to_float32_array(getattr(item, attr))
    if isinstance(item, tuple):
        if not item:
            return np.array([], dtype=np.float32)
        # Prefer tensor/array payload if mixed metadata + audio tuple.
        for part in item:
            arr = _to_float32_array(part)
            if arr.size:
                return arr
        return np.array([], dtype=np.float32)
    if isinstance(item, dict):
        for key in ("audio", "wav", "waveform", "samples"):
            if key in item:
                return _to_float32_array(item[key])
        return np.array([], dtype=np.float32)
    return _to_float32_array(item)


def _collect_audio(generator):
    chunks = []
    for item in generator:
        arr = _extract_audio_from_item(item)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        raise ValueError("Kokoro generated empty audio.")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _load_pipeline(lang_code: str):
    try:
        from kokoro import KPipeline
    except ImportError as e:
        raise ImportError(
            "kokoro_tts requires optional package `kokoro`. "
            "Install with: python -m pip install kokoro soundfile"
        ) from e

    with _PIPELINE_LOCK:
        pipeline = _PIPELINE_CACHE.get(lang_code)
        if pipeline is None:
            pipeline = KPipeline(lang_code=lang_code)
            _PIPELINE_CACHE[lang_code] = pipeline
        return pipeline


def _build_generation_kwargs(pipeline, voice: str, speed: float, split_pattern: str):
    kwargs = {}
    try:
        sig = inspect.signature(pipeline)
        if "voice" in sig.parameters:
            kwargs["voice"] = voice
        if "speed" in sig.parameters:
            kwargs["speed"] = speed
        if "split_pattern" in sig.parameters:
            kwargs["split_pattern"] = split_pattern
    except Exception:
        kwargs = {"voice": voice, "speed": speed, "split_pattern": split_pattern}
    return kwargs


@except_handler("Failed to generate audio using Kokoro TTS", retry=1, delay=1)
def kokoro_tts(text: str, save_path: str) -> None:
    settings = _safe_load_key("kokoro_tts", {})
    lang_code = str(settings.get("lang_code", "a")).strip() or "a"
    voice = str(settings.get("voice", "af_heart")).strip() or "af_heart"
    speed = float(settings.get("speed", 1.0))
    split_pattern = str(settings.get("split_pattern", "\\n+"))
    sample_rate_cfg = int(settings.get("sample_rate", 24000))

    pipeline = _load_pipeline(lang_code)
    kwargs = _build_generation_kwargs(
        pipeline=pipeline,
        voice=voice,
        speed=speed,
        split_pattern=split_pattern,
    )

    with _GENERATE_LOCK:
        generator = pipeline(text, **kwargs)
        audio = _collect_audio(generator)

    sample_rate = int(getattr(pipeline, "sample_rate", sample_rate_cfg) or sample_rate_cfg)
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate)
    rprint(
        f"[green]Kokoro audio saved to {out_path} "
        f"(voice={voice}, lang_code={lang_code}, speed={speed:.2f})[/green]"
    )
