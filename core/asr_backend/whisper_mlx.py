import inspect
import json
import os
import platform
import time
from typing import Optional

from core.utils import except_handler, load_key, rprint, update_key
from core.asr_backend.audio_segment import load_audio_segment

MODEL_CANDIDATES = {
    "large-v3": [
        "mlx-community/whisper-large-v3-mlx",
        "mlx-community/whisper-large-v3",
        "mlx-community/whisper-large-v3-fp16",
    ],
    "large-v3-turbo": [
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/whisper-large-v3-turbo-fp16",
        "mlx-community/whisper-large-v3-turbo-4bit",
    ],
}
OUTPUT_LOG_DIR = "output/log"


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _resolve_model_candidates(model_name: str):
    candidates = MODEL_CANDIDATES.get(model_name)
    if candidates:
        return candidates
    return [model_name]


def _slice_audio(audio_path: str, start: float = None, end: float = None, sr: int = 16000):
    start = 0.0 if start is None else max(float(start), 0.0)
    end = None if end is None else max(float(end), start)
    audio = load_audio_segment(audio_path, start=start, end=end, sample_rate=sr)
    if end is None:
        end = start + (len(audio) / sr)
    return audio, start, end


def _transcribe_with_model(transcribe, audio_segment, model_name: str, language: Optional[str]):
    language_kwargs = {"language": language} if language else {}

    # Try a signature-aware call first.
    try:
        sig = inspect.signature(transcribe)
        kwargs = {}
        if "word_timestamps" in sig.parameters:
            kwargs["word_timestamps"] = True
        if language and "language" in sig.parameters:
            kwargs["language"] = language
        if "path_or_hf_repo" in sig.parameters:
            kwargs["path_or_hf_repo"] = model_name
        elif "model_path_or_name" in sig.parameters:
            kwargs["model_path_or_name"] = model_name
        elif "model" in sig.parameters:
            kwargs["model"] = model_name
        elif "model_name" in sig.parameters:
            kwargs["model_name"] = model_name
        return transcribe(audio_segment, **kwargs)
    except TypeError:
        pass

    # Fallback calls for different mlx_whisper versions.
    attempts = [
        dict(path_or_hf_repo=model_name, word_timestamps=True, **language_kwargs),
        dict(model_path_or_name=model_name, word_timestamps=True, **language_kwargs),
        dict(model=model_name, word_timestamps=True, **language_kwargs),
        dict(model_name=model_name, word_timestamps=True, **language_kwargs),
    ]

    last_err = None
    for kwargs in attempts:
        try:
            return transcribe(audio_segment, **kwargs)
        except TypeError as e:
            last_err = e

    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to call mlx_whisper.transcribe with compatible signature")


def _call_mlx_transcribe(audio_segment, model_candidates, language: Optional[str]):
    try:
        import mlx_whisper
    except ImportError as e:
        raise ImportError(
            "whispermlx runtime selected but `mlx_whisper` is not installed. "
            "Install it on Apple Silicon with: pip install mlx-whisper"
        ) from e

    transcribe = mlx_whisper.transcribe
    errors = []
    for model_name in model_candidates:
        try:
            result = _transcribe_with_model(
                transcribe=transcribe,
                audio_segment=audio_segment,
                model_name=model_name,
                language=language,
            )
            return result, model_name
        except Exception as e:
            err_text = str(e)
            errors.append(f"{model_name}: {err_text}")

            # Common model-id failure should try next candidate.
            if "Repository Not Found" in err_text or "401 Client Error" in err_text:
                continue
            # Other errors may still be model-specific; keep trying candidates.
            continue

    raise RuntimeError(
        "All WhisperMLX model candidates failed:\n- " + "\n- ".join(errors)
    )


def _normalize_to_whisperx_format(result: dict, time_offset: float) -> dict:
    normalized_segments = []
    segments = result.get("segments", [])

    for seg in segments:
        if not isinstance(seg, dict):
            continue

        seg_text = str(seg.get("text", "")).strip()
        seg_words = []

        raw_words = seg.get("words", [])
        if isinstance(raw_words, list) and raw_words:
            for item in raw_words:
                if not isinstance(item, dict):
                    continue
                word_text = str(item.get("word") or item.get("text") or "").strip()
                if not word_text:
                    continue
                word_obj = {"word": word_text}
                if item.get("start") is not None:
                    word_obj["start"] = float(item["start"]) + time_offset
                if item.get("end") is not None:
                    word_obj["end"] = float(item["end"]) + time_offset
                seg_words.append(word_obj)

        # Fallback: no word timestamps, split by whitespace and estimate inside segment.
        if not seg_words and seg_text:
            tokens = seg_text.split()
            seg_start = seg.get("start")
            seg_end = seg.get("end")

            if tokens and seg_start is not None and seg_end is not None:
                seg_start = float(seg_start)
                seg_end = float(seg_end)
                duration = max(seg_end - seg_start, 0.0)
                step = duration / len(tokens) if tokens else 0.0
                for i, token in enumerate(tokens):
                    word_start = seg_start + i * step
                    word_end = seg_start + (i + 1) * step
                    seg_words.append(
                        {
                            "word": token,
                            "start": word_start + time_offset,
                            "end": word_end + time_offset,
                        }
                    )
            else:
                seg_words = [{"word": token} for token in tokens]

        if not seg_words:
            continue

        norm_seg = {
            "text": seg_text,
            "words": seg_words,
        }
        if seg.get("start") is not None:
            norm_seg["start"] = float(seg["start"]) + time_offset
        if seg.get("end") is not None:
            norm_seg["end"] = float(seg["end"]) + time_offset
        normalized_segments.append(norm_seg)

    return {"segments": normalized_segments}


def _segment_log_file(start: float, end: float) -> str:
    start_tag = "full" if start is None else f"{float(start):.3f}"
    end_tag = "full" if end is None else f"{float(end):.3f}"
    return f"{OUTPUT_LOG_DIR}/whispermlx_{start_tag}_{end_tag}.json"


@except_handler("WhisperMLX processing error:")
def transcribe_audio_mlx(
    raw_audio_file: str,
    vocal_audio_file: str,
    start: float = None,
    end: float = None,
):
    del vocal_audio_file  # WhisperMLX path transcribes directly from raw audio segment.

    if not _is_apple_silicon():
        raise EnvironmentError(
            "whispermlx runtime is only supported on Apple Silicon (Darwin arm64)."
        )

    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    log_file = _segment_log_file(start, end)
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)

    whisper_language = load_key("whisper.language")
    language = None if whisper_language == "auto" else whisper_language
    model_name = load_key("whisper.model")
    model_candidates = _resolve_model_candidates(model_name)

    audio_segment, start, end = _slice_audio(raw_audio_file, start, end, sr=16000)
    if len(audio_segment) == 0:
        raise ValueError("Audio segment is empty for WhisperMLX transcription")

    rprint(
        f"[cyan]🎤 Transcribing with WhisperMLX model candidates: "
        f"<{', '.join(model_candidates)}> segment {start:.2f}s -> {end:.2f}s[/cyan]"
    )
    start_time = time.time()
    result, selected_model = _call_mlx_transcribe(audio_segment, model_candidates, language)

    detected_language = result.get("language") or whisper_language
    if detected_language and detected_language != "auto":
        update_key("whisper.detected_language", detected_language)

    normalized = _normalize_to_whisperx_format(result, time_offset=start)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=4)

    rprint(
        f"[green]✓ WhisperMLX completed in {time.time() - start_time:.2f}s "
        f"using model <{selected_model}>[/green]"
    )
    return normalized
