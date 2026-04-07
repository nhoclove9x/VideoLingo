import subprocess
from typing import Optional

import numpy as np


def _fmt_time(value: float) -> str:
    return f"{float(value):.3f}"


def _build_ffmpeg_slice_cmd(
    audio_path: str,
    start: Optional[float],
    end: Optional[float],
    sample_rate: int,
    output_format: str,
) -> list[str]:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]

    if start is not None:
        cmd.extend(["-ss", _fmt_time(max(float(start), 0.0))])

    cmd.extend(["-i", audio_path])

    if end is not None:
        end = max(float(end), 0.0)
        if start is None:
            cmd.extend(["-to", _fmt_time(end)])
        else:
            duration = max(end - float(start), 0.0)
            cmd.extend(["-t", _fmt_time(duration)])

    cmd.extend(
        [
            "-f",
            output_format,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ]
    )
    return cmd


def _run_ffmpeg_to_bytes(cmd: list[str]) -> bytes:
    try:
        process = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg first.") from e
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed ({process.returncode}): {stderr}")
    return process.stdout

def load_audio_segment(
    audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    sample_rate: int = 16000,
) -> np.ndarray:
    cmd = _build_ffmpeg_slice_cmd(
        audio_path=audio_path,
        start=start,
        end=end,
        sample_rate=sample_rate,
        output_format="s16le",
    )
    raw_bytes = _run_ffmpeg_to_bytes(cmd)
    if not raw_bytes:
        return np.array([], dtype=np.float32)
    return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def load_audio_segment_wav_bytes(
    audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    sample_rate: int = 16000,
) -> bytes:
    cmd = _build_ffmpeg_slice_cmd(
        audio_path=audio_path,
        start=start,
        end=end,
        sample_rate=sample_rate,
        output_format="wav",
    )
    return _run_ffmpeg_to_bytes(cmd)
