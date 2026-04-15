import json
import os
import shutil
import subprocess
from pathlib import Path

from core._1_ytdlp import find_video_files
from core.utils.config_utils import load_key
from rich import print as rprint


def _safe_load_key(key: str, default):
    try:
        return load_key(key)
    except Exception:
        return default


def _build_atempo_chain(speed: float) -> str:
    """Build atempo chain that supports arbitrary positive speed."""
    if speed <= 0:
        raise ValueError("Speed factor must be > 0")

    factors = []
    remaining = float(speed)

    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    factors.append(remaining)
    return ",".join([f"atempo={x:.6f}" for x in factors])


def _has_audio_stream(video_file: str) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        video_file,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return bool(result.stdout.strip())
    except FileNotFoundError:
        # ffprobe may be missing in some environments; default to trying audio path.
        return True


def _backup_video_path(video_file: str) -> str:
    p = Path(video_file)
    return str(p.with_name(f"output_original_{p.stem}{p.suffix}"))


def _speed_meta_path(video_file: str) -> str:
    p = Path(video_file)
    return str(p.with_name(f"{p.stem}.video_speed.json"))


def _file_fingerprint(file_path: str):
    if not os.path.exists(file_path):
        return None
    stat = os.stat(file_path)
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_speed_meta(meta_file: str):
    if not os.path.exists(meta_file):
        return None
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_speed_meta(meta_file: str, video_file: str, backup_file: str | None, speed: float) -> None:
    data = {
        "video_file": os.path.abspath(video_file),
        "backup_file": os.path.abspath(backup_file) if backup_file else None,
        "applied_speed": float(speed),
        "current_fingerprint": _file_fingerprint(video_file),
        "backup_fingerprint": _file_fingerprint(backup_file) if backup_file else None,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def _media_duration(file_path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
        return None


def _matches_expected_duration(backup_file: str, video_file: str, speed: float) -> bool:
    if not os.path.exists(backup_file) or speed <= 0:
        return False

    original_duration = _media_duration(backup_file)
    current_duration = _media_duration(video_file)
    if original_duration is None or current_duration is None:
        return False

    expected_duration = original_duration / speed
    tolerance = max(0.35, expected_duration * 0.01)
    return abs(current_duration - expected_duration) <= tolerance


def _is_speed_already_applied(video_file: str, backup_file: str, meta_file: str, speed: float) -> bool:
    meta = _load_speed_meta(meta_file)
    current_fingerprint = _file_fingerprint(video_file)

    if meta:
        applied_speed = meta.get("applied_speed")
        if (
            applied_speed is not None
            and abs(float(applied_speed) - speed) < 1e-6
            and meta.get("current_fingerprint") == current_fingerprint
        ):
            return True

    if _matches_expected_duration(backup_file, video_file, speed):
        _save_speed_meta(meta_file, video_file, backup_file, speed)
        return True

    return False


def adjust_video_speed():
    speed = float(_safe_load_key("video_speed.factor", 1.0))
    if speed <= 0:
        raise ValueError("`video_speed.factor` must be > 0")

    video_file = find_video_files()
    backup_file = _backup_video_path(video_file)
    meta_file = _speed_meta_path(video_file)

    if _is_speed_already_applied(video_file, backup_file, meta_file, speed):
        rprint(f"[cyan]⏩ Video is already at {speed:.2f}x. Skipping speed adjustment.[/cyan]")
        return

    if abs(speed - 1.0) < 1e-6:
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, video_file)
            _save_speed_meta(meta_file, video_file, backup_file, speed)
            rprint("[green]⏩ Video speed is 1.0x, restored original video.[/green]")
        else:
            _save_speed_meta(meta_file, video_file, None, speed)
            rprint("[green]⏩ Video speed is already 1.0x. No adjustment needed.[/green]")
        return

    # Keep an untouched backup so users can change speed freely (absolute from original).
    if not os.path.exists(backup_file):
        shutil.copy2(video_file, backup_file)
        rprint(f"[cyan]📦 Saved original video backup: {backup_file}[/cyan]")

    tmp_file = str(Path(video_file).with_name(f"{Path(video_file).stem}.speed_tmp{Path(video_file).suffix}"))
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    has_audio = _has_audio_stream(backup_file)
    setpts_filter = f"setpts=PTS/{speed:.6f}"

    cmd = ["ffmpeg", "-y", "-i", backup_file, "-vf", setpts_filter]
    if has_audio:
        cmd += ["-af", _build_atempo_chain(speed)]
    else:
        cmd += ["-an"]
    cmd += [tmp_file]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError as e:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise RuntimeError("FFmpeg is not available in PATH. Please install ffmpeg first.") from e
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        err = (e.stderr or "").strip()
        raise RuntimeError(f"Failed to adjust video speed ({speed}x): {err}") from e

    os.replace(tmp_file, video_file)
    _save_speed_meta(meta_file, video_file, backup_file, speed)
    rprint(f"[green]✅ Video speed adjusted to {speed:.2f}x before subtitle processing.[/green]")


if __name__ == "__main__":
    adjust_video_speed()
