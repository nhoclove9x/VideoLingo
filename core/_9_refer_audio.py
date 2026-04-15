import os
import subprocess
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import pandas as pd
import soundfile as sf

from core.utils import *
from core.utils.models import *
from core.asr_backend.demucs_vl import demucs_audio

console = Console()


def _is_valid_audio_file(path: str, min_size_bytes: int = 1024) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_size_bytes
    except OSError:
        return False


def _read_audio_with_ffmpeg_fallback(path: str):
    try:
        return sf.read(path)
    except Exception as sf_error:
        os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
        tmp_wav = os.path.join(_AUDIO_TMP_DIR, "refer_source.wav")
        cmd = ["ffmpeg", "-y", "-i", path, tmp_wav]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return sf.read(tmp_wav)
        except Exception as ffmpeg_error:
            raise RuntimeError(
                f"Cannot read audio file: {path}. "
                f"soundfile error: {sf_error}; ffmpeg fallback error: {ffmpeg_error}"
            ) from ffmpeg_error


def _load_reference_audio():
    candidates = [_VOCAL_AUDIO_FILE, _RAW_AUDIO_FILE]
    errors = []
    for path in candidates:
        if not _is_valid_audio_file(path):
            errors.append(f"{path} is missing or empty")
            continue
        try:
            data, sr = _read_audio_with_ffmpeg_fallback(path)
            if len(data) == 0:
                raise RuntimeError("decoded audio is empty")
            if path != _VOCAL_AUDIO_FILE:
                rprint(
                    f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} is unavailable. "
                    f"Using {_RAW_AUDIO_FILE} as reference source.[/yellow]"
                )
            return data, sr
        except Exception as e:
            errors.append(f"{path}: {e}")

    raise RuntimeError(
        "No valid source audio found for reference extraction. "
        + " | ".join(errors)
    )

def time_to_samples(time_str, sr):
    """Unified time conversion function"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',') if ',' in s else (s, '0')
    seconds = int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
    return int(seconds * sr)

def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Simplified audio extraction function"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    if end <= start:
        return False
    sf.write(out_file, audio_data[start:end], sr)
    return True

def extract_refer_audio_main():
    demucs_audio() #!!! in case demucs not run
    if _is_valid_audio_file(os.path.join(_AUDIO_REFERS_DIR, "1.wav"), min_size_bytes=256):
        rprint(Panel("Audio segments already exist, skipping extraction", title="Info", border_style="blue"))
        return

    # Create output directory
    os.makedirs(_AUDIO_REFERS_DIR, exist_ok=True)
    
    # Read task file and audio data
    df = pd.read_excel(_8_1_AUDIO_TASK)
    data, sr = _load_reference_audio()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Extracting audio segments...", total=len(df))
        written = 0
        for _, row in df.iterrows():
            out_file = os.path.join(_AUDIO_REFERS_DIR, f"{int(row['number'])}.wav")
            ok = extract_audio(data, sr, row['start_time'], row['end_time'], out_file)
            if ok:
                written += 1
            progress.update(task, advance=1)

    rprint(Panel(f"Audio segments saved to {_AUDIO_REFERS_DIR} ({written}/{len(df)})", title="Success", border_style="green"))

if __name__ == "__main__":
    extract_refer_audio_main()
