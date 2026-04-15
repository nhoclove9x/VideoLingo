import os
import torch
import shutil
import subprocess
from rich.console import Console
from rich import print as rprint
from demucs.pretrained import get_model
from demucs.audio import save_audio
from torch.cuda import is_available as is_cuda_available
from typing import Optional, Tuple
from demucs.api import Separator
from demucs.apply import BagOfModels
import gc
from core.utils.models import *
from core.utils import load_key
from pydub.utils import mediainfo

_DEMUCS_STATUS_FILE = os.path.join(_AUDIO_DIR, "demucs_status.txt")

class PreloadedSeparator(Separator):
    def __init__(self, model: BagOfModels, shifts: int = 1, overlap: float = 0.25,
                 split: bool = True, segment: Optional[int] = None, jobs: int = 0, device: str = "cpu"):
        self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                            segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)


def _safe_load_key(key: str, default):
    try:
        return load_key(key)
    except Exception:
        return default


def _safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _mps_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _resolve_demucs_device(device_cfg: str, console: Console) -> str:
    device_cfg = str(device_cfg or "auto").strip().lower()
    if device_cfg == "auto":
        if is_cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if device_cfg == "cuda":
        if is_cuda_available():
            return "cuda"
        console.print("[yellow]⚠️ demucs_device=cuda but CUDA is unavailable, fallback to CPU.[/yellow]")
        return "cpu"
    if device_cfg == "mps":
        if _mps_available():
            return "mps"
        console.print("[yellow]⚠️ demucs_device=mps but MPS is unavailable, fallback to CPU.[/yellow]")
        return "cpu"
    return "cpu"


def _empty_device_cache(device: str):
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _to_2d_audio(tensor) -> torch.Tensor:
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach().float()
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _align_audio_shapes(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = _to_2d_audio(a)
    b = _to_2d_audio(b)

    # Align channel count with minimal assumptions.
    if a.shape[0] != b.shape[0]:
        if a.shape[0] == 1 and b.shape[0] > 1:
            a = a.repeat(b.shape[0], 1)
        elif b.shape[0] == 1 and a.shape[0] > 1:
            b = b.repeat(a.shape[0], 1)
        else:
            ch = min(a.shape[0], b.shape[0])
            a = a[:ch]
            b = b[:ch]

    t = min(a.shape[-1], b.shape[-1])
    a = a[..., :t]
    b = b[..., :t]
    return a, b


def _build_background_track(outputs: dict, mix: Optional[torch.Tensor], mode: str):
    mode = str(mode or "mix_minus_vocals").strip().lower()
    vocals = outputs.get("vocals")
    vocals = _to_2d_audio(vocals).cpu() if vocals is not None else None
    non_vocals = [_to_2d_audio(audio).cpu() for source, audio in outputs.items() if source != "vocals"]
    mix = _to_2d_audio(mix).cpu() if mix is not None else None

    if mode == "sum_non_vocals":
        if non_vocals:
            return sum(non_vocals), "sum_non_vocals"
    elif mode == "mix_minus_vocals":
        if mix is not None and vocals is not None:
            mix, vocals = _align_audio_shapes(mix, vocals)
            return mix - vocals, "mix_minus_vocals"
    else:
        # auto/hybrid: prefer subtraction from original mix for stronger vocal removal.
        if mix is not None and vocals is not None:
            mix, vocals = _align_audio_shapes(mix, vocals)
            return mix - vocals, "mix_minus_vocals"
        if non_vocals:
            return sum(non_vocals), "sum_non_vocals"

    if non_vocals:
        return sum(non_vocals), "sum_non_vocals_fallback"
    if mix is not None:
        return mix, "raw_mix_fallback"
    return None, "missing"


def _get_model_max_segment_seconds(model: BagOfModels) -> Optional[float]:
    try:
        sub_models = list(getattr(model, "models", []) or [])
        segments = []
        for m in sub_models:
            seg = getattr(m, "segment", None)
            if seg is None:
                continue
            segments.append(float(seg))
        if segments:
            return min(segments)
    except Exception:
        pass
    return None


def _get_audio_duration_seconds(audio_file: str) -> float:
    try:
        duration = float(mediainfo(audio_file).get("duration", 0) or 0)
        if duration > 0:
            return duration
    except Exception:
        pass

    # Fallback heuristic: raw audio is usually encoded at 32 kbps in this project.
    # This gives a safe upper estimate even when ffprobe is unavailable.
    try:
        raw_size_bytes = os.path.getsize(audio_file)
        estimated_bitrate_kbps = float(_safe_load_key("demucs_estimate_bitrate_kbps", 32))
        estimated_bitrate_kbps = max(estimated_bitrate_kbps, 1)
        return (raw_size_bytes * 8.0) / (estimated_bitrate_kbps * 1000.0)
    except Exception:
        return 0.0


def _probe_audio_duration_seconds(audio_file: str) -> Optional[float]:
    try:
        duration = float(mediainfo(audio_file).get("duration", 0) or 0)
        if duration > 0:
            return duration
    except Exception:
        return None
    return None


def _is_valid_audio_file(audio_file: str, min_size_bytes: int = 1024) -> bool:
    try:
        if not os.path.isfile(audio_file):
            return False
        if os.path.getsize(audio_file) < min_size_bytes:
            return False
    except OSError:
        return False

    duration = _probe_audio_duration_seconds(audio_file)
    if duration is None:
        # If ffprobe is unavailable, keep a best-effort size-based validation.
        return True
    return duration > 0.0


def _ensure_fallback_tracks():
    # Fallback path: no Demucs separation, keep pipeline running with raw track.
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    shutil.copyfile(_RAW_AUDIO_FILE, _VOCAL_AUDIO_FILE)
    if _try_build_simple_no_vocal_background():
        rprint(
            "[yellow]⚠️ Demucs skipped. Built approximate no-vocal background with ffmpeg fallback filter.[/yellow]"
        )
    else:
        shutil.copyfile(_RAW_AUDIO_FILE, _BACKGROUND_AUDIO_FILE)
        rprint("[yellow]⚠️ Demucs skipped. Using raw audio as fallback vocal/background tracks.[/yellow]")
    _write_demucs_status("fallback")


def _tracks_up_to_date() -> bool:
    if not (_is_valid_audio_file(_VOCAL_AUDIO_FILE) and _is_valid_audio_file(_BACKGROUND_AUDIO_FILE)):
        return False
    try:
        raw_mtime = os.path.getmtime(_RAW_AUDIO_FILE)
        vocal_mtime = os.path.getmtime(_VOCAL_AUDIO_FILE)
        background_mtime = os.path.getmtime(_BACKGROUND_AUDIO_FILE)
        up_to_date = vocal_mtime >= raw_mtime and background_mtime >= raw_mtime
        if not up_to_date:
            return False
        require_separated = _safe_bool(_safe_load_key("demucs_require_separated", True), True)
        if require_separated:
            return _read_demucs_status() == "separated"
        return True
    except OSError:
        return False


def _write_demucs_status(status: str):
    try:
        os.makedirs(_AUDIO_DIR, exist_ok=True)
        with open(_DEMUCS_STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(status.strip().lower())
    except Exception:
        pass


def _read_demucs_status() -> str:
    try:
        with open(_DEMUCS_STATUS_FILE, "r", encoding="utf-8") as f:
            return f.read().strip().lower()
    except Exception:
        return ""


def _try_build_simple_no_vocal_background() -> bool:
    """
    Build an approximate karaoke/no-vocal background if Demucs is unavailable.
    This is a best-effort fallback and intentionally conservative.
    """
    if not _safe_bool(_safe_load_key("demucs_background_fallback_filter", True), True):
        return False

    filters = [
        "pan=stereo|c0=FL-FR|c1=FR-FL,highpass=f=120,lowpass=f=12000,volume=1.6",
        "pan=stereo|c0=c0-c1|c1=c1-c0,highpass=f=120,lowpass=f=12000,volume=1.6",
    ]
    for af in filters:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            _RAW_AUDIO_FILE,
            "-ac",
            "2",
            "-ar",
            "44100",
            "-af",
            af,
            _BACKGROUND_AUDIO_FILE,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(_BACKGROUND_AUDIO_FILE) and os.path.getsize(_BACKGROUND_AUDIO_FILE) > 0:
                return True
        except Exception:
            continue
    return False

def demucs_audio():
    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    duration_sec = _get_audio_duration_seconds(_RAW_AUDIO_FILE)
    max_minutes = float(_safe_load_key("demucs_max_minutes", 90))
    max_seconds = max(max_minutes * 60, 60)
    if duration_sec > max_seconds:
        console.print(
            f"[yellow]⚠️ Audio duration is {duration_sec/60:.1f} min, exceeding demucs_max_minutes={max_minutes:.1f}. "
            "Skip Demucs to avoid OOM kill.[/yellow]"
        )
        _ensure_fallback_tracks()
        return

    if _tracks_up_to_date():
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return

    demucs_device_cfg = _safe_load_key("demucs_device", "auto")
    demucs_device = _resolve_demucs_device(demucs_device_cfg, console)
    demucs_background_mode = _safe_load_key("demucs_background_mode", "mix_minus_vocals")
    demucs_segment_seconds = int(_safe_load_key("demucs_segment_seconds", 15))
    demucs_segment_seconds = max(demucs_segment_seconds, 5)
    demucs_overlap = float(_safe_load_key("demucs_overlap", 0.1))
    demucs_overlap = min(max(demucs_overlap, 0.0), 0.5)
    demucs_mps_segment_seconds = int(_safe_load_key("demucs_mps_segment_seconds", 8))
    demucs_mps_segment_seconds = max(demucs_mps_segment_seconds, 5)
    demucs_mps_overlap = float(_safe_load_key("demucs_mps_overlap", 0.0))
    demucs_mps_overlap = min(max(demucs_mps_overlap, 0.0), 0.5)
    demucs_mps_fallback_cpu = _safe_bool(_safe_load_key("demucs_mps_fallback_cpu", True), True)
    demucs_fulltrack_fallback = _safe_bool(_safe_load_key("demucs_fulltrack_fallback", True), True)
    demucs_fulltrack_max_minutes = float(_safe_load_key("demucs_fulltrack_max_minutes", 20))
    
    console.print(f"🤖 Loading <htdemucs> model... (device={demucs_device})")
    model = get_model('htdemucs')
    outputs = None
    mix = None
    successful_device = None
    model_max_segment = _get_model_max_segment_seconds(model)
    safe_max_segment_int = None
    if model_max_segment is not None and model_max_segment > 0:
        safe_max_segment_int = max(1, int(model_max_segment - 1e-6))
        if demucs_segment_seconds > safe_max_segment_int:
            console.print(
                f"[yellow]⚠️ demucs_segment_seconds={demucs_segment_seconds}s exceeds "
                f"model-safe max ~{model_max_segment:.2f}s, clamped to {safe_max_segment_int}s.[/yellow]"
            )
            demucs_segment_seconds = safe_max_segment_int
        if demucs_mps_segment_seconds > safe_max_segment_int:
            console.print(
                f"[yellow]⚠️ demucs_mps_segment_seconds={demucs_mps_segment_seconds}s exceeds "
                f"model-safe max ~{model_max_segment:.2f}s, clamped to {safe_max_segment_int}s.[/yellow]"
            )
            demucs_mps_segment_seconds = safe_max_segment_int

    device_attempts = [demucs_device]
    if demucs_device == "mps" and demucs_mps_fallback_cpu:
        device_attempts.append("cpu")

    attempt_counter = 0
    total_attempts = 0
    per_device_attempts = {}
    fulltrack_allowed = (
        demucs_fulltrack_fallback
        and duration_sec <= max(60.0, demucs_fulltrack_max_minutes * 60.0)
    )
    for device in device_attempts:
        base_segment = demucs_mps_segment_seconds if device == "mps" else demucs_segment_seconds
        base_overlap = demucs_mps_overlap if device == "mps" else demucs_overlap
        safe_chunk_segment = max(5, min(base_segment, safe_max_segment_int or base_segment))
        attempts = [
            {"segment": base_segment, "overlap": base_overlap, "split": True, "tag": "chunk-base"},
            {"segment": safe_chunk_segment, "overlap": 0.0, "split": True, "tag": "chunk-safe"},
            {"segment": None, "overlap": 0.0 if device == "mps" else 0.1, "split": True, "tag": "model-default"},
        ]
        if device == "cpu" and fulltrack_allowed:
            attempts.append({"segment": None, "overlap": 0.0, "split": False, "tag": "fulltrack"})
        seen = set()
        deduped = []
        for a in attempts:
            key = (a["segment"], round(float(a["overlap"]), 4), bool(a["split"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)
        per_device_attempts[device] = deduped
        total_attempts += len(deduped)

    for device in device_attempts:
        attempts = per_device_attempts[device]
        for attempt in attempts:
            segment_seconds = attempt["segment"]
            overlap = float(attempt["overlap"])
            split = bool(attempt["split"])
            tag = attempt["tag"]
            jobs = 0 if (device == "mps" or not split) else 1
            attempt_counter += 1
            separator = PreloadedSeparator(
                model=model,
                device=device,
                shifts=1,
                overlap=overlap,
                split=split,
                segment=segment_seconds,
                jobs=jobs,
            )
            try:
                segment_label = "default" if segment_seconds is None else f"{segment_seconds}s"
                console.print(
                    f"🎵 Separating audio... (attempt {attempt_counter}/{total_attempts}, "
                    f"device={device}, mode={tag}, split={split}, segment={segment_label}, overlap={overlap:.2f})"
                )
                mix, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)
                break
            except RuntimeError as e:
                msg = str(e)
                console.print(
                    f"[yellow]⚠️ Demucs runtime error on {device}: {msg}[/yellow]"
                )
                if "shape" in msg and "invalid for input of size" in msg:
                    console.print(
                        "[yellow]⚠️ Detected Demucs shape mismatch. Retrying with safer settings/device...[/yellow]"
                    )
            except Exception as e:
                console.print(
                    f"[yellow]⚠️ Demucs failed on {device}: {e}[/yellow]"
                )
            finally:
                del separator
                gc.collect()
                _empty_device_cache(device)

        if outputs:
            successful_device = device
            if device == "cpu" and demucs_device == "mps":
                console.print("[yellow]⚠️ Demucs succeeded on CPU fallback (MPS path failed).[/yellow]")
            break

    if not outputs or "vocals" not in outputs:
        console.print(
            "[yellow]⚠️ Demucs output is incomplete (missing vocals). Falling back to raw audio.[/yellow]"
        )
        _ensure_fallback_tracks()
        return
    
    kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2, 
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}
    
    console.print("🎤 Saving vocals track...")
    vocals = outputs["vocals"]
    vocals = _to_2d_audio(vocals).detach().cpu().float()
    save_audio(vocals, _VOCAL_AUDIO_FILE, **kwargs)
    
    console.print("🎹 Building no-vocal background track...")
    if mix is not None:
        mix = _to_2d_audio(mix).detach().cpu().float()
    background, bg_mode = _build_background_track(outputs, mix, demucs_background_mode)
    if background is None:
        console.print(
            "[yellow]⚠️ Cannot build no-vocal background from Demucs outputs. Using raw audio as background.[/yellow]"
        )
        shutil.copyfile(_RAW_AUDIO_FILE, _BACKGROUND_AUDIO_FILE)
    else:
        background = _to_2d_audio(background).detach().cpu().float()
        save_audio(background, _BACKGROUND_AUDIO_FILE, **kwargs)
        console.print(f"[green]✅ Background built with mode: {bg_mode}[/green]")
        del background
        _write_demucs_status("separated")

    if not (_is_valid_audio_file(_VOCAL_AUDIO_FILE) and _is_valid_audio_file(_BACKGROUND_AUDIO_FILE)):
        console.print(
            "[yellow]⚠️ Demucs output tracks are invalid or empty. Rebuilding with raw-audio fallback.[/yellow]"
        )
        _ensure_fallback_tracks()
        return
    
    # Clean up memory
    del outputs, model
    if mix is not None:
        del mix
    gc.collect()
    if successful_device:
        _empty_device_cache(successful_device)
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_audio()
