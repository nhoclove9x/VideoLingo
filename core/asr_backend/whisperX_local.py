import os
import warnings
import time
import subprocess
import json
import torch
import functools

warnings.filterwarnings("ignore")

# =============================================================================
# Compatibility shim — applied BEFORE importing whisperx
# =============================================================================

# torch.load: default weights_only=False for pyannote checkpoints
# PyTorch >=2.6 changed torch.load default to weights_only=True.
# pyannote checkpoints contain omegaconf objects that fail the safety check.
# Monkey-patch torch.load to default to weights_only=False (matching <2.6
# behavior).  This is safe here because all model files come from trusted
# sources (HuggingFace / pyannote).
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# =============================================================================
# Now safe to import whisperx and the rest of the application
# =============================================================================
import whisperx
from rich import print as rprint
from core.utils import *
from core.asr_backend.audio_segment import load_audio_segment

MODEL_DIR = load_key("model_dir")
OUTPUT_LOG_DIR = "output/log"
_HF_ENDPOINT_CACHE = None

@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    mirrors = {'Official': 'huggingface.co', 'Mirror': 'hf-mirror.com'}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        if os.name == 'nt':
            cmd = ['ping', '-n', '1', '-w', '3000', domain]
        else:
            cmd = ['ping', '-c', '1', '-W', '3', domain]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        response_time = time.time() - start
        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url


def _get_hf_endpoint():
    global _HF_ENDPOINT_CACHE
    if _HF_ENDPOINT_CACHE:
        return _HF_ENDPOINT_CACHE
    _HF_ENDPOINT_CACHE = check_hf_mirror() or "https://huggingface.co"
    return _HF_ENDPOINT_CACHE


def _infer_runtime_params(device: str):
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 8 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        rprint(
            f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, "
            f"[cyan]📦 Batch size:[/cyan] {batch_size}, "
            f"[cyan]⚙️ Compute type:[/cyan] {compute_type}"
        )
        return batch_size, compute_type

    batch_size = 1
    compute_type = "int8"
    rprint(
        f"[cyan]📦 Batch size:[/cyan] {batch_size}, "
        f"[cyan]⚙️ Compute type:[/cyan] {compute_type}"
    )
    return batch_size, compute_type


def _transcribe_with_fallback(model, raw_audio_segment, batch_size: int):
    # Retry with smaller batch sizes if GPU OOM happens.
    candidates = [batch_size] + [b for b in [8, 4, 2, 1] if b < batch_size]
    used = set()
    for bs in candidates:
        if bs in used:
            continue
        used.add(bs)
        try:
            return model.transcribe(
                raw_audio_segment,
                batch_size=bs,
                print_progress=True,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            rprint(f"[yellow]⚠️ OOM at batch_size={bs}, retrying with smaller batch...[/yellow]")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("WhisperX failed after retrying smaller batch sizes.")


def _segment_log_file(start, end):
    start_tag = "full" if start is None else f"{float(start):.3f}"
    end_tag = "full" if end is None else f"{float(end):.3f}"
    return f"{OUTPUT_LOG_DIR}/whisperxlocal_{start_tag}_{end_tag}.json"


@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    log_file = _segment_log_file(start, end)
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)

    os.environ['HF_ENDPOINT'] = _get_hf_endpoint()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting WhisperX using device: {device} ...")
    batch_size, compute_type = _infer_runtime_params(device)
    rprint(f"[green]▶️ Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    if WHISPER_LANGUAGE == 'zh':
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)
        
    if os.path.exists(local_model):
        rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
        model_name = local_model
    else:
        rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

    vad_options = {"vad_onset": 0.500,"vad_offset": 0.363}
    asr_options = {"temperatures": [0],"initial_prompt": "",}
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    rprint("[bold yellow] You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`[/bold yellow]")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)

    raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
    vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)
    if len(raw_audio_segment) == 0:
        raise ValueError("Audio segment is empty for WhisperX transcription")
    
    # -------------------------
    # 1. transcribe raw audio
    # -------------------------
    transcribe_start_time = time.time()
    rprint("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
    result = _transcribe_with_fallback(model, raw_audio_segment, batch_size)
    transcribe_time = time.time() - transcribe_start_time
    rprint(f"[cyan]⏱️ time transcribe:[/cyan] {transcribe_time:.2f}s")

    # Free GPU resources
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save language
    update_key("whisper.language", result['language'])
    if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
        raise ValueError("Please specify the transcription language as zh and try again!")

    # -------------------------
    # 2. align by vocal audio
    # -------------------------
    align_start_time = time.time()
    # Align timestamps using vocal audio
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)
    align_time = time.time() - align_start_time
    rprint(f"[cyan]⏱️ time align:[/cyan] {align_time:.2f}s")

    # Free GPU resources again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model_a

    # Adjust timestamps
    for segment in result['segments']:
        segment['start'] += start
        segment['end'] += start
        for word in segment.get('words', []):
            if 'start' in word:
                word['start'] += start
            if 'end' in word:
                word['end'] += start

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return result
