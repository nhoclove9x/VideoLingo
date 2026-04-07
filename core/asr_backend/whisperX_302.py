import os
import io
import json
import time
import requests
from rich import print as rprint
from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_segment import load_audio_segment_wav_bytes

OUTPUT_LOG_DIR = "output/log"


def _segment_log_file(start: float, end: float) -> str:
    start_tag = "full" if start is None else f"{float(start):.3f}"
    end_tag = "full" if end is None else f"{float(end):.3f}"
    return f"{OUTPUT_LOG_DIR}/whisperx302_{start_tag}_{end_tag}.json"


def transcribe_audio_302(raw_audio_path: str, vocal_audio_path: str, start: float = None, end: float = None):
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    LOG_FILE = _segment_log_file(start, end)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
        
    WHISPER_LANGUAGE = load_key("whisper.language")
    update_key("whisper.language", WHISPER_LANGUAGE)
    url = "https://api.302.ai/302/whisperx"

    audio_buffer = io.BytesIO(
        load_audio_segment_wav_bytes(
            vocal_audio_path,
            start=start,
            end=end,
            sample_rate=16000,
        )
    )
    
    files = [('audio_input', ('audio_slice.wav', audio_buffer, 'application/octet-stream'))]
    payload = {"processing_type": "align", "language": WHISPER_LANGUAGE, "output": "raw"}
    
    start_time = time.time()
    rprint(f"[cyan]🎤 Transcribing audio with language:  <{WHISPER_LANGUAGE}> ...[/cyan]")
    headers = {'Authorization': f'Bearer {load_key("whisper.whisperX_302_api_key")}'}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    response.raise_for_status()
    
    response_json = response.json()
    
    if start is not None:
        for segment in response_json['segments']:
            segment['start'] += start
            segment['end'] += start
            for word in segment.get('words', []):
                if 'start' in word:
                    word['start'] += start
                if 'end' in word:
                    word['end'] += start
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(response_json, f, indent=4, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    rprint(f"[green]✓ Transcription completed in {elapsed_time:.2f} seconds[/green]")
    return response_json

if __name__ == "__main__":  
    result = transcribe_audio_302(_RAW_AUDIO_FILE, _RAW_AUDIO_FILE)
    rprint(result)
