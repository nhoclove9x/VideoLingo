from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_video_files
from core.utils.models import *
import gc

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        try:
            demucs_audio()
            vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
        except Exception as e:
            rprint(
                f"[yellow]⚠️ Demucs/normalize failed, fallback to raw audio for ASR alignment: {e}[/yellow]"
            )
            vocal_audio = _RAW_AUDIO_FILE
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)
    total = len(segments)
    max_len = max((end - start) for start, end in segments) if segments else 0
    rprint(f"[cyan]🧩 ASR will run on {total} segment(s), longest segment: {max_len:.2f}s[/cyan]")
    
    # 4. Transcribe audio by clips
    combined_result = {'segments': []}
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")
    elif runtime == "whispermlx":
        from core.asr_backend.whisper_mlx import transcribe_audio_mlx as ts
        rprint("[cyan]🎤 Transcribing audio with WhisperMLX...[/cyan]")
    else:
        raise ValueError(f"Unsupported whisper.runtime: {runtime}")

    for idx, (start, end) in enumerate(segments, 1):
        rprint(f"[cyan]🎧 Segment {idx}/{total}: {start:.2f}s -> {end:.2f}s[/cyan]")
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        combined_result['segments'].extend(result.get('segments', []))
        del result
        gc.collect()
    
    # 5. Process df
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()
