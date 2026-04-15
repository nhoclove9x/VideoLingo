import os, subprocess, tempfile
import pandas as pd
from typing import Dict, List, Tuple
from pydub import AudioSegment
from core.utils import *
from core.utils.models import *
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from rich import print as rprint


def _safe_load_key(key: str, default):
    try:
        return load_key(key)
    except KeyError:
        return default


def _enforce_max_segment_length(
    segments: List[Tuple[float, float]],
    max_len_sec: float,
) -> List[Tuple[float, float]]:
    if max_len_sec <= 0:
        return segments

    enforced: List[Tuple[float, float]] = []
    for start, end in segments:
        cursor = float(start)
        end = float(end)
        while end - cursor > max_len_sec:
            next_cut = cursor + max_len_sec
            enforced.append((cursor, next_cut))
            cursor = next_cut
        if end > cursor:
            enforced.append((cursor, end))
    return enforced

def _ffmpeg_has_encoder(encoder_name: str) -> bool:
    """Check if the current ffmpeg installation supports a given audio encoder."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10
        )
        return encoder_name in result.stdout
    except Exception:
        return False

def normalize_audio_volume(audio_path, output_path, target_db = -20.0, format = "wav"):
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)

    same_path = os.path.abspath(audio_path) == os.path.abspath(output_path)
    export_path = output_path
    temp_path = None

    try:
        if same_path:
            out_dir = os.path.dirname(output_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix="normalize_", suffix=f".{format}", dir=out_dir)
            os.close(fd)
            export_path = temp_path

        normalized_audio.export(export_path, format=format)

        if same_path:
            if not os.path.isfile(export_path) or os.path.getsize(export_path) == 0:
                raise RuntimeError(f"Normalized audio export failed: {export_path}")
            os.replace(export_path, output_path)

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Normalized audio output is invalid: {output_path}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {target_db:.1f}dB[/green]")
    return output_path

def convert_video_to_audio(video_file: str):
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    if not os.path.exists(_RAW_AUDIO_FILE):
        rprint(f"[blue]🎬➡️🎵 Converting to high quality audio with FFmpeg ......[/blue]")
        if _ffmpeg_has_encoder('libmp3lame'):
            cmd = [
                'ffmpeg', '-y', '-i', video_file, '-vn',
                '-c:a', 'libmp3lame', '-b:a', '32k',
                '-ar', '16000', '-ac', '1',
                '-metadata', 'encoding=UTF-8', _RAW_AUDIO_FILE
            ]
        else:
            # Fallback: conda-forge ffmpeg often lacks libmp3lame.
            # Output as WAV (PCM) which all ffmpeg builds support.
            # Downstream readers (pydub, librosa, whisperX) detect format by
            # file header, not extension, so .mp3 path with WAV content works.
            rprint("[yellow]⚠️ libmp3lame not found in ffmpeg, falling back to WAV (PCM) encoding[/yellow]")
            cmd = [
                'ffmpeg', '-y', '-i', video_file, '-vn',
                '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-f', 'wav', _RAW_AUDIO_FILE
            ]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        rprint(f"[green]🎬➡️🎵 Converted <{video_file}> to <{_RAW_AUDIO_FILE}> with FFmpeg\n[/green]")

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')
    
    try:
        duration_str = [line for line in output.split('\n') if 'Duration' in line][0]
        duration_parts = duration_str.split('Duration: ')[1].split(',')[0].split(':')
        duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])
    except Exception as e:
        print(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration

def split_audio(audio_file: str, target_len: float = None, win: float = None) -> List[Tuple[float, float]]:
    # Detect split points near silence in [target_len-win, target_len+win].
    if target_len is None:
        target_len = float(_safe_load_key("whisper.segment_minutes", 12)) * 60
    if win is None:
        win = float(_safe_load_key("whisper.segment_search_window_sec", 60))
    split_by_silence = bool(_safe_load_key("whisper.segment_by_silence", True))
    hard_max_len = float(_safe_load_key("whisper.max_segment_minutes", 12)) * 60

    target_len = max(target_len, 60.0)
    win = max(win, 5.0)
    hard_max_len = max(hard_max_len, 60.0)
    if target_len > hard_max_len:
        rprint(
            f"[yellow]⚠️ target segment length {target_len:.1f}s is larger than hard max {hard_max_len:.1f}s. Using hard max.[/yellow]"
        )
        target_len = hard_max_len

    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    duration = float(mediainfo(audio_file)["duration"])
    if duration <= target_len + win:
        segments = [(0.0, duration)]
    else:
        segments = []
        pos = 0.0
        safe_margin = 0.5  # 静默点前后安全边界，单位秒

        while pos < duration:
            if duration - pos <= target_len:
                segments.append((pos, duration))
                break

            threshold = pos + target_len
            split_at = threshold
            if split_by_silence:
                window_start = max(threshold - win, pos)
                window_end = min(threshold + win, duration)
                window_dur = max(window_end - window_start, 0.0)

                if window_dur > 0:
                    window_audio = AudioSegment.from_file(
                        audio_file,
                        start_second=window_start,
                        duration=window_dur,
                    )
                    silence_regions = detect_silence(
                        window_audio,
                        min_silence_len=int(safe_margin * 1000),
                        silence_thresh=-30,
                    )
                    silence_regions = [
                        (s / 1000 + window_start, e / 1000 + window_start)
                        for s, e in silence_regions
                    ]
                    valid_regions = [
                        (start, end)
                        for start, end in silence_regions
                        if (end - start) >= (safe_margin * 2)
                        and threshold <= start + safe_margin <= threshold + win
                    ]

                    if valid_regions:
                        start, _ = valid_regions[0]
                        split_at = start + safe_margin
                    else:
                        rprint(
                            f"[yellow]⚠️ No valid silence regions found for {audio_file} at {threshold}s, using threshold[/yellow]"
                        )

            split_at = min(max(split_at, pos + 1.0), duration)
            segments.append((pos, split_at))
            pos = split_at

    segments = _enforce_max_segment_length(segments, hard_max_len)

    if segments:
        first_start, first_end = segments[0]
        last_start, last_end = segments[-1]
        rprint(
            f"[green]🎙️ Audio split completed {len(segments)} segments "
            f"(first: {first_start:.2f}-{first_end:.2f}s, last: {last_start:.2f}-{last_end:.2f}s)[/green]"
        )
        return segments

    rprint("[yellow]⚠️ Audio split produced no segments, fallback to full range.[/yellow]")
    return [(0.0, duration)]

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Get speaker_id, if not exists, set to None
        speaker_id = segment.get('speaker_id', None)
        
        for word in segment['words']:
            # Check word length
            if len(word["word"]) > 30:
                rprint(f"[yellow]⚠️ Warning: Detected word longer than 30 characters, skipping: {word['word']}[/yellow]")
                continue
                
            # ! For French, we need to convert guillemets to empty strings
            word["word"] = word["word"].replace('»', '').replace('«', '')
            
            if 'start' not in word and 'end' not in word:
                if all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    word_dict = {
                        'text': word["word"],
                        'start': all_words[-1]['end'],
                        'end': all_words[-1]['end'],
                        'speaker_id': speaker_id
                    }
                    all_words.append(word_dict)
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                    if next_word:
                        word_dict = {
                            'text': word["word"],
                            'start': next_word["start"],
                            'end': next_word["end"],
                            'speaker_id': speaker_id
                        }
                        all_words.append(word_dict)
                    else:
                        raise Exception(f"No next word with timestamp found for the current word : {word}")
            else:
                # Normal case, with start and end times
                word_dict = {
                    'text': f'{word["word"]}',
                    'start': word.get('start', all_words[-1]['end'] if all_words else 0),
                    'end': word['end'],
                    'speaker_id': speaker_id
                }
                
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = df[df['text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        rprint(f"[blue]ℹ️ Removed {removed_rows} row(s) with empty text.[/blue]")
    
    # Check for and remove words longer than 20 characters
    long_words = df[df['text'].str.len() > 30]
    if not long_words.empty:
        rprint(f"[yellow]⚠️ Warning: Detected {len(long_words)} word(s) longer than 30 characters. These will be removed.[/yellow]")
        df = df[df['text'].str.len() <= 30]
    
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(_2_CLEANED_CHUNKS, index=False)
    rprint(f"[green]📊 Excel file saved to {_2_CLEANED_CHUNKS}[/green]")

def save_language(language: str):
    update_key("whisper.detected_language", language)
