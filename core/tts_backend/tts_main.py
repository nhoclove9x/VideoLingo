import os
import re
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.gpt_sovits_tts import gpt_sovits_tts_for_videolingo
from core.tts_backend.sf_fishtts import siliconflow_fish_tts_for_videolingo
from core.tts_backend.openai_tts import openai_tts
from core.tts_backend.chatanywhere_tts import chatanywhere_tts
from core.tts_backend.chatterbox_tts import chatterbox_tts
from core.tts_backend.kokoro_tts import kokoro_tts
from core.tts_backend.fish_tts import fish_tts
from core.tts_backend.azure_tts import azure_tts
from core.tts_backend.edge_tts import edge_tts
from core.tts_backend.sf_cosyvoice2 import cosyvoice_tts_for_videolingo
from core.tts_backend.custom_tts import custom_tts
from core.prompts import get_correct_text_prompt
from core.tts_backend._302_f5tts import f5_tts_for_videolingo
from core.utils import *

def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return text.strip()


def _is_non_retryable_tts_error(tts_method: str, error: Exception) -> bool:
    """Errors that should fail fast instead of retrying."""
    msg = str(error).lower()
    if tts_method == "chatterbox_tts":
        return (
            "requires optional package `chatterbox-tts`" in msg
            or "installed without some dependencies" in msg
            or "missing dependency" in msg
            or "perth watermarker backend is unavailable" in msg
            or (
                "'nonetype' object is not callable" in msg
                and "perth" in msg
            )
        )
    if tts_method == "kokoro_tts":
        return (
            "requires optional package `kokoro`" in msg
            or "no module named 'kokoro'" in msg
            or "no module named \"kokoro\"" in msg
        )
    return False


def tts_main(text, save_as, number, task_df):
    text = clean_text_for_tts(text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return
    
    # Skip if file exists
    if os.path.exists(save_as):
        return
    
    print(f"Generating <{text}...>")
    TTS_METHOD = load_key("tts_method")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt >= max_retries - 1:
                print("Asking GPT to correct text...")
                correct_text = ask_gpt(get_correct_text_prompt(text),resp_type="json", log_title='tts_correct_text')
                text = correct_text['text']
            if TTS_METHOD == 'openai_tts':
                openai_tts(text, save_as)
            elif TTS_METHOD == 'chatanywhere_tts':
                chatanywhere_tts(text, save_as)
            elif TTS_METHOD == 'chatterbox_tts':
                chatterbox_tts(text, save_as, number=number)
            elif TTS_METHOD == 'kokoro_tts':
                kokoro_tts(text, save_as)
            elif TTS_METHOD == 'gpt_sovits':
                gpt_sovits_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'fish_tts':
                fish_tts(text, save_as)
            elif TTS_METHOD == 'azure_tts':
                azure_tts(text, save_as)
            elif TTS_METHOD == 'sf_fish_tts':
                siliconflow_fish_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'edge_tts':
                edge_tts(text, save_as)
            elif TTS_METHOD == 'custom_tts':
                custom_tts(text, save_as)
            elif TTS_METHOD == 'sf_cosyvoice2':
                cosyvoice_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'f5tts':
                f5_tts_for_videolingo(text, save_as, number, task_df)
                
            # Check generated audio duration
            duration = get_audio_duration(save_as)
            if duration > 0:
                break
            else:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    # Create silent audio file
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed, retrying...")
        except Exception as e:
            if _is_non_retryable_tts_error(TTS_METHOD, e):
                raise Exception(f"{TTS_METHOD} dependency error: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying...")
