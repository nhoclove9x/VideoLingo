import pandas as pd
import json
import concurrent.futures
import os
from datetime import datetime
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from core.utils.models import *
console = Console()
TRANSLATION_PROGRESS_FILE = "output/log/translation_progress.json"

def _safe_load_key(key: str, default):
    try:
        return load_key(key)
    except Exception:
        return default


def _write_translation_progress(
    total_chunks: int,
    completed_chunks: int,
    status: str = "running",
    stage: str = "translation",
    message: str = "",
):
    os.makedirs("output/log", exist_ok=True)
    total = max(int(total_chunks), 0)
    completed = min(max(int(completed_chunks), 0), total if total else 0)
    percent = (completed / total * 100.0) if total > 0 else 0.0
    payload = {
        "stage": stage,  # summary | translation | completed | error
        "status": status,  # running | completed | error
        "total_chunks": total,
        "completed_chunks": completed,
        "percent": round(percent, 2),
        "message": message,
        "updated_at": datetime.now().isoformat(),
    }
    with open(TRANSLATION_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def mark_translation_summary_start():
    _write_translation_progress(
        total_chunks=0,
        completed_chunks=0,
        status="running",
        stage="summary",
        message="Preparing summary/context...",
    )


def _mark_translation_error(message: str):
    _write_translation_progress(
        total_chunks=0,
        completed_chunks=0,
        status="error",
        stage="error",
        message=message,
    )


# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i):
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.read().strip().split('\n') if line.strip()]

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index, context_prev_lines):
    if chunk_index == 0 or context_prev_lines <= 0:
        return None
    return chunks[chunk_index - 1].split('\n')[-context_prev_lines:]

def get_after_content(chunks, chunk_index, context_next_lines):
    if chunk_index == len(chunks) - 1 or context_next_lines <= 0:
        return None
    return chunks[chunk_index + 1].split('\n')[:context_next_lines]

# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i, context_prev_lines, context_next_lines):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i, context_prev_lines)
    after_content_prompt = get_after_content(chunks, i, context_next_lines)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation

# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")
    mark_translation_summary_start()
    chunk_size_chars = int(_safe_load_key("translation.chunk_size_chars", 600))
    chunk_max_lines = int(_safe_load_key("translation.chunk_max_lines", 10))
    context_prev_lines = int(_safe_load_key("translation.context_prev_lines", 3))
    context_next_lines = int(_safe_load_key("translation.context_next_lines", 2))
    translation_workers = int(_safe_load_key("translation.max_workers", load_key("max_workers")))

    chunks = split_chunks_by_chars(chunk_size=chunk_size_chars, max_i=chunk_max_lines)
    console.print(
        f"[cyan]Translation chunks: {len(chunks)} "
        f"(chunk_size_chars={chunk_size_chars}, chunk_max_lines={chunk_max_lines}, "
        f"workers={translation_workers})[/cyan]"
    )
    _write_translation_progress(
        total_chunks=len(chunks),
        completed_chunks=0,
        status="running",
        stage="translation",
        message="Translating chunks...",
    )
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    # 🔄 Use concurrent execution for translation
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=translation_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(
                    translate_chunk,
                    chunk,
                    chunks,
                    theme_prompt,
                    i,
                    context_prev_lines,
                    context_next_lines,
                )
                futures.append(future)
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    _mark_translation_error(f"Translation chunk failed: {e}")
                    raise
                progress.update(task, advance=1)
                _write_translation_progress(
                    total_chunks=len(chunks),
                    completed_chunks=len(results),
                    status="running",
                    stage="translation",
                    message="Translating chunks...",
                )

    results.sort(key=lambda x: x[0])  # Sort results based on original order
    results_by_index = {idx: (src, trans) for idx, src, trans in results}
    
    # 💾 Save results to lists and Excel file
    src_text, trans_text = [], []
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)

        if i not in results_by_index:
            _mark_translation_error(f"Missing translation result for chunk {i}")
            raise ValueError(f"Missing translation result for chunk {i}")
        _, chunk_translation = results_by_index[i]
        chunk_trans_lines = chunk_translation.split('\n')
        if len(chunk_lines) != len(chunk_trans_lines):
            _mark_translation_error(
                f"Chunk {i} line mismatch: src={len(chunk_lines)}, trans={len(chunk_trans_lines)}"
            )
            raise ValueError(
                f"Chunk {i} line mismatch: src={len(chunk_lines)}, trans={len(chunk_trans_lines)}"
            )
        trans_text.extend(chunk_trans_lines)
    
    # Trim long translation text
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    console.print(df_time)
    # apply check_len_then_trim to df_time['Translation'], only when duration > MIN_TRIM_DURATION.
    df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    _write_translation_progress(
        total_chunks=len(chunks),
        completed_chunks=len(chunks),
        status="completed",
        stage="completed",
        message="Translation completed.",
    )
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
