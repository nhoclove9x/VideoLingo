import os
import re
import json
from typing import Callable, Iterable


def parse_srt_blocks(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n").strip()
    if not content:
        return []

    blocks = []
    for raw_block in re.split(r"\n\s*\n", content):
        lines = [line.strip() for line in raw_block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue

        if lines[0].isdigit():
            timestamp = lines[1]
            text_lines = lines[2:]
        else:
            timestamp = lines[0]
            text_lines = lines[1:]

        blocks.append(
            {
                "timestamp": timestamp,
                "text": "\n".join(text_lines).strip(),
            }
        )
    return blocks


def render_srt_blocks(blocks: list[dict]) -> str:
    rendered = []
    for i, block in enumerate(blocks, start=1):
        text = str(block.get("text", "")).strip()
        timestamp = str(block.get("timestamp", "")).strip()
        rendered.append(f"{i}\n{timestamp}\n{text}")
    return ("\n\n".join(rendered) + "\n") if rendered else ""


def _safe_load_key(key: str, default):
    try:
        from core.utils import load_key

        return load_key(key)
    except Exception:
        return default


def _translate_batch_with_llm(source_texts: list[str]) -> list[str]:
    from core.utils import ask_gpt

    if not source_texts:
        return []

    source_language = _safe_load_key("whisper.detected_language", "source language")
    target_language = _safe_load_key("target_language", "target language")

    payload = {str(i + 1): text for i, text in enumerate(source_texts)}
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    expected_keys = [str(i + 1) for i in range(len(source_texts))]

    prompt = f"""
## Role
You are a professional subtitle translator.

## Task
Translate each subtitle line from {source_language} to {target_language}.

Rules:
1. Keep exactly the same JSON keys as input.
2. Return only translated text values in {target_language}.
3. Do not leave any translated value empty.
4. Do not add explanations or extra keys.

## INPUT
```json
{payload_json}
```

## Output in only JSON format and no other text
```json
{{
  "1": "translated line 1",
  "2": "translated line 2"
}}
```
""".strip()

    def valid_translate(response_data):
        if not isinstance(response_data, dict):
            return {"status": "error", "message": "Response is not a JSON object"}

        if set(response_data.keys()) != set(expected_keys):
            return {
                "status": "error",
                "message": f"Response keys mismatch. Expected: {expected_keys}, got: {sorted(response_data.keys())}",
            }

        for key in expected_keys:
            value = str(response_data.get(key, "")).strip()
            if not value:
                return {"status": "error", "message": f"Empty translation at key {key}"}

        return {"status": "success", "message": "Translation completed"}

    parsed = ask_gpt(
        prompt,
        resp_type="json",
        valid_def=valid_translate,
        log_title="srt_recheck_translate",
    )
    return [str(parsed[str(i + 1)]).strip() for i in range(len(source_texts))]


def _translate_missing_texts_with_llm(source_texts: list[str]) -> list[str]:
    if not source_texts:
        return []

    max_lines = int(_safe_load_key("translation.chunk_max_lines", 50))
    max_lines = max(1, max_lines)

    translated = []
    for start in range(0, len(source_texts), max_lines):
        batch = source_texts[start : start + max_lines]
        translated.extend(_translate_batch_with_llm(batch))
    return translated


def recheck_and_patch_target_srt(
    source_srt: str,
    target_srt: str,
    translate_fn: Callable[[list[str]], list[str]] | None = None,
) -> dict:
    source_blocks = parse_srt_blocks(source_srt)
    if not source_blocks:
        return {
            "checked": False,
            "changed": False,
            "source_path": source_srt,
            "target_path": target_srt,
            "source_count": 0,
            "target_count": 0,
            "filled_missing": 0,
            "filled_empty": 0,
        }

    target_blocks = parse_srt_blocks(target_srt)
    target_by_timestamp = {}
    for block in target_blocks:
        timestamp = block["timestamp"]
        target_by_timestamp.setdefault(timestamp, []).append(block)

    rebuilt_target = []
    missing_line_indexes = []
    missing_source_texts = []
    filled_missing = 0
    filled_empty = 0

    for src_block in source_blocks:
        candidates = target_by_timestamp.get(src_block["timestamp"], [])
        target_block = candidates.pop(0) if candidates else None
        if target_block is None:
            rebuilt_target.append(
                {
                    "timestamp": src_block["timestamp"],
                    "text": "",
                }
            )
            missing_line_indexes.append(len(rebuilt_target) - 1)
            missing_source_texts.append(str(src_block["text"]).strip())
            filled_missing += 1
            continue

        target_text = str(target_block.get("text", "")).strip()
        if not target_text:
            missing_line_indexes.append(len(rebuilt_target))
            missing_source_texts.append(str(src_block["text"]).strip())
            filled_empty += 1
        rebuilt_target.append(
            {
                "timestamp": src_block["timestamp"],
                "text": target_text,
            }
        )

    if missing_source_texts:
        translator = translate_fn or _translate_missing_texts_with_llm
        translatable_source_texts = []
        for source_text in missing_source_texts:
            if source_text:
                translatable_source_texts.append(source_text)

        translated_texts = []
        if translatable_source_texts:
            translated_texts = translator(translatable_source_texts)
            if len(translated_texts) != len(translatable_source_texts):
                raise ValueError(
                    "Recheck translation count mismatch: "
                    f"expected {len(translatable_source_texts)}, got {len(translated_texts)}"
                )

        translated_cursor = 0
        for i, source_text in enumerate(missing_source_texts):
            target_idx = missing_line_indexes[i]
            if source_text:
                rebuilt_target[target_idx]["text"] = str(translated_texts[translated_cursor]).strip()
                translated_cursor += 1
            else:
                rebuilt_target[target_idx]["text"] = ""

    old_target_content = ""
    if os.path.exists(target_srt):
        with open(target_srt, "r", encoding="utf-8") as f:
            old_target_content = f.read().replace("\r\n", "\n").strip()
    new_target_content = render_srt_blocks(rebuilt_target)
    new_target_normalized = new_target_content.replace("\r\n", "\n").strip()

    changed = old_target_content != new_target_normalized
    if changed:
        os.makedirs(os.path.dirname(target_srt), exist_ok=True)
        with open(target_srt, "w", encoding="utf-8") as f:
            f.write(new_target_content)

    return {
        "checked": True,
        "changed": changed,
        "source_path": source_srt,
        "target_path": target_srt,
        "source_count": len(source_blocks),
        "target_count": len(target_blocks),
        "filled_missing": filled_missing,
        "filled_empty": filled_empty,
    }


def run_srt_recheck_pairs(pairs: Iterable[tuple[str, str]]) -> list[dict]:
    results = []
    for source_srt, target_srt in pairs:
        if os.path.exists(source_srt):
            results.append(recheck_and_patch_target_srt(source_srt, target_srt))
    return results
