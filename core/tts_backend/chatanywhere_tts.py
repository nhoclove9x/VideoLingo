from pathlib import Path
import http.client
import json
import base64

import requests

from core.utils import load_key, except_handler

VOICE_LIST = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

def _send_tts_request(payload: dict, headers: dict):
    conn = http.client.HTTPSConnection("api.chatanywhere.tech", timeout=120)
    try:
        conn.request("POST", "/v1/audio/speech", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        content_type = (res.getheader("Content-Type") or "").lower()
        return res.status, data, content_type
    finally:
        conn.close()


@except_handler("Failed to generate audio using ChatAnywhere TTS", retry=3, delay=1)
def chatanywhere_tts(text: str, save_path: str) -> None:
    api_key = load_key("chatanywhere_tts.api_key")
    voice = load_key("chatanywhere_tts.voice")
    model = load_key("chatanywhere_tts.model")
    custom_prompt = load_key("chatanywhere_tts.custom_prompt").strip()

    if not api_key:
        raise ValueError("ChatAnywhere API key is not set")
    if voice not in VOICE_LIST:
        raise ValueError(f"Invalid voice: {voice}. Please choose from {VOICE_LIST}")

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    if custom_prompt:
        payload["instructions"] = custom_prompt
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_fallback_used = False
    status_code, data, content_type = _send_tts_request(payload, headers)

    # Some models may not support `instructions`; fallback once without prompt.
    if status_code != 200 and custom_prompt:
        fallback_payload = payload.copy()
        fallback_payload.pop("instructions", None)
        status_code, data, content_type = _send_tts_request(fallback_payload, headers)
        prompt_fallback_used = status_code == 200

    if status_code != 200:
        err_text = data.decode("utf-8", errors="ignore")
        raise ValueError(f"ChatAnywhere TTS error {status_code}: {err_text}")

    # Primary path: API returns binary audio content (OpenAI-compatible behavior)
    audio_bytes = data

    # Fallback path: some providers may return JSON with an audio URL/base64
    if "application/json" in content_type:
        resp_json = json.loads(data.decode("utf-8"))

        if isinstance(resp_json, dict) and "error" in resp_json:
            raise ValueError(f"ChatAnywhere TTS error: {resp_json.get('error')}")

        audio_url = None
        if isinstance(resp_json, dict):
            if isinstance(resp_json.get("audio_url"), str):
                audio_url = resp_json["audio_url"]
            elif isinstance(resp_json.get("url"), str):
                audio_url = resp_json["url"]
            elif (
                isinstance(resp_json.get("audio_url"), dict)
                and isinstance(resp_json["audio_url"].get("url"), str)
            ):
                audio_url = resp_json["audio_url"]["url"]

        if audio_url:
            download_resp = requests.get(audio_url, timeout=120)
            download_resp.raise_for_status()
            audio_bytes = download_resp.content
        else:
            audio_base64 = None
            if isinstance(resp_json, dict):
                for key in ("audio", "data", "audio_base64", "b64_json"):
                    value = resp_json.get(key)
                    if isinstance(value, str) and value.strip():
                        audio_base64 = value.strip()
                        break
            if audio_base64:
                if "base64," in audio_base64:
                    audio_base64 = audio_base64.split("base64,", 1)[1]
                audio_bytes = base64.b64decode(audio_base64)
            else:
                raise ValueError(
                    "ChatAnywhere returned JSON but no audio payload (url/base64) was found"
                )

    with open(speech_file_path, "wb") as f:
        f.write(audio_bytes)
    if prompt_fallback_used:
        print("Warning: Current model does not support custom prompt. Generated audio without prompt.")
    print(f"Audio saved to {speech_file_path}")
