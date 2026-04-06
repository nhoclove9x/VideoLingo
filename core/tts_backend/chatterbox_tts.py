import inspect
import threading
import gc
import os
from pathlib import Path

from core.utils import except_handler, load_key

_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_GENERATE_LOCK = threading.Lock()

_MISSING_MODULE_TO_PACKAGE = {
    "perth": "resemble-perth",
    "pkg_resources": "setuptools",
    "s3tokenizer": "s3tokenizer",
    "conformer": "conformer",
    "omegaconf": "omegaconf",
    "pyloudnorm": "pyloudnorm",
    "spacy_pkuseg": "spacy-pkuseg",
    "pykakasi": "pykakasi",
    "safetensors": "safetensors",
    "diffusers": "diffusers",
}


def _prepare_chatterbox_runtime_env():
    """Set runtime env vars to avoid cache/numba issues in constrained setups."""
    cache_root = Path("output/cache")
    hf_cache = cache_root / "huggingface"
    numba_cache = cache_root / "numba"
    hf_cache.mkdir(parents=True, exist_ok=True)
    numba_cache.mkdir(parents=True, exist_ok=True)

    # Avoid librosa/numba compilation path on environments with numba ABI/version issues.
    os.environ.setdefault("LIBROSA_DISABLE_NUMBA", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache.resolve()))
    # Old runs may have set this and trigger numba edge-case crashes.
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        os.environ.pop("NUMBA_DISABLE_JIT", None)

    # Ensure huggingface model cache is writable.
    os.environ.setdefault("HF_HOME", str(hf_cache.resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((hf_cache / "hub").resolve()))

    _patch_librosa_float32()


def _resolve_device(device_pref: str) -> str:
    import torch

    if device_pref and device_pref != "auto":
        return device_pref
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_perth_watermarker_available():
    """
    Chatterbox always instantiates `perth.PerthImplicitWatermarker()`.
    In some environments, `resemble-perth` imports but exposes this as `None`
    (for example when an internal optional dependency is missing). In that case,
    fallback to `DummyWatermarker` so TTS can still run.
    """
    try:
        import perth
    except ModuleNotFoundError:
        return

    if callable(getattr(perth, "PerthImplicitWatermarker", None)):
        return

    dummy_cls = getattr(perth, "DummyWatermarker", None)
    if not callable(dummy_cls):
        try:
            from perth.dummy_watermarker import DummyWatermarker
        except Exception as e:
            raise ImportError(
                "Perth watermarker backend is unavailable in this environment. "
                "Install `setuptools` and `resemble-perth`."
            ) from e
        dummy_cls = DummyWatermarker

    perth.PerthImplicitWatermarker = dummy_cls
    print(
        "Warning: `perth.PerthImplicitWatermarker` is unavailable. "
        "Falling back to `DummyWatermarker` (no watermark)."
    )


def _patch_s3tokenizer_dtype():
    """Patch chatterbox tokenizer to avoid float64/float32 matmul crashes."""
    try:
        import torch
        from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer
    except Exception:
        return

    if getattr(S3Tokenizer, "_videolingo_dtype_patch", False):
        return

    original_prepare_audio = S3Tokenizer._prepare_audio

    def _prepare_audio_fp32(self, wavs):
        processed = original_prepare_audio(self, wavs)
        out = []
        for wav in processed:
            if torch.is_tensor(wav) and wav.dtype != torch.float32:
                wav = wav.float()
            out.append(wav)
        return out

    S3Tokenizer._prepare_audio = _prepare_audio_fp32
    S3Tokenizer._videolingo_dtype_patch = True


def _patch_librosa_float32():
    """Force librosa I/O/resample outputs to float32 for chatterbox compatibility."""
    try:
        import numpy as np
        import librosa
    except Exception:
        return

    if getattr(librosa, "_videolingo_fp32_patch", False):
        return

    original_load = librosa.load
    original_resample = librosa.resample

    def _load_fp32(*args, **kwargs):
        kwargs.setdefault("dtype", np.float32)
        y, sr = original_load(*args, **kwargs)
        if hasattr(y, "astype"):
            y = y.astype(np.float32, copy=False)
        return y, sr

    def _resample_fp32(*args, **kwargs):
        y = original_resample(*args, **kwargs)
        if hasattr(y, "astype"):
            y = y.astype(np.float32, copy=False)
        return y

    librosa.load = _load_fp32
    librosa.resample = _resample_fp32
    librosa._videolingo_fp32_patch = True


def _load_chatterbox_model(model_variant: str, device: str):
    import torch

    cache_key = (model_variant, device)
    with _MODEL_CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        # Chatterbox upstream provides this map_location workaround for Apple Silicon.
        original_torch_load = torch.load
        if device == "mps":
            mps_device = torch.device("mps")

            def patched_torch_load(*args, **kwargs):
                kwargs.setdefault("map_location", mps_device)
                return original_torch_load(*args, **kwargs)

            torch.load = patched_torch_load

        try:
            try:
                _ensure_perth_watermarker_available()
                if model_variant == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS

                    _patch_s3tokenizer_dtype()
                    model = ChatterboxTurboTTS.from_pretrained(device=device)
                elif model_variant == "multilingual":
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                    _patch_s3tokenizer_dtype()
                    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                else:
                    from chatterbox.tts import ChatterboxTTS

                    _patch_s3tokenizer_dtype()
                    model = ChatterboxTTS.from_pretrained(device=device)
            except ModuleNotFoundError as e:
                missing = getattr(e, "name", None)
                pip_package = _MISSING_MODULE_TO_PACKAGE.get(missing, missing)
                hint = (
                    f" Missing dependency: `{missing}`. "
                    f"Try `python -m pip install {pip_package}`."
                    if pip_package
                    else ""
                )
                raise ImportError(
                    "chatterbox-tts is installed without some dependencies." + hint
                ) from e
            except TypeError as e:
                msg = str(e).lower()
                if "nonetype" in msg and "callable" in msg:
                    raise ImportError(
                        "Perth watermarker backend is unavailable in this environment. "
                        "Install `setuptools` and `resemble-perth`, then retry."
                    ) from e
                raise
        finally:
            if device == "mps":
                torch.load = original_torch_load

        _MODEL_CACHE[cache_key] = model
        return model


def _build_generate_kwargs(model, model_variant: str, reference_audio_path: str):
    kwargs = {}
    params = inspect.signature(model.generate).parameters

    if reference_audio_path and "audio_prompt_path" in params:
        kwargs["audio_prompt_path"] = reference_audio_path

    if "exaggeration" in params:
        kwargs["exaggeration"] = float(load_key("chatterbox_tts.exaggeration"))

    if "cfg_weight" in params:
        kwargs["cfg_weight"] = float(load_key("chatterbox_tts.cfg_weight"))

    if model_variant == "multilingual" and "language_id" in params:
        language_id = load_key("chatterbox_tts.language_id").strip()
        if language_id:
            kwargs["language_id"] = language_id

    return kwargs


def _resolve_reference_audio_path(
    clone_mode: str, custom_path: str, number: int | None
) -> str:
    if clone_mode == "none":
        return ""

    if clone_mode == "custom":
        path = custom_path.strip()
        if not path:
            raise ValueError(
                "Chatterbox clone mode is `custom` but no reference path is set."
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"Chatterbox reference audio not found: {path}")
        return path

    first_ref = Path("output/audio/refers/1.wav")
    if clone_mode == "first_ref":
        if not first_ref.exists():
            raise FileNotFoundError(
                "Reference audio not found at output/audio/refers/1.wav. "
                "Please run `Extract reference audio` step first."
            )
        return str(first_ref)

    if clone_mode == "per_segment":
        if number is not None:
            segment_ref = Path(f"output/audio/refers/{int(number)}.wav")
            if segment_ref.exists():
                return str(segment_ref)
        if first_ref.exists():
            return str(first_ref)
        raise FileNotFoundError(
            "Reference audio for per-segment clone is missing. "
            "Expected output/audio/refers/<number>.wav (or fallback 1.wav)."
        )

    raise ValueError(
        f"Invalid chatterbox clone mode: {clone_mode}. "
        "Use one of: none, custom, first_ref, per_segment."
    )


@except_handler("Failed to generate audio using Chatterbox TTS", retry=1, delay=1)
def chatterbox_tts(text: str, save_path: str, number: int | None = None):
    _prepare_chatterbox_runtime_env()
    try:
        import torch
        import torchaudio as ta
    except ImportError as e:
        raise ImportError(
            "chatterbox_tts requires optional package `chatterbox-tts`. "
            "Install with: python -m pip install --no-deps chatterbox-tts"
        ) from e

    model_variant = load_key("chatterbox_tts.model_variant")
    device_pref = load_key("chatterbox_tts.device")
    clone_mode = load_key("chatterbox_tts.clone_mode")
    custom_ref_path = load_key("chatterbox_tts.reference_audio_path")
    reference_audio_path = _resolve_reference_audio_path(
        clone_mode=clone_mode, custom_path=custom_ref_path, number=number
    )

    if model_variant == "turbo" and not reference_audio_path:
        raise ValueError(
            "Chatterbox-Turbo requires clone reference audio. "
            "Please set clone mode to `custom`, `first_ref`, or `per_segment`."
        )

    device = _resolve_device(device_pref)
    model = _load_chatterbox_model(model_variant, device)
    generate_kwargs = _build_generate_kwargs(model, model_variant, reference_audio_path)

    with _GENERATE_LOCK:
        with torch.inference_mode():
            wav = model.generate(text, **generate_kwargs)

    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu()
    else:
        wav = torch.tensor(wav)

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    sample_rate = int(getattr(model, "sr", 24000))
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(speech_file_path), wav, sample_rate)
    del wav
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    print(
        f"Chatterbox audio saved to {speech_file_path} "
        f"(variant={model_variant}, device={device})"
    )
