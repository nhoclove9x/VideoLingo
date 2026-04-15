import streamlit as st
import os, sys, time
import warnings
import json
from core.st_utils.imports_and_utils import *
from core.st_utils.task_runner import TaskRunner
from core.utils.srt_recheck import run_srt_recheck_pairs
from core import *

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings(
    "ignore",
    message="Torchaudio's I/O functions now support per-call backend dispatch.*",
    category=UserWarning,
)

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"
DUB_AUDIO = "output/dub.mp3"
DUB_SUB_FILE = "output/dub.srt"
DUB_RECHECK_PAIRS = [
    ("output/src.srt", "output/trans.srt"),
    ("output/audio/src_subs_for_audio.srt", "output/audio/trans_subs_for_audio.srt"),
]
SUBTITLE_OUTPUT_FILES = [
    "output/src.srt",
    "output/trans.srt",
    "output/src_trans.srt",
    "output/trans_src.srt",
]


def _merge_dub_into_video_enabled() -> bool:
    try:
        return bool(load_key("merge_dub_into_video"))
    except Exception:
        return True


def _video_speed_factor() -> float:
    try:
        return float(load_key("video_speed.factor"))
    except Exception:
        return 1.0


def _run_dubbing_srt_recheck() -> list[dict]:
    return run_srt_recheck_pairs(DUB_RECHECK_PAIRS)


def _render_subtask_progress(runner_key: str, runner):
    """Render fine-grained progress for long-running step internals."""
    if runner_key != "_text_runner":
        return
    # In text pipeline, index 3 = Summarization and multi-step translation
    if runner.current_step != 3:
        return

    progress_file = "output/log/translation_progress.json"
    if not os.path.exists(progress_file):
        return

    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    stage = str(data.get("stage", "")).lower()
    total = int(data.get("total_chunks", 0) or 0)
    completed = int(data.get("completed_chunks", 0) or 0)
    percent = float(data.get("percent", 0.0) or 0.0)
    message = str(data.get("message", "") or "")

    if stage == "summary":
        st.caption("Sub-progress: preparing summary/context...")
        return

    if total > 0:
        st.caption(f"Sub-progress: translated {completed}/{total} chunks ({percent:.1f}%)")
        st.progress(min(max(percent / 100.0, 0.0), 1.0))
        return

    if message:
        st.caption(f"Sub-progress: {message}")


# ─── Task control UI (auto-refreshes every 1s while task is active) ───


@st.fragment(run_every=1)
def _task_control_panel(runner_key: str):
    """Renders progress bar + pause/stop buttons. Auto-refreshes every 1s."""
    runner = TaskRunner.get(st.session_state, runner_key)

    if runner.state == "idle":
        return

    # Progress
    step_text = (
        f"({runner.current_step + 1}/{runner.total_steps}) {runner.current_label}"
        if runner.current_step >= 0
        else ""
    )

    if runner.is_active:
        if runner.state == "paused":
            st.warning(f"⏸️ {t('Paused')} {step_text}")
        else:
            st.info(f"⏳ {t('Running...')} {step_text}")
        st.progress(runner.progress)
        _render_subtask_progress(runner_key, runner)

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if runner.state == "paused":
                if st.button(
                    f"▶️ {t('Resume')}",
                    key=f"{runner_key}_resume",
                    use_container_width=True,
                ):
                    runner.resume()
                    st.rerun()
            else:
                if st.button(
                    f"⏸️ {t('Pause')}",
                    key=f"{runner_key}_pause",
                    use_container_width=True,
                ):
                    runner.pause()
                    st.rerun()
        with col2:
            if st.button(
                f"⏹️ {t('Stop')}",
                key=f"{runner_key}_stop",
                use_container_width=True,
                type="primary",
            ):
                runner.stop()
                st.rerun()

    elif runner.state == "completed":
        st.success(t("Task completed!"))
        st.progress(1.0)
        runner.reset()
        time.sleep(0.5)
        st.rerun(scope="app")

    elif runner.state == "stopped":
        st.warning(f"⏹️ {t('Task stopped')} {step_text}")
        if st.button(t("OK"), key=f"{runner_key}_ack_stop", use_container_width=True):
            runner.reset()
            st.rerun(scope="app")

    elif runner.state == "error":
        st.error(f"❌ {t('Task error')}: {runner.error_msg}")
        st.caption("Detailed traceback: output/log/task_runner_error.log")
        if st.button(t("OK"), key=f"{runner_key}_ack_error", use_container_width=True):
            runner.reset()
            st.rerun(scope="app")


# ─── Text processing ───


def _get_text_steps():
    """Return the subtitle processing steps as (label, callable) list."""
    steps = [
        (t("Adjust video speed (optional)"), _1_5_speed.adjust_video_speed),
        (t("WhisperX word-level transcription"), _2_asr.transcribe),
        (
            t("Sentence segmentation using NLP and LLM"),
            lambda: (
                _3_1_split_nlp.split_by_spacy(),
                _3_2_split_meaning.split_sentences_by_meaning(),
            ),
        ),
        (
            t("Summarization and multi-step translation"),
            lambda: (
                _4_2_translate.mark_translation_summary_start(),
                _4_1_summarize.get_summary(),
                _4_2_translate.translate_all(),
            ),
        ),
        (
            t("Cutting and aligning long subtitles"),
            lambda: (
                _5_split_sub.split_for_sub_main(),
                _6_gen_sub.align_timestamp_main(),
            ),
        ),
    ]
    if load_key("burn_subtitles"):
        steps.append(
            (
                t("Merging subtitles into the video"),
                _7_sub_into_vid.merge_subtitles_to_video,
            )
        )
    return steps


def _is_subtitle_stage_completed() -> bool:
    """Check subtitle stage completion depending on merge setting."""
    if load_key("burn_subtitles"):
        return os.path.exists(SUB_VIDEO)
    return all(os.path.exists(path) for path in SUBTITLE_OUTPUT_FILES)


def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    runner = TaskRunner.get(st.session_state, "_text_runner")
    text_steps = _get_text_steps()
    step_lines = "<br>".join(
        [f"{i + 1}. {label}" for i, (label, _) in enumerate(text_steps)]
    )

    with st.container(border=True):
        current_speed = _video_speed_factor()
        input_speed = st.number_input(
            t("Video speed before subtitle processing (0.10~10.00)"),
            min_value=0.10,
            max_value=10.0,
            value=current_speed,
            step=0.05,
            format="%.2f",
            disabled=runner.is_active,
        )
        st.caption(t("1.00 = original speed, >1.00 = faster, <1.00 = slower"))
        if abs(float(input_speed) - current_speed) > 1e-6:
            update_key("video_speed.factor", float(input_speed))

        st.markdown(
            f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            {step_lines}
        """,
            unsafe_allow_html=True,
        )

        if not _is_subtitle_stage_completed():
            if runner.is_active:
                _task_control_panel("_text_runner")
            elif runner.is_done:
                _task_control_panel("_text_runner")
            else:
                if st.button(
                    t("Start Processing Subtitles"), key="text_processing_button"
                ):
                    runner.start(text_steps)
                    st.rerun()
        else:
            if load_key("burn_subtitles") and os.path.exists(SUB_VIDEO):
                st.video(SUB_VIDEO)
            else:
                st.info(
                    t(
                        "Subtitle files are ready. Video merge is skipped, so you can do post-production later."
                    )
                )
            download_subtitle_zip_button(text=t("Download All Srt Files"))

            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True


# ─── Audio processing ───


def _get_audio_steps():
    """Return the audio/dubbing processing steps as (label, callable) list."""
    steps = [
        (
            t("Generate audio tasks and chunks"),
            lambda: (
                _8_1_audio_task.gen_audio_task_main(),
                _8_2_dub_chunks.gen_dub_chunks(),
            ),
        ),
        (t("Extract reference audio"), _9_refer_audio.extract_refer_audio_main),
        (t("Generate and merge audio files"), _10_gen_audio.gen_audio),
        (t("Merge full audio"), _11_merge_audio.merge_full_audio),
    ]
    if _merge_dub_into_video_enabled():
        steps.append((t("Merge final audio into video"), _12_dub_to_vid.merge_video_audio))
    return steps


def _is_audio_stage_completed() -> bool:
    if _merge_dub_into_video_enabled():
        return os.path.exists(DUB_VIDEO)
    return os.path.exists(DUB_AUDIO)


def audio_processing_section():
    st.header(t("c. Dubbing"))
    runner = TaskRunner.get(st.session_state, "_audio_runner")
    audio_steps = _get_audio_steps()
    step_lines = "<br>".join(
        [f"{i + 1}. {label}" for i, (label, _) in enumerate(audio_steps)]
    )

    with st.container(border=True):
        st.markdown(
            f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            {step_lines}
        """,
            unsafe_allow_html=True,
        )

        if st.button(
            t("Recheck target SRT and fill missing lines"),
            key="recheck_dubbing_srt_button",
            use_container_width=True,
            disabled=runner.is_active,
        ):
            try:
                recheck_results = _run_dubbing_srt_recheck()
                if not recheck_results:
                    st.warning(
                        t(
                            "No source SRT found for recheck. Please complete subtitle processing first."
                        )
                    )
                else:
                    total_missing = sum(item["filled_missing"] for item in recheck_results)
                    total_empty = sum(item["filled_empty"] for item in recheck_results)
                    changed_files = sum(1 for item in recheck_results if item["changed"])
                    if changed_files > 0:
                        st.success(
                            t(
                                "Recheck complete: target SRT has been updated to match source line count."
                            )
                        )
                    else:
                        st.info(
                            t(
                                "Recheck complete: target SRT already matches source line count."
                            )
                        )
                    st.caption(
                        f"files updated: {changed_files}, missing filled: {total_missing}, empty filled: {total_empty}"
                    )
            except Exception as e:
                st.error(f"❌ {t('Task error')}: {e}")

        if not _is_audio_stage_completed():
            if runner.is_active:
                _task_control_panel("_audio_runner")
            elif runner.is_done:
                _task_control_panel("_audio_runner")
            else:
                if st.button(
                    t("Start Audio Processing"), key="audio_processing_button"
                ):
                    runner.start(audio_steps)
                    st.rerun()
        else:
            st.success(
                t(
                    "Audio processing is complete! You can check the audio files in the `output` folder."
                )
            )
            if _merge_dub_into_video_enabled() and os.path.exists(DUB_VIDEO):
                st.video(DUB_VIDEO)
            else:
                st.info(
                    t(
                        "Dub audio is ready. Video merge is skipped by setting."
                    )
                )
                if os.path.exists(DUB_AUDIO):
                    st.audio(DUB_AUDIO, format="audio/mp3")
                if os.path.exists(DUB_SUB_FILE):
                    st.caption(f"SRT: `{DUB_SUB_FILE}`")
            if st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_audio_processing"):
                cleanup()
                st.rerun()


# ─── Main ───


def main():
    logo_col, _ = st.columns([1, 1])
    with logo_col:
        st.image("docs/logo.png", width="stretch")
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t(
        'Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href="https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh" target="_blank">here</a>! You can also try out our SaaS website at <a href="https://videolingo.io" target="_blank">videolingo.io</a> for free!'
    )
    st.markdown(
        f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>",
        unsafe_allow_html=True,
    )
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)
    download_video_section()
    text_processing_section()
    audio_processing_section()


if __name__ == "__main__":
    main()
