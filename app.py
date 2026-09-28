"""Streamlit app for splitting recap videos, reviewing Whisper subtitles, and rendering clips.

The app deliberately keeps every uploaded file inside a per-browser-session temporary
workspace. Nothing is persisted after the process restarts unless the user downloads it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import srt
import streamlit as st


APP_TITLE = "Recap Video Processor"
CLIP_SECONDS = 120
SPEED = 1.05
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_MODELS = ("base", "small", "tiny")
WORK_ROOT = Path(os.environ.get("RECAP_WORK_DIR", tempfile.gettempdir())) / "recap_video_processor"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
FONT_CHOICES = [
    "Noto Sans Myanmar",
    "Noto Sans",
    "Liberation Sans",
    "Arial",
    "Impact",
]
ASPECT_PRESETS: dict[str, tuple[int, int] | None] = {
    "9:16 vertical (1080 × 1920)": (1080, 1920),
    "16:9 horizontal (1920 × 1080)": (1920, 1080),
    "Keep original frame": None,
}


# ---------- Workspace and command helpers ----------


def init_state() -> None:
    """Create the small amount of Streamlit session state used by the app."""
    defaults: dict[str, Any] = {
        "job_dir": str(WORK_ROOT / f"job_{uuid.uuid4().hex}"),
        "video_fingerprint": None,
        "srt_editor": "",
        "last_zip": None,
        "last_outputs": [],
        "last_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    Path(st.session_state.job_dir).mkdir(parents=True, exist_ok=True)


def reset_job() -> None:
    """Start a clean workspace when the input video is changed."""
    old = Path(st.session_state.job_dir)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    new = WORK_ROOT / f"job_{uuid.uuid4().hex}"
    new.mkdir(parents=True, exist_ok=True)
    st.session_state.job_dir = str(new)
    st.session_state.srt_editor = ""
    st.session_state.last_zip = None
    st.session_state.last_outputs = []
    st.session_state.last_report = None


def run_command(command: list[str], log_path: Path, title: str) -> str:
    """Run FFmpeg/FFprobe and add a useful, bounded record to a local log."""
    printable = " ".join(f'"{item}"' if " " in item else item for item in command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n### {title}\n$ {printable}\n")

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout or ""
    with log_path.open("a", encoding="utf-8") as log:
        log.write(output)

    if completed.returncode != 0:
        tail = output[-5000:] if output else "No FFmpeg output was returned."
        raise RuntimeError(f"{title} failed.\n\n{tail}")
    return output


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "FFmpeg and FFprobe are required. Install FFmpeg on the host, then restart the app."
        )
    filters = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout
    if "subtitles" not in filters:
        raise RuntimeError(
            "This FFmpeg build does not include the `subtitles` (libass) filter. "
            "Install an FFmpeg build compiled with libass."
        )


def file_fingerprint(uploaded_file: Any) -> str:
    """Hash an upload without retaining a second full in-memory copy."""
    digest = hashlib.sha256()
    original_pos = uploaded_file.tell()
    uploaded_file.seek(0)
    while True:
        chunk = uploaded_file.read(8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    uploaded_file.seek(original_pos)
    return f"{uploaded_file.name}:{uploaded_file.size}:{digest.hexdigest()}"


def save_upload(uploaded_file: Any, directory: Path, stem: str, allowed: set[str]) -> Path:
    """Save a Streamlit upload safely, retaining only a permitted file extension."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in allowed:
        suffix = ".bin"
    destination = directory / f"{stem}{suffix}"
    uploaded_file.seek(0)
    with destination.open("wb") as target:
        while True:
            chunk = uploaded_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
    uploaded_file.seek(0)
    return destination


def probe_duration(video_path: Path, log_path: Path) -> float:
    output = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        log_path,
        f"Inspecting duration: {video_path.name}",
    ).strip()
    try:
        duration = float(output.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Could not read a usable duration from {video_path.name}.") from exc
    if duration <= 0:
        raise RuntimeError(f"{video_path.name} has no usable duration.")
    return duration


def has_audio_stream(video_path: Path, log_path: Path) -> bool:
    output = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        log_path,
        f"Inspecting audio stream: {video_path.name}",
    ).strip()
    return bool(output)


# ---------- Whisper and SRT helpers ----------


@st.cache_resource(show_spinner=False, max_entries=1)
def load_whisper_model(model_name: str) -> Any:
    """Load one CPU Whisper model at a time so model switching cannot leak RAM."""
    try:
        import torch
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper dependencies are missing. Run `pip install -r requirements.txt` and restart."
        ) from exc

    # Community Cloud does not provide a CUDA runtime. Limiting intra-op
    # parallelism also reduces peak memory and CPU contention on shared hosts.
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


def make_srt_from_whisper(result: dict[str, Any]) -> str:
    subtitles: list[srt.Subtitle] = []
    for number, segment in enumerate(result.get("segments", []), start=1):
        text = str(segment.get("text", "")).strip()
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if not text or end <= start:
            continue
        subtitles.append(
            srt.Subtitle(
                index=number,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=text,
            )
        )
    if not subtitles:
        raise RuntimeError("Whisper returned no subtitle segments. Check that the video has clear speech.")
    return srt.compose(subtitles, reindex=True)


def transcribe_to_srt(
    video_path: Path,
    language: str | None,
    model_name: str,
    work_dir: Path,
    log_path: Path,
) -> str:
    """Extract mono WAV audio and transcribe it with the selected OpenAI Whisper model."""
    audio_path = work_dir / "whisper_input.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(video_path),
            "-vn",
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        log_path,
        "Extracting audio for Whisper",
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing; install the project requirements.") from exc

    model = load_whisper_model(model_name)
    result = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        fp16=torch.cuda.is_available(),
        verbose=False,
        temperature=0,
        condition_on_previous_text=True,
    )
    return make_srt_from_whisper(result)


def parse_srt_or_raise(srt_text: str) -> list[srt.Subtitle]:
    if not srt_text.strip():
        raise RuntimeError("No subtitles are available. Generate or upload a valid SRT first.")
    try:
        subtitles = list(srt.parse(srt_text))
    except (srt.SRTParseError, ValueError) as exc:
        raise RuntimeError(f"The SRT editor contains invalid subtitle timing or formatting: {exc}") from exc
    if not subtitles:
        raise RuntimeError("The SRT editor contains no subtitle cues.")
    for subtitle in subtitles:
        if subtitle.end <= subtitle.start:
            raise RuntimeError("Every SRT cue must end after it starts.")
    return subtitles


def speed_adjust_subtitles(subtitles: Iterable[srt.Subtitle], speed: float) -> list[srt.Subtitle]:
    """Move subtitle times onto the faster output-video timeline."""
    adjusted: list[srt.Subtitle] = []
    for index, subtitle in enumerate(subtitles, start=1):
        start_seconds = subtitle.start.total_seconds() / speed
        end_seconds = subtitle.end.total_seconds() / speed
        adjusted.append(
            srt.Subtitle(
                index=index,
                start=timedelta(seconds=start_seconds),
                end=timedelta(seconds=max(end_seconds, start_seconds + 0.01)),
                content=subtitle.content,
                proprietary=subtitle.proprietary,
            )
        )
    return adjusted


def write_clip_srt(
    adjusted_subtitles: Iterable[srt.Subtitle],
    clip_start: float,
    clip_duration: float,
    destination: Path,
) -> bool:
    """Write only the cues visible in one clip, rebasing timestamps to zero."""
    clip_end = clip_start + clip_duration
    clip_subtitles: list[srt.Subtitle] = []
    for subtitle in adjusted_subtitles:
        start = subtitle.start.total_seconds()
        end = subtitle.end.total_seconds()
        if end <= clip_start or start >= clip_end:
            continue
        local_start = max(0.0, start - clip_start)
        local_end = min(clip_duration, end - clip_start)
        if local_end <= local_start:
            continue
        clip_subtitles.append(
            srt.Subtitle(
                index=len(clip_subtitles) + 1,
                start=timedelta(seconds=local_start),
                end=timedelta(seconds=local_end),
                content=subtitle.content,
                proprietary=subtitle.proprietary,
            )
        )
    if not clip_subtitles:
        return False
    destination.write_text(srt.compose(clip_subtitles, reindex=True), encoding="utf-8")
    return True


# ---------- FFmpeg filter construction ----------


def ass_colour(hex_colour: str, opacity_percent: int = 100) -> str:
    """Convert #RRGGBB and normal opacity into libass &HAABBGGRR notation."""
    colour = hex_colour.lstrip("#")
    if len(colour) != 6:
        raise ValueError("Colour must be in #RRGGBB format.")
    red, green, blue = int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)
    alpha = round(255 * (1 - max(0, min(100, opacity_percent)) / 100))
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def escape_filter_filename(path: Path) -> str:
    """Escape the characters FFmpeg's filter parser treats specially in a filename."""
    return str(path.resolve()).replace("\\", "\\\\").replace("'", r"\'").replace(":", r"\:")


def build_video_filter(aspect_preset: str) -> str:
    """Return hflip + 1.05x video speed with an optional blurred fit frame."""
    target = ASPECT_PRESETS[aspect_preset]
    if target is None:
        return f"[0:v]hflip,setpts=PTS/{SPEED},setsar=1[v]"
    width, height = target
    return (
        f"[0:v]hflip,setpts=PTS/{SPEED},split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=20:10[bg];"
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
    )


def build_subtitle_filter(srt_path: Path | None, style: dict[str, Any]) -> str:
    """Build a drawbox + libass subtitles video filter for one clip."""
    filters: list[str] = []
    if style["cover_original"]:
        band_height = style["cover_height"] / 100
        top = 1 - band_height
        opacity = style["background_opacity"] / 100
        filters.append(
            "drawbox="
            f"x=0:y=ih*{top:.4f}:w=iw:h=ih*{band_height:.4f}:"
            f"color=0x{style['background_color'].lstrip('#')}@{opacity:.3f}:t=fill"
        )

    if srt_path is not None:
        alignment = {"Bottom": 2, "Middle": 5, "Top": 8}[style["position"]]
        margin_v = style["margin_v"] if style["position"] != "Middle" else 0
        force_style = (
            f"FontName={style['font_name']},"
            f"FontSize={style['font_size']},"
            f"PrimaryColour={ass_colour(style['text_color'])},"
            f"OutlineColour={ass_colour(style['outline_color'])},"
            f"BackColour={ass_colour(style['background_color'], style['background_opacity'])},"
            "BorderStyle=3,"
            f"Outline={style['outline_width']},"
            "Shadow=0,"
            f"Alignment={alignment},"
            f"MarginV={margin_v},"
            "MarginL=36,MarginR=36"
        )
        filters.append(
            f"subtitles=filename='{escape_filter_filename(srt_path)}':charenc=UTF-8:"
            f"force_style='{force_style}'"
        )
    return ",".join(filters)


def create_master_video(
    video_path: Path,
    custom_audio_path: Path | None,
    original_has_audio: bool,
    aspect_preset: str,
    work_dir: Path,
    log_path: Path,
) -> Path:
    """Apply mirror/speed/aspect changes once and force keyframes at clip boundaries."""
    master_path = work_dir / "master_adjusted.mp4"
    output_duration = probe_duration(video_path, log_path) / SPEED
    command = ["ffmpeg", "-y", "-hide_banner", "-i", str(video_path)]
    if custom_audio_path is not None:
        command += ["-i", str(custom_audio_path)]

    filter_parts = [build_video_filter(aspect_preset)]
    audio_available = custom_audio_path is not None or original_has_audio
    if custom_audio_path is not None:
        filter_parts.append(f"[1:a]atempo={SPEED},apad=whole_dur={output_duration:.3f}[a]")
    elif original_has_audio:
        filter_parts.append(f"[0:a]atempo={SPEED},apad=whole_dur={output_duration:.3f}[a]")

    command += ["-filter_complex", ";".join(filter_parts), "-map", "[v]"]
    if audio_available:
        command += ["-map", "[a]"]

    command += [
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{CLIP_SECONDS})",
    ]
    if audio_available:
        # Pad a short custom audio track with silence; -t below keeps it exactly
        # aligned with the accelerated video rather than letting padding run forever.
        command += ["-c:a", "aac", "-b:a", "192k"]
    command += ["-t", f"{output_duration:.3f}", "-movflags", "+faststart", str(master_path)]
    run_command(command, log_path, "Applying video adjustments")
    return master_path


def split_into_two_minute_clips(master_path: Path, work_dir: Path, log_path: Path) -> list[Path]:
    """Losslessly segment a master that has forced keyframes every 120 seconds."""
    raw_dir = work_dir / "raw_clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = raw_dir / "raw_%03d.mp4"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(master_path),
            "-map",
            "0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(CLIP_SECONDS),
            "-segment_time_delta",
            "0.05",
            "-reset_timestamps",
            "1",
            "-segment_format",
            "mp4",
            "-avoid_negative_ts",
            "make_zero",
            str(output_pattern),
        ],
        log_path,
        "Splitting into sequential two-minute clips",
    )
    clips = sorted(raw_dir.glob("raw_*.mp4"))
    if not clips:
        raise RuntimeError("FFmpeg did not produce any video clips.")
    return clips


def render_clips(
    raw_clips: list[Path],
    adjusted_subtitles: list[srt.Subtitle],
    style: dict[str, Any],
    work_dir: Path,
    log_path: Path,
    progress: Callable[[str, float], None] | None = None,
) -> tuple[list[Path], list[Path]]:
    """Burn each rebased clip SRT and retain the matching SRT beside final output."""
    output_dir = work_dir / "output_clips"
    subtitle_dir = work_dir / "clip_subtitles"
    output_dir.mkdir(parents=True, exist_ok=True)
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    clip_srts: list[Path] = []
    timeline_start = 0.0
    for index, raw_clip in enumerate(raw_clips, start=1):
        clip_duration = probe_duration(raw_clip, log_path)
        clip_srt = subtitle_dir / f"clip_{index:03d}.srt"
        has_subtitles = write_clip_srt(adjusted_subtitles, timeline_start, clip_duration, clip_srt)
        if has_subtitles:
            clip_srts.append(clip_srt)

        subtitle_filter = build_subtitle_filter(clip_srt if has_subtitles else None, style)
        final_clip = output_dir / f"recap_clip_{index:03d}.mp4"
        command = ["ffmpeg", "-y", "-hide_banner", "-i", str(raw_clip)]
        if subtitle_filter:
            command += ["-vf", subtitle_filter]
        command += [
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(final_clip),
        ]
        run_command(command, log_path, f"Burning styled subtitles: clip {index}")
        rendered.append(final_clip)
        timeline_start += clip_duration
        if progress:
            progress(
                f"Burning subtitles into clip {index} of {len(raw_clips)}…",
                0.55 + 0.40 * index / len(raw_clips),
            )
    return rendered, clip_srts


def create_output_zip(
    clips: list[Path],
    clip_srts: list[Path],
    reviewed_srt: Path,
    report_path: Path,
    log_path: Path,
    work_dir: Path,
) -> Path:
    zip_path = work_dir / "recap_video_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for clip in clips:
            archive.write(clip, arcname=f"clips/{clip.name}")
        for clip_srt in clip_srts:
            archive.write(clip_srt, arcname=f"clip_subtitles/{clip_srt.name}")
        archive.write(reviewed_srt, arcname="reviewed_subtitles.srt")
        archive.write(report_path, arcname="processing_report.txt")
        archive.write(log_path, arcname="ffmpeg_command_log.txt")
    return zip_path


# ---------- Pipeline ----------


def process_video(
    video_upload: Any,
    custom_audio_upload: Any | None,
    subtitle_text: str,
    language: str | None,
    model_name: str,
    aspect_preset: str,
    style: dict[str, Any],
    progress: Callable[[str, float], None],
) -> dict[str, Any]:
    """Execute the complete deterministic render pipeline for the current upload."""
    require_ffmpeg()
    work_dir = Path(st.session_state.job_dir)
    log_path = work_dir / "ffmpeg_command_log.txt"
    log_path.write_text("Recap Video Processor command log\n", encoding="utf-8")

    progress("Saving uploaded media…", 0.02)
    video_path = save_upload(video_upload, work_dir, "input_video", VIDEO_EXTENSIONS)
    custom_audio_path = (
        save_upload(custom_audio_upload, work_dir, "custom_audio", AUDIO_EXTENSIONS)
        if custom_audio_upload is not None
        else None
    )
    input_duration = probe_duration(video_path, log_path)
    source_has_audio = has_audio_stream(video_path, log_path)

    # The renderer accepts a user-edited SRT. If the user intentionally selects
    # one-click mode without subtitles, it generates an SRT and packages it for later review.
    if subtitle_text.strip():
        original_subtitles = parse_srt_or_raise(subtitle_text)
        reviewed_srt_text = srt.compose(original_subtitles, reindex=True)
    else:
        transcription_media = video_path if source_has_audio else custom_audio_path
        if transcription_media is None:
            raise RuntimeError(
                "The source video has no audio and no custom audio was supplied. Upload a reviewed SRT, "
                "a custom voice file, or use a source video that contains speech for Whisper to transcribe."
            )
        progress(f"Transcribing with OpenAI Whisper {model_name}…", 0.08)
        reviewed_srt_text = transcribe_to_srt(transcription_media, language, model_name, work_dir, log_path)
        original_subtitles = parse_srt_or_raise(reviewed_srt_text)

    reviewed_srt = work_dir / "reviewed_subtitles.srt"
    reviewed_srt.write_text(reviewed_srt_text, encoding="utf-8")

    progress("Mirroring, speeding up, and creating the blurred-fit master…", 0.20)
    master_path = create_master_video(
        video_path,
        custom_audio_path,
        source_has_audio,
        aspect_preset,
        work_dir,
        log_path,
    )

    progress("Splitting the master at two-minute boundaries…", 0.48)
    raw_clips = split_into_two_minute_clips(master_path, work_dir, log_path)
    adjusted_subtitles = speed_adjust_subtitles(original_subtitles, SPEED)

    rendered_clips, clip_srts = render_clips(
        raw_clips,
        adjusted_subtitles,
        style,
        work_dir,
        log_path,
        progress,
    )

    output_duration = probe_duration(master_path, log_path)
    report_path = work_dir / "processing_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "Recap Video Processor report",
                "=" * 30,
                f"Input duration: {input_duration:.2f} seconds",
                f"Output duration after {SPEED}x speed: {output_duration:.2f} seconds",
                f"Two-minute clips: {len(rendered_clips)}",
                f"Custom audio used: {'yes' if custom_audio_path else 'no'}",
                f"Frame preset: {aspect_preset}",
                "Video codec settings: libx264, CRF 18, preset medium",
                "Subtitle timing was shifted by 1/1.05 to match the sped-up video.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    progress("Packaging clips, per-clip SRT files, and the processing report…", 0.98)
    zip_path = create_output_zip(rendered_clips, clip_srts, reviewed_srt, report_path, log_path, work_dir)
    progress("Complete.", 1.0)
    return {
        "zip": zip_path,
        "clips": rendered_clips,
        "reviewed_srt": reviewed_srt,
        "report": report_path,
        "used_generated_srt": not subtitle_text.strip(),
    }


# ---------- Streamlit interface ----------


def sidebar_settings() -> tuple[str | None, str, str, dict[str, Any]]:
    """Render processing controls and return normalized configuration."""
    with st.sidebar:
        st.header("Processing settings")
        st.caption("Video clips are always mirrored and sped up slightly at **1.05×**.")

        language_label = st.selectbox(
            "Spoken language for Whisper",
            ["Burmese / Myanmar (recommended)", "Auto-detect", "English"],
            index=0,
            help="Choosing Burmese avoids language-detection mistakes when the speech is primarily Myanmar language.",
        )
        language = {"Burmese / Myanmar (recommended)": "my", "Auto-detect": None, "English": "en"}[language_label]
        model_name = st.selectbox(
            "Whisper model (Community Cloud-safe)",
            list(WHISPER_MODELS),
            index=list(WHISPER_MODELS).index(DEFAULT_WHISPER_MODEL)
            if DEFAULT_WHISPER_MODEL in WHISPER_MODELS
            else 0,
            help=(
                "base is the default because Community Cloud has about 2.7 GB RAM. "
                "small is more accurate but uses more memory; tiny is the fallback for very tight limits. "
                "large-v3/turbo/medium are intentionally disabled on this deployment target."
            ),
        )
        aspect_preset = st.selectbox("Output frame", list(ASPECT_PRESETS), index=0)

        st.divider()
        st.subheader("Subtitle style")
        font_name = st.selectbox("Font style", FONT_CHOICES, index=0)
        font_size = st.slider("Font size", min_value=16, max_value=120, value=54, step=1)
        text_color = st.color_picker("Text color", "#FFFFFF")
        outline_color = st.color_picker("Outline color", "#000000")
        outline_width = st.slider("Outline width", min_value=0, max_value=12, value=3, step=1)
        background_color = st.color_picker("Subtitle background box color", "#000000")
        background_opacity = st.slider("Subtitle background box opacity", 0, 100, 92, 1)

        cover_original = st.checkbox(
            "Cover original lower-third subtitles / logos",
            value=True,
            help="Adds a solid coloured band beneath the subtitle box. Use it when original burnt-in text is in the lower area.",
        )
        cover_height = st.slider("Cover band height (% of frame)", 8, 45, 25, 1, disabled=not cover_original)
        position = st.selectbox("Subtitle position", ["Bottom", "Middle", "Top"], index=0)
        margin_v = st.slider("Top/bottom margin (px)", 0, 240, 70, 5, disabled=position == "Middle")

    style = {
        "font_name": font_name,
        "font_size": font_size,
        "text_color": text_color,
        "outline_color": outline_color,
        "outline_width": outline_width,
        "background_color": background_color,
        "background_opacity": background_opacity,
        "cover_original": cover_original,
        "cover_height": cover_height,
        "position": position,
        "margin_v": margin_v,
    }
    return language, model_name, aspect_preset, style


def render_completed_outputs() -> None:
    """Keep downloads visible on normal Streamlit reruns after a completed render."""
    zip_name = st.session_state.get("last_zip")
    if not zip_name:
        return
    zip_path = Path(zip_name)
    if not zip_path.exists():
        return
    st.success(f"Last render is ready: {len(st.session_state.get('last_outputs', []))} clip(s).")
    with zip_path.open("rb") as output:
        st.download_button(
            "Download all clips + reviewed SRT (.zip)",
            data=output.read(),
            file_name="recap_video_outputs.zip",
            mime="application/zip",
            type="primary",
        )
    report_name = st.session_state.get("last_report")
    if report_name and Path(report_name).exists():
        st.caption("The ZIP includes individual clips, clip-specific SRT files, the reviewed SRT, and an FFmpeg command log.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.write(
        "Upload a recap video, correct its Whisper transcript if needed, then render "
        "mirrored 2-minute clips with a clean subtitle mask and high-quality FFmpeg export."
    )
    st.info(
        "**Recommended review flow:** generate the SRT first, edit it below, then render. "
        "The final button can also auto-transcribe if you deliberately skip review."
    )

    language, model_name, aspect_preset, style = sidebar_settings()

    video_upload = st.file_uploader(
        "1. Upload a recap video",
        type=[item.lstrip(".") for item in sorted(VIDEO_EXTENSIONS)],
        help="MP4 and MOV are usually the most reliable inputs. Very long 1080p/4K videos need substantial CPU/GPU time and disk space.",
    )
    custom_audio_upload = st.file_uploader(
        "Optional: upload a custom voice-cloned audio file",
        type=[item.lstrip(".") for item in sorted(AUDIO_EXTENSIONS)],
        help="This replaces the original audio. The app speeds it to 1.05× to remain synchronized with the video. If it ends early, remaining video is silent rather than repeated.",
    )

    if video_upload is None:
        st.caption("Your uploaded media is processed in a temporary workspace and is not kept by this app after the server restarts.")
        render_completed_outputs()
        return

    fingerprint = file_fingerprint(video_upload)
    if st.session_state.video_fingerprint != fingerprint:
        reset_job()
        st.session_state.video_fingerprint = fingerprint

    st.caption(f"Selected video: **{video_upload.name}** ({video_upload.size / 1024 / 1024:.1f} MB)")

    st.subheader("2. Generate and review subtitles")
    subtitle_upload = st.file_uploader(
        "Optional: load an already-edited SRT instead",
        type=["srt"],
        help="Its timestamps must correspond to the original, unsped-up source video. The app applies the 1.05× timing adjustment automatically.",
    )

    col_generate, col_load, col_tip = st.columns([1.2, 1.2, 2.6])
    with col_generate:
        generate_clicked = st.button("Generate editable SRT", use_container_width=True)
    with col_load:
        load_clicked = st.button("Load uploaded SRT", use_container_width=True, disabled=subtitle_upload is None)
    with col_tip:
        st.caption("`base` is the default multilingual model for Community Cloud; use `small` only if the app has enough memory.")

    # These actions occur before the editor widget is instantiated, allowing safe
    # initialization of its Streamlit state key.
    if load_clicked and subtitle_upload is not None:
        try:
            loaded = subtitle_upload.getvalue().decode("utf-8-sig")
            parse_srt_or_raise(loaded)
            st.session_state.srt_editor = loaded
            st.success("Loaded the SRT into the editor.")
        except UnicodeDecodeError:
            st.error("The uploaded SRT must be UTF-8 encoded.")
        except RuntimeError as exc:
            st.error(str(exc))

    if generate_clicked:
        stage = st.empty()
        try:
            require_ffmpeg()
            work_dir = Path(st.session_state.job_dir)
            log_path = work_dir / "ffmpeg_command_log.txt"
            log_path.write_text("Recap Video Processor command log\n", encoding="utf-8")
            source_path = save_upload(video_upload, work_dir, "input_video", VIDEO_EXTENSIONS)
            transcription_source = source_path
            if not has_audio_stream(source_path, log_path):
                if custom_audio_upload is None:
                    raise RuntimeError(
                        "This source video has no audio track to transcribe. Upload a custom voice file "
                        "or load a reviewed SRT instead."
                    )
                transcription_source = save_upload(custom_audio_upload, work_dir, "custom_audio", AUDIO_EXTENSIONS)
            if not has_audio_stream(transcription_source, log_path):
                raise RuntimeError("The selected transcription source does not contain a usable audio track.")
            stage.info(f"Extracting audio and transcribing with OpenAI Whisper {model_name}. This can take a while…")
            generated_srt = transcribe_to_srt(transcription_source, language, model_name, work_dir, log_path)
            st.session_state.srt_editor = generated_srt
            (work_dir / "generated_subtitles.srt").write_text(generated_srt, encoding="utf-8")
            stage.success("SRT generated. Review or correct it in the editor before rendering.")
        except Exception as exc:  # surfaced in the UI with FFmpeg's useful error tail
            stage.error(str(exc))

    subtitle_text = st.text_area(
        "Editable SRT transcript",
        key="srt_editor",
        height=330,
        placeholder="Click ‘Generate editable SRT’ or load your own .srt file. You can correct text and timing here.",
    )
    if subtitle_text.strip():
        try:
            parsed = parse_srt_or_raise(subtitle_text)
            st.caption(f"SRT ready for rendering: {len(parsed)} cue(s). Timings are currently on the original video timeline.")
            st.download_button(
                "Download current reviewed SRT",
                data=srt.compose(parsed, reindex=True).encode("utf-8"),
                file_name="reviewed_subtitles.srt",
                mime="text/plain",
            )
        except RuntimeError as exc:
            st.error(str(exc))

    st.subheader("3. One-click processing and export")
    auto_note = "Uses your reviewed SRT." if subtitle_text.strip() else "No SRT entered: this will auto-transcribe and render without a review pause."
    st.caption(f"{auto_note} Output is split into sequential **{CLIP_SECONDS // 60}-minute** clips and encoded with `-crf 18 -preset medium`.")
    process_clicked = st.button("Process video and create downloadable clips", type="primary", use_container_width=True)

    if process_clicked:
        progress_bar = st.progress(0, text="Preparing…")
        status_line = st.empty()

        def update_progress(message: str, fraction: float) -> None:
            fraction = max(0.0, min(1.0, fraction))
            progress_bar.progress(fraction, text=message)
            status_line.caption(message)

        try:
            result = process_video(
                video_upload,
                custom_audio_upload,
                subtitle_text,
                language,
                model_name,
                aspect_preset,
                style,
                update_progress,
            )
            st.session_state.last_zip = str(result["zip"])
            st.session_state.last_outputs = [str(path) for path in result["clips"]]
            st.session_state.last_report = str(result["report"])
            if result["used_generated_srt"]:
                st.warning(
                    "The SRT was auto-generated and used immediately because the editor was empty. "
                    "A copy is inside the ZIP for review; use the recommended review flow next time when corrections matter."
                )
            st.success(f"Processing complete: {len(result['clips'])} two-minute clip(s) created.")
        except Exception as exc:
            progress_bar.empty()
            st.error(str(exc))

    render_completed_outputs()
    with st.expander("Operational notes", expanded=False):
        st.markdown(
            "- **Font availability:** the font must be installed on the deployment host. `Noto Sans Myanmar` is recommended for Burmese.\n"
            "- **Masking:** the solid lower-third cover band hides originals that are located in that area. Move the subtitle position or disable the band for different layouts.\n"
            "- **Audio:** custom audio is not looped. If it is shorter than the sped-up video, the remaining section is silent.\n"
            "- **Accuracy:** `base` keeps this Community Cloud deployment usable; `small` can improve Burmese accuracy but may exceed the shared memory limit. `large-v3`, `turbo`, and `medium` require a GPU-capable host."
        )


if __name__ == "__main__":
    main()
