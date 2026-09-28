---
title: Recap Video Processor
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: streamlit
---

# Recap Video Processor

A Streamlit app for turning a long recap video into sequential **two-minute** clips with high-quality, editable subtitles. It uses local multilingual **OpenAI Whisper** (Community Cloud defaults to `base`; `small` is also available) and FFmpeg for the final render.

## What this project does

| Requirement | Implementation |
|---|---|
| 2-minute splitter | Makes a processed master with forced keyframes every 120 seconds, then segments it into sequential clips. |
| Burmese subtitles | Uses local multilingual OpenAI Whisper, defaulting to `base` with the Burmese language hint (`my`) so it fits Community Cloud's shared memory. `small` is available as a higher-accuracy, higher-memory option. |
| Review before burn-in | Generates a UTF-8 SRT in an editable Streamlit text area. You can download it, correct it, re-upload it, and then render. |
| Styled subtitle overlay | Controls for font, size, text and outline colours, a solid subtitle box, and an optional lower-third mask band. |
| Hide existing lower-third text/logos | The optional solid lower-third band is drawn **before** the subtitle overlay. Adjust its height, colour, opacity, and subtitle position in the sidebar. |
| Custom voice-cloned audio | Optionally replaces the original audio with the uploaded audio. It is sped to 1.05× to match the video. For a silent source video it is also used as the Whisper transcription source. It is not looped; a too-short track is silent after it ends. |
| Video adjustments | Horizontal mirror, fixed 1.05× speed, and a blurred background canvas for 9:16 or 16:9 fitting. |
| High-quality export | Every final clip uses `libx264 -crf 18 -preset medium`, AAC audio, and `yuv420p` for broad mobile compatibility. |
| Downloadable result | One ZIP containing the final MP4 clips, clip-specific SRTs, reviewed full SRT, processing report, and FFmpeg command log. |

## Important operating notes

> **Review-first is the accuracy workflow.** Click **Generate editable SRT**, correct the text/timestamps, then click **Process video and create downloadable clips**. The rendering button can automatically transcribe if no SRT is present, but that deliberately bypasses the review pause.

> **Community Cloud resource choice.** The app defaults to the multilingual `base` model on Community Cloud. `small` can improve accuracy but needs more memory. `medium`, `turbo`, and `large-v3` are intentionally not offered in this deployment because the official Whisper reference lists approximate requirements of ~5 GB, ~6 GB, and ~10 GB VRAM respectively; Community Cloud has approximately 2.7 GB maximum memory.

> **The source SRT timeline is the original timeline.** The app divides subtitle timestamps by `1.05` before burning them, so they stay synchronized after the video is sped up.

## Project files

```text
recap-video-processor/
├── app.py                     # Streamlit UI and complete FFmpeg / Whisper pipeline
├── requirements.txt           # Python packages
├── packages.txt               # FFmpeg and font packages for cloud Linux hosts
├── runtime.txt                # Python 3.12 for Streamlit Community Cloud
├── .streamlit/config.toml     # Upload size and UI theme (does not change port)
└── README.md                  # This guide and Hugging Face Space metadata
```

## Local setup

### 1. Install system dependencies

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install -y ffmpeg fontconfig fonts-noto-core fonts-liberation python3.12 python3.12-venv
```

`fonts-noto-core` provides the Noto family needed for reliable Myanmar/Burmese glyph coverage. On Windows or macOS, install FFmpeg and a Burmese-capable Noto font, then ensure `ffmpeg` is available on your `PATH`.

Verify that your FFmpeg build can burn subtitles:

```bash
ffmpeg -filters | grep subtitles
```

The result must include `subtitles`. If it does not, install an FFmpeg build compiled with **libass**.

### 2. Create a virtual environment and install Python packages

```bash
cd recap-video-processor
python3.12 -m venv .venv
source .venv/bin/activate                  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first use of each Whisper model downloads its model weights. No `OPENAI_API_KEY` is required because this app runs the open-source Whisper model on the host.

### 3. Start the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`). For a phone on the same Wi-Fi network, start it with:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then open `http://YOUR_COMPUTER_LAN_IP:8501` from the phone. Keep the computer running while processing.

## How to use the app

1. **Upload the video.** MP4 or MOV is recommended. Set the output frame in the sidebar; 9:16 vertical is the default for mobile recap clips.
2. **Optional: upload custom voice audio.** It replaces the original audio during final render. The audio should correspond to the original video timeline; when the video has no audio it is also transcribed by Whisper.
3. **Generate the SRT.** Leave “Burmese / Myanmar” and `base` selected for the most reliable Community Cloud run. Try `small` if you have enough memory and want higher accuracy.
4. **Review the editable SRT.** Correct spelling, line breaks, and timestamps in the text area. Download it if you want an external backup. You may instead upload a pre-edited UTF-8 `.srt`.
5. **Set overlay controls.** Pick a host-installed font, colours, opacity, and the solid lower-third cover band. For original burned-in subtitles or logos near the bottom, leave the band enabled and increase its height until it covers the unwanted area.
6. **Process and download.** The final button mirrors, speeds, fits, splits, burns subtitles, and packages every clip into one ZIP.

### Font guidance for Burmese

Use **Noto Sans Myanmar** whenever possible. `Arial` and `Impact` are exposed because they are familiar choices, but they may be absent on Linux and may not contain every Myanmar glyph. FFmpeg/libass will then silently fall back to another installed font; inspect a short test clip before processing a full production video.

## Cloud deployment choices

This app performs transcription and video encoding inside the web app process, so cloud hardware matters more than the Streamlit UI itself. The Community Cloud-compatible configuration is deliberately CPU-only and defaults to Whisper `base`.

| Approach | Tradeoffs | Cost | Setup complexity |
|---|---|---:|---|
| **Streamlit Community Cloud (CPU)** | Lightest deployment path and good for the UI, short clips, SRT editing, and `base`/some `small` transcription. Larger models exceed shared memory; model weights can be re-downloaded when the app restarts. | Free tier available; platform limits apply. | Low |
| **Hugging Face Streamlit Space with a GPU** | Better practical route for long Burmese videos with `large-v3`; still has storage, upload, session, and selected-hardware limits. Choose a private Space if the videos are sensitive. | Depends on selected hardware/account. | Medium |
| **Your own GPU workstation / server** | Best control over model cache, fonts, disk, runtime duration, and private media. You must keep the host online and secure it yourself. | Uses hardware you already operate or your host cost. | Medium–high |

### Option A — Deploy to Streamlit Community Cloud

Streamlit Community Cloud supports a root-level `requirements.txt` for Python packages and `packages.txt` for Debian `apt-get` packages such as FFmpeg. This repository already includes both.

1. Create an empty GitHub repository, then commit this project:

   ```bash
   cd recap-video-processor
   git init
   git add app.py requirements.txt packages.txt runtime.txt README.md .streamlit/config.toml
   git commit -m "Add recap video processor"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/recap-video-processor.git
   git push -u origin main
   ```

2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3. Select **Create app**, choose the repository and `main` branch, and set the main file to `app.py`.
4. In Advanced settings, explicitly select **Python 3.12** (the included `runtime.txt` also specifies it). Do not leave an existing app configured for Python 3.14.
5. Deploy and open the generated HTTPS URL from your phone browser.
6. Confirm the build log shows installation of `ffmpeg` from `packages.txt`, then use a short test video before uploading production-length media.

The repository pins a CPU-only PyTorch wheel and Python 3.12. This is important: `openai-whisper` has a broad `torch` dependency, and leaving it unpinned can cause pip to install a CUDA-enabled PyTorch distribution whose download and runtime memory exceed Community Cloud. The app also caches only one Whisper model at a time.

If `small` still exceeds the instance's available memory, use `base` (the default). For `medium`, `turbo`, or `large-v3`, use a GPU-capable host rather than Community Cloud.

### Option B — Deploy to Hugging Face Spaces

Hugging Face supports Streamlit as a Space SDK and installs Python dependencies from `requirements.txt`; it also supports `packages.txt` for `apt-get` dependencies. This project’s `README.md` already has the required `sdk: streamlit` metadata. Do **not** add a Streamlit port override: Streamlit Spaces use port 8501.

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Streamlit** as the SDK. Select **Private** if the uploaded videos or voice audio must not be public.
3. In the Space settings, select hardware appropriate for `large-v3`; choose an NVIDIA GPU option for real long-video work rather than CPU Basic.
4. Push this directory to the new Space repository:

   ```bash
   cd recap-video-processor
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```

   If the Space uses a different default branch, push to that branch instead.

5. Wait for the build to complete, then open the **App** tab or the Space URL on your phone.
6. Upload a short test, generate an SRT, render one output, and verify the Burmese font and mask-band position before processing long footage.

For Spaces that restart frequently, expect the Whisper model cache to be recreated unless persistent storage is configured on the selected hardware plan. Plan enough disk headroom for the input, intermediate master, raw segments, final clips, ZIP, and model weights at the same time.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `FFmpeg and FFprobe are required` | Install the packages in `packages.txt` and restart Streamlit. |
| `does not include the subtitles filter` | The host FFmpeg lacks libass. Use a full FFmpeg build that includes the `subtitles` filter. |
| Burmese characters display as boxes or wrong glyphs | Install `fonts-noto-core`, select `Noto Sans Myanmar`, then test a short clip. |
| Whisper runs out of memory / is too slow | Keep `base` on Community Cloud; try `small` only if available memory allows it. `medium`, `turbo`, and `large-v3` need a GPU-capable host. |
| Original burned-in subtitles remain visible | Increase **Cover band height**, set its opacity near 100%, or move the subtitle position. The band is designed for lower-third material; it cannot erase content elsewhere without masking that area too. |
| Custom audio ends before the video | The app preserves video duration and leaves the remaining output silent. Provide a custom audio file that matches the source duration. |
| Uploaded file is rejected | The app config requests a 4 GB limit, but a cloud provider, browser, proxy, or account tier can impose a lower limit. Use a smaller input or a host with a higher upload allowance. |
| Subtitle timing is off | Make sure the edited/uploaded SRT is timed to the **original**, unsped-up source. Do not pre-adjust it to 1.05×. |

## Privacy and retention

The app does not send video frames or audio to the OpenAI API. Whisper weights run on the machine hosting Streamlit. However, uploaded media and model cache exist on that host while processing. Choose a private deployment and review the storage policy of the hosting provider before uploading sensitive footage or voice-cloned audio.

## Source references

- [Streamlit Community Cloud dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Hugging Face Streamlit Spaces documentation](https://huggingface.co/docs/hub/en/spaces-sdks-streamlit)
- [Hugging Face Spaces dependency documentation](https://huggingface.co/docs/hub/en/spaces-dependencies)
