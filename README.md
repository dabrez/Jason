# Video-Education
# ACM_Hackathon

This project turns a long video into short clips. It uses
[OpenAI Whisper](https://github.com/openai/whisper) to transcribe audio, then
picks clip boundaries by fusing several highlight signals:

- **Chat spikes** (Twitch VODs only): viewers chatting more = something
  interesting is happening.
- **Audio energy**: loud/energetic moments (laughter, shouting, big reactions)
  -- works for any source, including plain podcasts with no chat at all.
- **Semantic hooks**: transcript moments with questions, exclamations,
  emphatic language, or a sharp topic/tone shift.

These signals are combined into one ranked list of candidate clips (see
`highlights/fusion.py`), each snapped to the nearest sentence boundary in the
transcript. If no source has a clear highlight anywhere (e.g. a very quiet,
low-variation video), clips fall back to topic-based chaptering instead (via
sentence embeddings + clustering).

Each clip can optionally also be reformatted for vertical platforms
(Shorts/Reels/TikTok): a 9:16 crop that follows the detected speaker's face
(OpenCV), plus word-by-word burned-in captions (TikTok/Reels style, generated
from Whisper's word-level timestamps). See `reformat/`.

See `roadmap.txt` for an Adobe Premiere plugin and other future platforms.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### ffmpeg

Clip cutting, audio-energy analysis, and vertical reformatting rely on the
`ffmpeg` command-line tool -- for caption burn-in specifically, it needs to be
built with `libass` support (most package-manager and static builds are).
If your package manager is unavailable, download a static build from
[the ffmpeg website](https://ffmpeg.org/download.html), extract it, and
place the `ffmpeg` binary somewhere on your `PATH`.
You can verify the installation with:

```bash
ffmpeg -version
```

and confirm `libass` support with:

```bash
ffmpeg -version | grep enable-libass
```

### Twitch VOD support (optional)

To pull Twitch VODs and their chat replay (used for the chat-spike signal),
you'll additionally need:

1. **A Twitch API app**, for VOD metadata lookups. Register one at
   [dev.twitch.tv/console/apps](https://dev.twitch.tv/console/apps), then set:

   ```bash
   export TWITCH_CLIENT_ID=your_client_id
   export TWITCH_CLIENT_SECRET=your_client_secret
   ```

2. **[TwitchDownloaderCLI](https://github.com/lay295/TwitchDownloader)**, for
   downloading chat replay (Twitch has no public REST endpoint for this).
   Download a release for your platform and ensure the `TwitchDownloaderCLI`
   binary is on your `PATH`.

Without these, YouTube links still work (using audio + semantic signals only);
Twitch links will raise a clear error explaining what's missing.

### Ollama-based re-ranking (optional)

By default, highlight windows are ranked purely by the local chat/audio/
semantic scoring signals -- no LLM or API key required. You can optionally
have a locally-running [Ollama](https://ollama.com) model re-rank the
shortlist for a more semantically-aware pass (see
`highlights.rank_with_ollama` / `ClipPipeline.run(use_ollama=True)`):

```bash
ollama pull gpt-oss
```

Ollama runs entirely locally, so this adds no external API cost. If Ollama
isn't running or reachable, the pipeline silently keeps the original
heuristic ranking rather than failing.

## Usage

Run the script and provide a YouTube or Twitch VOD link:

```bash
python main.py
```

The program downloads the video, transcribes the audio with Whisper, picks
clip boundaries from the fused highlight signals (falling back to topic
chapters if nothing stands out), and saves each clip in the `segments`
folder. It will also ask whether to generate vertical (Shorts/Reels/TikTok)
versions of each clip -- these are saved alongside the originals with a
`_vertical` suffix.

The face-tracking crop downloads a small (~10MB) OpenCV DNN face-detection
model to a local cache on first use; if that download fails (e.g. no network
access), it falls back to OpenCV's bundled Haar cascade detector. If no face
is ever detected in a clip, the crop falls back to a static center crop
instead of failing.
