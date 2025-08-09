# Video-Education
# ACM_Hackathon

This project now uses [OpenAI Whisper](https://github.com/openai/whisper) to
generate transcripts for YouTube videos instead of the YouTube subtitle API.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### ffmpeg

The clip-generation step relies on the `ffmpeg` command-line tool.
If your package manager is unavailable, download a static build from
[the ffmpeg website](https://ffmpeg.org/download.html), extract it, and
place the `ffmpeg` binary somewhere on your `PATH`.
You can verify the installation with:

```bash
ffmpeg -version
```

## Usage

Run the script and provide a YouTube link:

```bash
python main.py
```

The program downloads the video, transcribes the audio with Whisper,
groups the transcript into topics, and saves each topic segment as a
separate clip in the `segments` folder.
