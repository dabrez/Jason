"""CLI entrypoint: paste a YouTube or Twitch VOD link, get short clips back.

Clip boundaries come from fusing chat-spike (Twitch only), audio-energy, and
transcript semantic-hook signals, falling back to topic segmentation if
nothing stands out. See roadmap.txt for planned platforms and features (an
Adobe Premiere plugin, etc).
"""
from pipeline import ClipPipeline
from sources import resolve_source


# Kept for backwards compatibility with flaskGUI.py's quiz-generation flow,
# which only needs the transcript, not the full clip pipeline.
class VideoTranscript:
    def __init__(self, link: str):
        self.link = link
        self._pipeline = ClipPipeline(resolve_source(link))

    def check_youtube_link(self, link):
        return self._pipeline.source.is_valid()

    def getVideoText(self, link=None):
        return self._pipeline.transcribe()


if __name__ == "__main__":
    link = input("Enter your YouTube or Twitch VOD link here: ")
    make_vertical = input("Also generate vertical (Shorts/Reels/TikTok) versions? [y/N]: ").strip().lower() == "y"
    try:
        source = resolve_source(link)
    except ValueError as e:
        print(e)
    else:
        pipeline = ClipPipeline(source)
        output_dir, clips, clip_paths, vertical_paths = pipeline.run(vertical=make_vertical)
        print(f"Saved {len(clips)} clips to {output_dir}")
        if vertical_paths:
            print(f"Saved {len(vertical_paths)} vertical versions alongside them")
