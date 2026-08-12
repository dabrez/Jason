"""Auto-crop a 16:9 (or any wide) clip to 9:16 by tracking faces.

Single-speaker approach: sample frames at a fixed interval, run OpenCV's DNN
face detector on each sample, and track the horizontal center of the largest
detected face over time. Gaps (no face detected) hold the last known
position; the whole track is then smoothed (moving average) to avoid jittery
panning.

Multi-speaker approach (see track_speakers): when more than one face is ever
detected, faces are tracked as persistent identities across samples (simple
centroid tracker), and an active-speaker signal -- mouth-aspect-ratio
variance per tracked face, gated by overall audio energy -- picks which
tracked face is talking at each sample. The crop follows that speaker's
x-center, switching only after a minimum dwell time so it doesn't whip-pan
on every syllable. Falls back to the single-face path when at most one face
is ever detected (cheaper, no landmark/audio work needed).

Either track is turned into a piecewise-linear ffmpeg `crop` expression
(interpolating `x` between samples via ffmpeg's built-in expression
evaluator), so cropping is done in a single ffmpeg pass rather than
frame-by-frame in Python.

If no face is ever detected (silent slideshow, gameplay-only footage, etc.),
falls back to a static center crop.
"""
import os
import shutil
import subprocess
import tempfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SAMPLE_INTERVAL_SECONDS = 0.5
SMOOTHING_WINDOW = 5
TARGET_ASPECT = 9 / 16

# Multi-speaker tuning.
TRACK_MATCH_MAX_DISTANCE_FRAC = 0.15  # of frame width, for centroid matching
TRACK_MAX_MISSES = 6  # ~3s at the default sample interval before a track is dropped
MAR_HISTORY_WINDOW = 6  # samples (~3s) of mouth-aspect-ratio history for variance
MIN_DWELL_SECONDS = 1.5  # minimum time before the active speaker can switch
AUDIO_GATE_WINDOW_SECONDS = 0.5

_DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_DNN_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
_LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"

# Mouth landmark indices in the standard 68-point scheme (cv2.face LBF model).
_MOUTH_LANDMARK_RANGE = range(48, 68)
_MOUTH_TOP = 51
_MOUTH_BOTTOM = 57
_MOUTH_LEFT = 48
_MOUTH_RIGHT = 54

_model_cache_dir = os.path.join(tempfile.gettempdir(), "clip_pipeline_face_model")


def _download_if_missing(url: str, dest: str):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _load_face_net():
    """Loads OpenCV's bundled SSD face detector, downloading the (small,
    ~10MB) model files to a local cache on first use. Falls back to the
    Haar cascade (bundled with opencv-python, no download needed) if the
    download fails, e.g. no network access.
    """
    try:
        prototxt = _download_if_missing(_DNN_PROTOTXT_URL, os.path.join(_model_cache_dir, "deploy.prototxt"))
        model = _download_if_missing(_DNN_MODEL_URL, os.path.join(_model_cache_dir, "res10_300x300.caffemodel"))
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        return ("dnn", net)
    except Exception:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return ("haar", cv2.CascadeClassifier(cascade_path))


def _load_facemark():
    """Loads OpenCV-contrib's LBF facemark model for mouth-landmark
    detection, downloading the (~50MB) pretrained model to the same local
    cache as the face detector on first use. Returns None if the facemark
    module isn't available (plain opencv-python instead of
    opencv-contrib-python) or the download fails -- callers should treat
    that as "no active-speaker signal available".
    """
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
        return None
    try:
        model_path = _download_if_missing(_LBF_MODEL_URL, os.path.join(_model_cache_dir, "lbfmodel.yaml"))
        facemark = cv2.face.createFacemarkLBF()
        facemark.loadModel(model_path)
        return facemark
    except Exception:
        return None


def _detect_all_faces(frame: np.ndarray, detector) -> List[Tuple[int, int, int, int]]:
    """Returns all detected face boxes as (x, y, w, h) in pixel coordinates,
    for either the DNN or Haar cascade detector.
    """
    kind, model = detector
    h, w = frame.shape[:2]
    boxes = []

    if kind == "dnn":
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        detections = model.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            if bw > 0 and bh > 0:
                boxes.append((x1, y1, bw, bh))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = [tuple(int(v) for v in f) for f in faces]

    return boxes


def _detect_largest_face_center(frame: np.ndarray, detector) -> Optional[Tuple[float, float]]:
    boxes = _detect_all_faces(frame, detector)
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return (x + w / 2, y + h / 2)


def track_face_centers(video_path: str, interval_seconds: float = SAMPLE_INTERVAL_SECONDS) -> List[Tuple[float, float]]:
    """Returns a list of (timestamp, x_center_fraction) samples, where
    x_center_fraction is the detected face's horizontal center as a fraction
    of frame width (0.0 = left edge, 1.0 = right edge). Gaps where no face
    was found hold the previous value; if no face is ever found, returns [].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video for face tracking: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * interval_seconds)))
    detector = _load_face_net()

    samples = []
    last_x_frac = None
    frame_idx = 0
    any_face_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            h, w = frame.shape[:2]
            center = _detect_largest_face_center(frame, detector)
            if center is not None:
                last_x_frac = center[0] / w
                any_face_found = True
            if last_x_frac is not None:
                samples.append((timestamp, last_x_frac))
        frame_idx += 1

    cap.release()
    return samples if any_face_found else []


# -- multi-speaker active-speaker tracking ---------------------------------

class _FaceTrack:
    def __init__(self, track_id: int, timestamp: float, box: Tuple[int, int, int, int]):
        self.id = track_id
        self.box = box
        self.last_seen = timestamp
        self.misses = 0
        self.mar_history: List[float] = []  # recent mouth-aspect-ratio samples
        self.samples: List[Tuple[float, float]] = []  # (timestamp, x_center_frac)
        self.mar_variance_samples: List[Tuple[float, float]] = []  # (timestamp, rolling MAR variance as of that time)

    def centroid(self) -> Tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)


def _match_track(track: _FaceTrack, box, frame_width) -> float:
    cx, cy = track.centroid()
    x, y, w, h = box
    bx, by = x + w / 2, y + h / 2
    return abs(cx - bx) / frame_width


def _update_tracks(tracks: List[_FaceTrack], boxes, timestamp: float, frame_width: int, next_id: List[int]) -> List[_FaceTrack]:
    """Greedy nearest-centroid matching of this frame's detected boxes
    against existing tracks. Unmatched tracks accumulate a miss and are
    dropped after TRACK_MAX_MISSES consecutive misses (handles people
    briefly turning away or occluding each other). Unmatched boxes start
    new tracks.
    """
    unmatched_boxes = list(boxes)
    for track in tracks:
        best_idx, best_dist = None, TRACK_MATCH_MAX_DISTANCE_FRAC
        for i, box in enumerate(unmatched_boxes):
            dist = _match_track(track, box, frame_width)
            if dist < best_dist:
                best_idx, best_dist = i, dist
        if best_idx is not None:
            track.box = unmatched_boxes.pop(best_idx)
            track.last_seen = timestamp
            track.misses = 0
        else:
            track.misses += 1

    tracks = [t for t in tracks if t.misses <= TRACK_MAX_MISSES]

    for box in unmatched_boxes:
        tracks.append(_FaceTrack(next_id[0], timestamp, box))
        next_id[0] += 1

    return tracks


def _mouth_aspect_ratio(landmarks: np.ndarray) -> Optional[float]:
    pts = landmarks[0]
    if len(pts) < 68:
        return None
    top = pts[_MOUTH_TOP]
    bottom = pts[_MOUTH_BOTTOM]
    left = pts[_MOUTH_LEFT]
    right = pts[_MOUTH_RIGHT]
    width = np.linalg.norm(left - right)
    if width == 0:
        return None
    height = np.linalg.norm(top - bottom)
    return float(height / width)


def _extract_audio_envelope(video_path: str, interval_seconds: float) -> List[float]:
    """Short-time RMS energy at the same cadence as frame sampling, used to
    gate the mouth-movement signal (a still face with a low audio floor is
    not "speaking", even if MAR jitters from detection noise).
    """
    import librosa

    if shutil.which("ffmpeg") is None:
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "22050", audio_path,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        hop_length = max(1, int(interval_seconds * sr))
        if len(y) == 0:
            return []
        rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
        return [float(v) for v in rms]
    finally:
        os.remove(audio_path)


def track_speakers(
    video_path: str,
    interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
) -> List[Tuple[float, float]]:
    """Multi-speaker version of track_face_centers: returns (timestamp,
    x_center_fraction) samples that follow whichever tracked face is
    actively speaking, switching only after MIN_DWELL_SECONDS. Falls back
    to the single-largest-face behavior (and to [] if no face is ever
    found) when at most one face track is ever observed, or when the
    facemark landmark model isn't available.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video for face tracking: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps * interval_seconds)))
    detector = _load_face_net()
    facemark = _load_facemark()

    tracks: List[_FaceTrack] = []
    next_id = [0]
    frame_width = None
    frame_idx = 0
    max_concurrent_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            h, w = frame.shape[:2]
            frame_width = w
            boxes = _detect_all_faces(frame, detector)
            max_concurrent_faces = max(max_concurrent_faces, len(boxes))
            tracks = _update_tracks(tracks, boxes, timestamp, w, next_id)

            mar_by_track = {}
            if facemark is not None and boxes:
                try:
                    ok, landmarks_list = facemark.fit(frame, np.array([t.box for t in tracks if t.misses == 0], dtype=np.int32))
                except Exception:
                    ok, landmarks_list = False, []
                if ok:
                    active_tracks = [t for t in tracks if t.misses == 0]
                    for track, landmarks in zip(active_tracks, landmarks_list):
                        mar = _mouth_aspect_ratio(landmarks)
                        if mar is not None:
                            mar_by_track[track.id] = mar

            for track in tracks:
                if track.misses == 0:
                    x, y, w_box, h_box = track.box
                    track.samples.append((timestamp, (x + w_box / 2) / w))
                    mar = mar_by_track.get(track.id)
                    track.mar_history.append(mar if mar is not None else 0.0)
                    if len(track.mar_history) > MAR_HISTORY_WINDOW:
                        track.mar_history.pop(0)
                    variance = float(np.var(track.mar_history)) if len(track.mar_history) >= 2 else 0.0
                    track.mar_variance_samples.append((timestamp, variance))

        frame_idx += 1

    cap.release()

    if max_concurrent_faces <= 1 or facemark is None:
        return track_face_centers(video_path, interval_seconds)

    return _resolve_active_speaker(video_path, tracks, interval_seconds)


def _resolve_active_speaker(video_path: str, tracks: List[_FaceTrack], interval_seconds: float) -> List[Tuple[float, float]]:
    """Combines per-track mouth-movement variance with an overall audio
    envelope to pick the active speaker at each sampled timestamp, then
    applies a minimum-dwell-time hysteresis so the crop doesn't switch on
    every syllable.
    """
    all_timestamps = sorted({t for track in tracks for t, _ in track.samples})
    if not all_timestamps:
        return []

    audio_envelope = _extract_audio_envelope(video_path, interval_seconds)
    audio_floor = (sum(audio_envelope) / len(audio_envelope) * 0.5) if audio_envelope else 0.0

    def audio_at(timestamp: float) -> float:
        if not audio_envelope:
            return 1.0  # no audio signal available -- don't gate
        idx = min(len(audio_envelope) - 1, int(timestamp / interval_seconds))
        return audio_envelope[idx]

    x_by_track_at_time = {
        track.id: dict(track.samples) for track in tracks
    }
    mar_variance_by_track_at_time = {
        track.id: dict(track.mar_variance_samples) for track in tracks
    }

    def mar_variance_at(track: _FaceTrack, timestamp: float) -> float:
        # Rolling-window MAR variance as of this exact sample timestamp
        # (recorded during collection), not the track's final variance --
        # using the final value would let later frames leak into the
        # decision for earlier ones.
        return mar_variance_by_track_at_time.get(track.id, {}).get(timestamp, 0.0)

    active_speaker_id = None
    active_since = None
    result = []

    for timestamp in all_timestamps:
        live_tracks = [t for t in tracks if timestamp in x_by_track_at_time[t.id]]
        if not live_tracks:
            continue

        gated = audio_at(timestamp) >= audio_floor
        if gated:
            candidate = max(live_tracks, key=lambda t: mar_variance_at(t, timestamp))
        else:
            candidate = None

        if candidate is not None and candidate.id != active_speaker_id:
            if active_speaker_id is None or active_since is None or (timestamp - active_since) >= MIN_DWELL_SECONDS:
                active_speaker_id = candidate.id
                active_since = timestamp
        elif active_speaker_id is None and live_tracks:
            active_speaker_id = live_tracks[0].id
            active_since = timestamp

        if active_speaker_id is not None and active_speaker_id in x_by_track_at_time and timestamp in x_by_track_at_time[active_speaker_id]:
            result.append((timestamp, x_by_track_at_time[active_speaker_id][timestamp]))
        elif live_tracks:
            # Active speaker not visible at this instant (brief miss); hold
            # on the largest currently-visible face rather than jumping.
            fallback = max(live_tracks, key=lambda t: t.box[2] * t.box[3])
            result.append((timestamp, x_by_track_at_time[fallback.id][timestamp]))

    return result


def _smooth(values: List[float], window: int = SMOOTHING_WINDOW) -> List[float]:
    if len(values) < 2:
        return values
    arr = np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid"))


def _write_crop_sendcmd_file(
    samples: List[Tuple[float, float]],
    src_width: int,
    crop_width: int,
    duration_seconds: float,
    fps: float,
) -> str:
    """Writes an ffmpeg `sendcmd` script that sets crop's `x` parameter once
    per output frame, linearly interpolated between sampled face-center
    positions (via numpy, not ffmpeg's expression parser).

    A single nested-if ffmpeg expression (the previous approach) doesn't
    scale: ffmpeg's expression parser fails to configure the filter once the
    expression gets too long/deeply nested, which real clips longer than
    ~30-40s at the default 0.5s sample interval hit in practice (confirmed
    against real footage -- ffmpeg accepts up to ~80 nested `if()` clauses
    but rejects 100+). sendcmd has no such limit since it reads timed
    commands from a file line-by-line rather than compiling one expression,
    so per-frame granularity for clips of any length is safe.
    """
    max_x = max(0, src_width - crop_width)

    def x_for_frac(frac: float) -> float:
        return min(max_x, max(0, frac * src_width - crop_width / 2))

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sendcmd") as tmp:
        sendcmd_path = tmp.name
        if not samples:
            tmp.write(f"0.0 crop x {max_x / 2:.2f};\n")
        elif len(samples) == 1:
            tmp.write(f"0.0 crop x {x_for_frac(samples[0][1]):.2f};\n")
        else:
            sample_ts = np.array([t for t, _ in samples])
            sample_xs = np.array([x_for_frac(frac) for _, frac in samples])
            n_frames = max(1, int(round(duration_seconds * fps)))
            frame_ts = np.arange(n_frames) / fps
            interpolated = np.interp(frame_ts, sample_ts, sample_xs)
            for t, x in zip(frame_ts, interpolated):
                tmp.write(f"{t:.4f} crop x {x:.2f};\n")

    return sendcmd_path


def crop_to_vertical(
    video_path: str,
    output_path: Optional[str] = None,
    interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
    multi_speaker: bool = True,
) -> str:
    """Crops a video to 9:16 (portrait), following the active speaker's face
    horizontally over time. With `multi_speaker` (default), uses
    track_speakers -- mouth-movement + audio gating to pick which of
    several visible faces is talking, with switch hysteresis -- and falls
    back automatically to single-largest-face tracking when at most one
    face is ever detected. Falls back to a static center crop if no face is
    detected anywhere in the clip.
    """
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is required for vertical reformatting. "
            "Install it from https://ffmpeg.org/ and ensure it is on your PATH."
        )
    if output_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix="_vertical.mp4") as tmp:
            output_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_seconds = (frame_count / fps) if frame_count else 0.0
    cap.release()

    crop_width = int(src_height * TARGET_ASPECT)
    if crop_width > src_width:
        crop_width = src_width

    if multi_speaker:
        samples = track_speakers(video_path, interval_seconds)
    else:
        samples = track_face_centers(video_path, interval_seconds)
    smoothed = list(zip(
        [t for t, _ in samples],
        _smooth([frac for _, frac in samples]),
    )) if samples else []

    sendcmd_path = _write_crop_sendcmd_file(smoothed, src_width, crop_width, duration_seconds, fps)
    try:
        # Escape the same way captions.py escapes its ASS path: ffmpeg's
        # filtergraph treats ':' as an option separator.
        escaped_sendcmd_path = sendcmd_path.replace("\\", "/").replace(":", "\\:")
        initial_x = max(0, src_width - crop_width) // 2
        # sendcmd must precede the filter it targets in the chain -- it
        # attaches commands to each frame's side data as the frame passes
        # through, which the downstream crop@1 (named so sendcmd's per-line
        # `crop@1 x <value>` commands can address it) then reads. Verified
        # empirically: crop@1 before sendcmd applies each command one frame
        # late (confirmed via a pixel-value probe), a one-frame lag that's
        # invisible in a diff of the command list but visibly wrong in the
        # actual output.
        crop_filter = (
            f"sendcmd=f={escaped_sendcmd_path},"
            f"crop@1={crop_width}:{src_height}:{initial_x}:0"
        )

        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", crop_filter,
            "-c:a", "copy",
            output_path,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg crop failed:\n{result.stderr[-2000:]}")
    finally:
        os.remove(sendcmd_path)

    return output_path
