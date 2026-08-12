"""Real API for the clip pipeline: submit a video link as a job, poll for
status, fetch results. Replaces the old single-shot synchronous /process
route with async job submission (see jobs.py) so a request doesn't block on
however long transcription + scoring + cutting takes.

Mounted as a Flask Blueprint so it composes with flaskGUI.py's separate
quiz-generation route rather than replacing that file.
"""
from flask import Blueprint, jsonify, request, send_file

from jobs import JobRunner, JobStore

clips_api = Blueprint("clips_api", __name__)
_store = JobStore()
_runner = JobRunner(_store)


@clips_api.route("/api/jobs", methods=["POST"])
def submit_job():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify(error="url is required"), 400

    job = _runner.submit(
        url=url,
        vertical=bool(data.get("vertical", False)),
        use_ollama=bool(data.get("use_ollama", False)),
        multi_speaker=bool(data.get("multi_speaker", True)),
    )
    return jsonify(job.to_dict()), 202


@clips_api.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = _store.get(job_id)
    if job is None:
        return jsonify(error="job not found"), 404
    return jsonify(job.to_dict())


@clips_api.route("/api/jobs", methods=["GET"])
def list_jobs():
    return jsonify([job.to_dict() for job in _store.list()])


@clips_api.route("/api/jobs/<job_id>/clips/<int:index>", methods=["GET"])
def download_clip(job_id, index):
    job = _store.get(job_id)
    if job is None:
        return jsonify(error="job not found"), 404
    paths = job.clip_paths
    vertical = request.args.get("vertical", "false").lower() == "true"
    if vertical:
        paths = job.vertical_paths
    if index < 0 or index >= len(paths):
        return jsonify(error="clip index out of range"), 404
    return send_file(paths[index])
