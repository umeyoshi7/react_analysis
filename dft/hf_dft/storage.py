"""ローカルパスと gs:// の両方を同じ関数で読み書きする."""
from __future__ import annotations

from pathlib import Path


def is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")


def join(base: str, *parts: str) -> str:
    base = str(base)
    if is_gcs(base):
        return "/".join([base.rstrip("/"), *[p.strip("/") for p in parts]])
    return str(Path(base, *parts))


def _split(uri: str) -> tuple[str, str]:
    bucket, _, blob = uri[len("gs://"):].partition("/")
    return bucket, blob


def _client():
    from google.cloud import storage
    return storage.Client()


def exists(path: str) -> bool:
    if is_gcs(path):
        bucket, blob = _split(path)
        return _client().bucket(bucket).blob(blob).exists()
    return Path(path).exists()


def read_text(path: str) -> str:
    if is_gcs(path):
        bucket, blob = _split(path)
        return _client().bucket(bucket).blob(blob).download_as_text(encoding="utf-8")
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    if is_gcs(path):
        bucket, blob = _split(path)
        _client().bucket(bucket).blob(blob).upload_from_string(text.encode("utf-8"), content_type="text/plain; charset=utf-8")
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
