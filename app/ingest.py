"""Import images into the gallery store (copy + thumbnail + queue)."""
import hashlib
import logging
import shutil
import time
from pathlib import Path

from PIL import Image, ImageOps

from . import config, db, worker

log = logging.getLogger("gallery.ingest")

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_thumbnail(src: Path, sha: str) -> None:
    out = config.THUMB_DIR / f"{sha}.jpg"
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((config.THUMB_MAX_SIDE, config.THUMB_MAX_SIDE), Image.LANCZOS)
        im.save(out, "JPEG", quality=85, optimize=True)


def import_file(src: Path) -> dict:
    """Copy one file into the store and enqueue ML analysis. Idempotent (sha1)."""
    src = Path(src)
    if src.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"unsupported file type: {src.name}")
    sha = sha1_file(src)
    existing = db.one("SELECT id FROM photos WHERE sha1=?", (sha,))
    if existing:
        return {"id": existing["id"], "duplicate": True}

    dest = config.PHOTO_DIR / sha[:2] / f"{sha}{src.suffix.lower()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)

    with Image.open(dest) as im:
        w, h = im.size
    taken = None
    from . import pipeline

    taken = pipeline.read_taken_at(str(dest)) or src.stat().st_mtime
    _make_thumbnail(dest, sha)

    pid = db.execute(
        "INSERT INTO photos(sha1, path, width, height, taken_at, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (sha, str(dest.relative_to(config.PHOTO_DIR)), w, h, taken, time.time()),
    )
    worker.submit(pid)
    return {"id": pid, "duplicate": False}


def scan_dir(path: str | Path) -> dict:
    """Recursively import every supported image under ``path``."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    imported, skipped = 0, 0
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for f in files:
        try:
            res = import_file(f)
            skipped += int(res["duplicate"])
            imported += int(not res["duplicate"])
        except Exception:
            log.exception("failed to import %s", f)
    log.info("scan %s: imported=%d duplicates=%d", root, imported, skipped)
    return {"imported": imported, "duplicates": skipped, "found": len(files)}
