"""SQLite persistence layer (stdlib sqlite3, one connection per thread)."""
import sqlite3
import threading
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id         INTEGER PRIMARY KEY,
    sha1       TEXT UNIQUE NOT NULL,
    path       TEXT NOT NULL,
    width      INTEGER NOT NULL DEFAULT 0,
    height     INTEGER NOT NULL DEFAULT 0,
    taken_at   REAL,
    processed  INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
    id        INTEGER PRIMARY KEY,
    photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    box       TEXT NOT NULL,
    score     REAL NOT NULL DEFAULT 0,
    face_emb  BLOB,
    person_id INTEGER,
    reviewed  INTEGER,
    quality   REAL,
    yaw       REAL,
    blur      REAL,            -- Ente blurValue (Laplacian variance)
    sideways  INTEGER          -- Ente faceDirection != "straight"
);
CREATE TABLE IF NOT EXISTS rejected_faces (   -- Ente per-(face x person) rejection memory
    face_key  TEXT NOT NULL,
    person_id INTEGER NOT NULL,
    PRIMARY KEY (face_key, person_id)
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);

CREATE TABLE IF NOT EXISTS detections (   -- person (body) detections
    id         INTEGER PRIMARY KEY,
    photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    box        TEXT NOT NULL,
    conf       REAL NOT NULL DEFAULT 0,
    face_id    INTEGER,
    body_emb   BLOB,
    person_id  INTEGER,
    score      REAL,
    reason     TEXT,
    personness REAL,
    head       TEXT,                -- pose head (x,y,r photo px) for face validation
    track_id   INTEGER              -- burst track (physical continuity)
    margin     REAL
);
CREATE INDEX IF NOT EXISTS idx_det_photo ON detections(photo_id);
CREATE INDEX IF NOT EXISTS idx_det_person ON detections(person_id);

CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY,
    name       TEXT,
    created_at REAL NOT NULL,
    pinned     INTEGER NOT NULL DEFAULT 0   -- Ente isPinned: pinned people sort first
);

CREATE TABLE IF NOT EXISTS labels (    -- manual ground truth (training set)
    id         INTEGER PRIMARY KEY,
    photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,          -- 'face' | 'det'
    ref_id     INTEGER NOT NULL,
    action     TEXT NOT NULL,          -- 'assign' | 'new' | 'unassign'
    person_id  INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_photo ON labels(photo_id);

CREATE TABLE IF NOT EXISTS review_pairs (  -- merge-review decisions
    a          INTEGER NOT NULL,
    b          INTEGER NOT NULL,
    status     TEXT NOT NULL,             -- 'rejected' | 'merged'
    updated_at REAL NOT NULL,
    PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS review_rejects (  -- "not this person" memory
    target_person_id INTEGER NOT NULL,
    other_person_id  INTEGER,
    emb              BLOB NOT NULL,        -- unit centroid of what was rejected
    ts               REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS review_log (   -- undoable review history
    id         INTEGER PRIMARY KEY,
    action     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS review_dets (  -- back-view review decisions
    person_id  INTEGER NOT NULL,
    det_id     INTEGER NOT NULL,
    status     TEXT NOT NULL,             -- 'rejected' | 'merged'
    updated_at REAL NOT NULL,
    PRIMARY KEY (person_id, det_id)
);
"""

_local = threading.local()


def conn() -> sqlite3.Connection:
    """Return this thread's sqlite connection (creates it on first use)."""
    c = getattr(_local, "conn", None)
    if c is None:
        config.ensure_dirs()
        c = sqlite3.connect(str(config.DB_PATH), timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        _local.conn = c
    return c


def init_db() -> None:
    c = conn()
    c.executescript(SCHEMA)
    # migrations for databases created before newer columns existed
    cols = {r[1] for r in c.execute("PRAGMA table_info(detections)")}
    if "personness" not in cols:
        c.execute("ALTER TABLE detections ADD COLUMN personness REAL")
    if "head" not in cols:
        c.execute("ALTER TABLE detections ADD COLUMN head TEXT")
    if "track_id" not in cols:
        c.execute("ALTER TABLE detections ADD COLUMN track_id INTEGER")
    if "margin" not in cols:
        c.execute("ALTER TABLE detections ADD COLUMN margin REAL")
    fcols = {r[1] for r in c.execute("PRAGMA table_info(faces)")}
    if "reviewed" not in fcols:
        c.execute("ALTER TABLE faces ADD COLUMN reviewed INTEGER")
    if "quality" not in fcols:
        c.execute("ALTER TABLE faces ADD COLUMN quality REAL")
    if "yaw" not in fcols:
        c.execute("ALTER TABLE faces ADD COLUMN yaw REAL")
    if "blur" not in fcols:
        c.execute("ALTER TABLE faces ADD COLUMN blur REAL")
    if "sideways" not in fcols:
        c.execute("ALTER TABLE faces ADD COLUMN sideways INTEGER")
    pcols = {r[1] for r in c.execute("PRAGMA table_info(persons)")}
    if "pinned" not in pcols:
        c.execute("ALTER TABLE persons ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
    c.commit()


def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return conn().execute(sql, params).fetchall()


def one(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    return conn().execute(sql, params).fetchone()


def execute(sql: str, params: tuple = ()) -> int:
    c = conn()
    cur = c.execute(sql, params)
    c.commit()
    return cur.lastrowid


def purge_person_refs(person_ids, move_labels_to: int | None = None) -> None:
    """Drop/neutralise everything that refers to persons being deleted.

    ``persons.id`` is a rowid alias, so SQLite hands a deleted id to the next
    INSERT - left behind, a dead person's labels and review decisions would
    silently attach to whoever is created next (wrong identity in the labels
    export and in the decided-review history). Labels are user work: on a
    merge they follow the person into its survivor, otherwise they are
    unlinked rather than deleted.
    """
    ids = [int(p) for p in person_ids]
    if not ids:
        return
    q = ",".join("?" * len(ids))
    c = conn()
    if move_labels_to is not None:
        c.execute(f"UPDATE labels SET person_id=? WHERE person_id IN ({q})",
                  [int(move_labels_to), *ids])
    else:
        c.execute(f"UPDATE labels SET person_id=NULL WHERE person_id IN ({q})",
                  ids)
    c.execute(f"DELETE FROM review_pairs WHERE a IN ({q}) OR b IN ({q})",
              ids + ids)
    c.execute(f"DELETE FROM review_dets WHERE person_id IN ({q})", ids)
    c.commit()


def now() -> float:
    return time.time()


def blob_to_vec(blob) -> "list[float] | None":
    import numpy as np

    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def vec_to_blob(vec) -> bytes:
    import numpy as np

    return np.asarray(vec, dtype=np.float32).tobytes()
