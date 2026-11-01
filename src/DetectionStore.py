"""SQLite-backed detection results; embedding vectors remain in EmbeddingCache."""

from contextlib import closing, contextmanager
import os
from pathlib import Path
import sqlite3

from src.Detection import Detection, DetectionRun, _identity, _score
from src.EmbeddingCorpus import EmbeddingRef, EmbeddingSpaceKey


class _DetectionRunWriter:
    """Write one scan in a transaction; never expose incomplete results."""

    def __init__(self, connection, run):
        self._connection = connection
        self.run = run
        self.scored_count = 0
        self.detection_count = 0

    def add_batch(self, detections, *, scanned_count):
        """Persist one bounded scan batch, including its zero-detection cases.

        Each item must be a ``Detection`` for this run with a score at or
        above the run's threshold. Duplicates are rejected, not overwritten.
        """
        scanned_count = int(scanned_count)
        if scanned_count < 0:
            raise ValueError("scanned_count cannot be negative")

        def rows():
            for detection in detections:
                if not isinstance(detection, Detection):
                    raise TypeError("detections must contain Detection objects")
                if detection.run_id != self.run.run_id:
                    raise ValueError("detection belongs to a different run")
                if detection.score < self.run.threshold:
                    raise ValueError("detection score is below the scan threshold")
                yield (
                    detection.run_id,
                    detection.ref.recording_id,
                    detection.ref.chunk_index,
                    detection.score,
                )

        cursor = self._connection.executemany(
            "INSERT INTO detections (run_id, recording_id, chunk_index, score) "
            "VALUES (?, ?, ?, ?)",
            rows(),
        )
        added = cursor.rowcount
        if added > scanned_count:
            raise ValueError("batch contains more detections than scored embeddings")
        self.scored_count += scanned_count
        self.detection_count += added


class DetectionStore:
    """Query completed detection runs in bounded pages using indexed SQL.

    Each operation uses its own connection; one writer can scan on a background
    thread while readers browse previously completed runs in the GUI thread.
    SQLite WAL mode permits those operations to overlap.
    """

    def __init__(self, path=None):
        self.path = Path(path or self.default_path()).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS detection_runs (
                    run_id TEXT PRIMARY KEY,
                    classifier_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    pipeline_hash TEXT NOT NULL,
                    embedding_name TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    scored_count INTEGER NOT NULL,
                    detection_count INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_detection_runs_classifier
                    ON detection_runs(classifier_id, created_at DESC, run_id DESC);

                CREATE TABLE IF NOT EXISTS detections (
                    run_id TEXT NOT NULL REFERENCES detection_runs(run_id)
                        ON DELETE CASCADE,
                    recording_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    score REAL NOT NULL,
                    PRIMARY KEY (run_id, recording_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_detections_score
                    ON detections(run_id, score DESC, recording_id, chunk_index);
                """
            )

    @staticmethod
    def default_path():
        root = os.environ.get("XDG_DATA_HOME")
        if root:
            return Path(root) / "caracal" / "detections.sqlite3"
        return Path.home() / ".local" / "share" / "caracal" / "detections.sqlite3"

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection

    @contextmanager
    def write_run(self, run: DetectionRun):
        """Atomically create a completed scan, or roll everything back.

        Usage::

            with store.write_run(run) as writer:
                writer.add_batch(detections, scanned_count=len(batch))

        Results become visible only after a successful exit. Any exception
        (including a cancelled job) rolls back both run and detections.
        """
        if not isinstance(run, DetectionRun):
            raise TypeError("run must be a DetectionRun")
        if run.scored_count or run.detection_count:
            raise ValueError("new runs must have zero counts")
        with self._connect() as connection:
            with connection:
                connection.execute(
                    "INSERT INTO detection_runs VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)",
                    (
                        run.run_id,
                        run.classifier_id,
                        run.model_id,
                        run.embedding_space.pipeline_hash,
                        run.embedding_space.embedding_name,
                        run.threshold,
                        run.created_at,
                    ),
                )
                writer = _DetectionRunWriter(connection, run)
                yield writer
                connection.execute(
                    "UPDATE detection_runs SET scored_count = ?, "
                    "detection_count = ? WHERE run_id = ?",
                    (writer.scored_count, writer.detection_count, run.run_id),
                )

    def get_run(self, run_id):
        run_id = _identity(run_id, "run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM detection_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._run_from_row(row) if row is not None else None

    def list_runs(self, *, classifier_id=None, limit=100, offset=0):
        limit, offset = self._page(limit, offset)
        query = "SELECT * FROM detection_runs"
        parameters = []
        if classifier_id is not None:
            query += " WHERE classifier_id = ?"
            parameters.append(_identity(classifier_id, "classifier_id"))
        query += " ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._run_from_row(row) for row in rows)

    def count_detections(self, run_id, *, min_score=None, recording_id=None):
        where, parameters = self._filters(run_id, min_score, recording_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM detections" + where, parameters
            ).fetchone()
        return int(row[0])

    def list_detections(
        self, run_id, *, min_score=None, recording_id=None,
        sort_by="score", descending=True, limit=200, offset=0,
    ):
        """Return one deterministic page, without loading the full result set."""
        limit, offset = self._page(limit, offset)
        where, parameters = self._filters(run_id, min_score, recording_id)
        if sort_by not in ("score", "recording"):
            raise ValueError("sort_by must be 'score' or 'recording'")
        direction = "DESC" if descending else "ASC"
        if sort_by == "score":
            order = f"score {direction}, recording_id ASC, chunk_index ASC"
        else:
            order = f"recording_id {direction}, chunk_index ASC, score DESC"
        query = (
            "SELECT run_id, recording_id, chunk_index, score FROM detections"
            + where + " ORDER BY " + order + " LIMIT ? OFFSET ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, (*parameters, limit, offset)).fetchall()
        return tuple(
            Detection(
                run_id=row["run_id"],
                ref=EmbeddingRef(row["recording_id"], row["chunk_index"]),
                score=row["score"],
            )
            for row in rows
        )

    @staticmethod
    def _page(limit, offset):
        limit, offset = int(limit), int(offset)
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset nonnegative")
        return limit, offset

    @staticmethod
    def _filters(run_id, min_score, recording_id):
        where = " WHERE run_id = ?"
        parameters = [_identity(run_id, "run_id")]
        if min_score is not None:
            where += " AND score >= ?"
            parameters.append(_score(min_score, "min_score"))
        if recording_id is not None:
            where += " AND recording_id = ?"
            parameters.append(str(recording_id))
        return where, tuple(parameters)

    @staticmethod
    def _run_from_row(row):
        return DetectionRun(
            run_id=row["run_id"],
            classifier_id=row["classifier_id"],
            model_id=row["model_id"],
            embedding_space=EmbeddingSpaceKey(
                pipeline_hash=row["pipeline_hash"],
                embedding_name=row["embedding_name"],
            ),
            threshold=row["threshold"],
            created_at=row["created_at"],
            scored_count=row["scored_count"],
            detection_count=row["detection_count"],
        )
